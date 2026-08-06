use std::collections::BTreeMap;

use ash::vk;

use crate::PipelineLayout;

/// A texture's slot in a [`RenderGraph`](crate::RenderGraph)'s bindless table.
///
/// This is the value a shader indexes its texture array with, so it is what a draw pushes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

impl std::fmt::Display for TextureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "@{}", self.0) }
}

/// Every descriptor set one pipeline layout declares, and the pool they came out of.
///
/// A pool per layout costs one object each but sizes itself exactly, so nothing has to guess
/// how many descriptors the pipelines a graph ends up holding will want between them.
#[derive(Debug)]
pub struct DescriptorSets {
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl DescriptorSets {
    pub fn sets(&self) -> &[vk::DescriptorSet] { &self.sets }

    pub fn set(&self, index: u32) -> Option<vk::DescriptorSet> { self.sets.get(index as usize).copied() }

    pub(crate) fn create(device: &ash::Device, layout: &PipelineLayout) -> Result<Option<Self>, vk::Result> {
        if layout.sets.is_empty() {
            return Ok(None);
        }

        let mut totals: BTreeMap<i32, (vk::DescriptorType, u32)> = BTreeMap::new();
        let mut update_after_bind = false;
        for set in &layout.sets {
            update_after_bind |= set.variable_count > 0;
            for (descriptor_type, count) in &set.sizes {
                let entry = totals.entry(descriptor_type.as_raw()).or_insert((*descriptor_type, 0));
                entry.1 += count;
            }
        }

        // a set with no bindings at all still has to come out of the pool
        let sizes = totals
            .values()
            .filter(|(_, count)| *count > 0)
            .map(|(descriptor_type, count)| {
                vk::DescriptorPoolSize::default()
                    .ty(*descriptor_type)
                    .descriptor_count(*count)
            })
            .collect::<Vec<_>>();

        let mut flags = vk::DescriptorPoolCreateFlags::empty();
        if update_after_bind {
            flags |= vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
        }

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(flags)
            .max_sets(layout.sets.len() as u32)
            .pool_sizes(&sizes);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

        let set_layouts = layout.sets.iter().map(|set| set.handle).collect::<Vec<_>>();
        let variable_counts = layout.sets.iter().map(|set| set.variable_count).collect::<Vec<_>>();

        let mut count_info =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&variable_counts);
        let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        if update_after_bind {
            alloc_info = alloc_info.push_next(&mut count_info);
        }

        let sets = match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets,
            Err(err) => {
                unsafe { device.destroy_descriptor_pool(pool, None) };
                return Err(err);
            },
        };

        Ok(Some(Self { pool, sets }))
    }

    pub(crate) fn destroy(&self, device: &ash::Device) { unsafe { device.destroy_descriptor_pool(self.pool, None) }; }
}
