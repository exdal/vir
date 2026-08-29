use std::collections::BTreeMap;

use ash::vk;

#[derive(Debug)]
pub(crate) struct DescriptorArena {
    pools: Vec<vk::DescriptorPool>,
    max_sets: u32,
    totals: BTreeMap<i32, (vk::DescriptorType, u32)>,
}

impl DescriptorArena {
    pub(crate) fn create(
        device: &ash::Device, max_sets: u32, totals: &BTreeMap<i32, (vk::DescriptorType, u32)>,
    ) -> Result<Option<Self>, vk::Result> {
        if max_sets == 0 {
            return Ok(None);
        }

        let pool = Self::create_pool(device, max_sets, totals)?;

        Ok(Some(Self {
            pools: vec![pool],
            max_sets,
            totals: totals.clone(),
        }))
    }

    fn create_pool(
        device: &ash::Device, max_sets: u32, totals: &BTreeMap<i32, (vk::DescriptorType, u32)>,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        let sizes = totals
            .values()
            .filter(|(_, count)| *count > 0)
            .map(|(descriptor_type, count)| {
                vk::DescriptorPoolSize::default()
                    .ty(*descriptor_type)
                    .descriptor_count(*count)
            })
            .collect::<Vec<_>>();
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&sizes);
        unsafe { device.create_descriptor_pool(&info, None) }
    }

    pub(crate) fn allocate(
        &mut self, device: &ash::Device, layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, vk::Result> {
        let layouts = [layout];
        let allocate = |pool| {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            unsafe { device.allocate_descriptor_sets(&info) }.map(|sets| sets[0])
        };

        match allocate(*self.pools.last().expect("an arena always owns one pool")) {
            Ok(set) => Ok(set),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                let pool = Self::create_pool(device, self.max_sets, &self.totals)?;
                self.pools.push(pool);
                allocate(pool)
            },
            Err(err) => Err(err),
        }
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        for pool in self.pools {
            unsafe { device.destroy_descriptor_pool(pool, None) };
        }
    }
}
