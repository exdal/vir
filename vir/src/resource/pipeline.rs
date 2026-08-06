pub mod state;

use std::collections::BTreeMap;

use ash::vk::{self, Handle};
pub use state::{
    BlendPreset,
    ColorBlendAttachmentState,
    DynamicStateFlags,
    DynamicValues,
    PassState,
    PipelineState,
    PushConstants,
    RasterizationState,
    Rect2D,
    RenderingState,
    ResolvedViewport,
    StateChange,
    Viewport,
};

use crate::resource::shader::Reflection;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub(crate) u32);

impl std::fmt::Display for PipelineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "#{}", self.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

impl VertexLayout {
    pub fn interleaved(reflections: &[Reflection]) -> Self {
        let Some(vertex) = reflections
            .iter()
            .find(|reflection| reflection.stage == vk::ShaderStageFlags::VERTEX)
        else {
            return Self::default();
        };

        let mut offset = 0;
        let attributes = vertex
            .vertex_inputs
            .iter()
            .map(|input| {
                let attribute = VertexAttribute {
                    location: input.location,
                    format: input.format,
                    offset,
                };
                offset += input.size;
                attribute
            })
            .collect();

        Self {
            stride: offset,
            attributes,
        }
    }

    pub fn is_empty(&self) -> bool { self.attributes.is_empty() }
}

#[derive(Debug, Clone, Default)]
pub struct GraphicsPipelineInfo {
    pub shaders: Vec<Vec<u32>>,
}

impl GraphicsPipelineInfo {
    pub fn new() -> Self { Self::default() }

    pub fn with_shader(mut self, spirv: &[u32]) -> Self {
        self.shaders.push(spirv.to_vec());
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct ComputePipelineInfo {
    pub shader: Vec<u32>,
}

impl ComputePipelineInfo {
    pub fn new(spirv: &[u32]) -> Self { Self { shader: spirv.to_vec() } }
}

pub fn push_constant_ranges(reflections: &[Reflection]) -> Vec<vk::PushConstantRange> {
    let mut merged: BTreeMap<(u32, u32), vk::ShaderStageFlags> = BTreeMap::new();

    for reflection in reflections {
        if reflection.push_constant_size == 0 {
            continue;
        }

        *merged
            .entry((reflection.push_constant_offset, reflection.push_constant_size))
            .or_insert(vk::ShaderStageFlags::empty()) |= reflection.stage;
    }

    merged
        .into_iter()
        .map(|((offset, size), stages)| {
            vk::PushConstantRange::default()
                .stage_flags(stages)
                .offset(offset)
                .size(size)
        })
        .collect()
}

#[derive(Debug, Default, Clone)]
pub struct SetLayout {
    pub handle: vk::DescriptorSetLayout,
    /// How many descriptors of each type the set holds.
    pub sizes: Vec<(vk::DescriptorType, u32)>,
    /// The count the set's variable-count binding is allocated with, or zero when it has none.
    pub variable_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureBinding {
    pub set: u32,
    pub binding: u32,
}

#[derive(Debug, Default)]
pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,
    pub sets: Vec<SetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
    pub texture_binding: Option<TextureBinding>,
}

impl PipelineLayout {
    pub fn cover(&self, offset: u32, size: u32) -> impl Iterator<Item = (vk::ShaderStageFlags, u32, u32)> + '_ {
        let end = offset.saturating_add(size);
        self.push_constant_ranges.iter().filter_map(move |range| {
            let start = range.offset.max(offset);
            let stop = (range.offset + range.size).min(end);
            (stop > start).then(|| (range.stage_flags, start, stop - start))
        })
    }

    pub(crate) fn create(
        device: &ash::Device, reflections: &[Reflection], max_variable_descriptor_count: u32,
    ) -> Result<Self, vk::Result> {
        struct Merged {
            descriptor_type: vk::DescriptorType,
            count: u32,
            variable_count: bool,
            stages: vk::ShaderStageFlags,
        }

        let mut merged: BTreeMap<(u32, u32), Merged> = BTreeMap::new();

        for reflection in reflections {
            for binding in &reflection.bindings {
                match merged.entry((binding.set, binding.binding)) {
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        let existing = entry.get_mut();
                        if existing.descriptor_type != binding.descriptor_type {
                            tracing::error!(
                                set = binding.set,
                                binding = binding.binding,
                                "stages disagree on descriptor type"
                            );
                            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                        }
                        existing.count = existing.count.max(binding.count);
                        existing.variable_count |= binding.variable_count;
                        existing.stages |= reflection.stage;
                    },
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(Merged {
                            descriptor_type: binding.descriptor_type,
                            count: binding.count,
                            variable_count: binding.variable_count,
                            stages: reflection.stage,
                        });
                    },
                }
            }
        }

        let set_count = merged.keys().map(|(set, _)| set + 1).max().unwrap_or(0);
        let mut sets: Vec<SetLayout> = Vec::with_capacity(set_count as usize);
        let mut texture_binding = None;

        let destroy_sets = |sets: &[SetLayout]| {
            sets.iter()
                .for_each(|set| unsafe { device.destroy_descriptor_set_layout(set.handle, None) });
        };

        for set in 0..set_count {
            let entries = merged.range((set, 0)..(set + 1, 0));

            let mut bindings = Vec::new();
            let mut binding_flags = Vec::new();
            let mut sizes = Vec::new();
            let mut variable_count = 0;

            for ((_, binding), info) in entries {
                let count = if info.variable_count {
                    variable_count = max_variable_descriptor_count;
                    max_variable_descriptor_count
                } else {
                    info.count
                };

                if info.variable_count && info.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                    texture_binding = Some(TextureBinding { set, binding: *binding });
                }

                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(*binding)
                        .descriptor_type(info.descriptor_type)
                        .descriptor_count(count)
                        .stage_flags(info.stages),
                );

                binding_flags.push(if info.variable_count {
                    vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                        | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                } else {
                    vk::DescriptorBindingFlags::empty()
                });

                sizes.push((info.descriptor_type, count));
            }

            let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
            let mut create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            if variable_count > 0 {
                create_info = create_info
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut flags_info);
            }

            match unsafe { device.create_descriptor_set_layout(&create_info, None) } {
                Ok(handle) => sets.push(SetLayout {
                    handle,
                    sizes,
                    variable_count,
                }),
                Err(err) => {
                    destroy_sets(&sets);
                    return Err(err);
                },
            }
        }

        let push_constant_ranges = push_constant_ranges(reflections);
        let set_layouts = sets.iter().map(|set| set.handle).collect::<Vec<_>>();

        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let handle = match unsafe { device.create_pipeline_layout(&create_info, None) } {
            Ok(handle) => handle,
            Err(err) => {
                destroy_sets(&sets);
                return Err(err);
            },
        };

        Ok(Self {
            handle,
            sets,
            push_constant_ranges,
            texture_binding,
        })
    }

    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.handle, None);
            self.sets
                .iter()
                .for_each(|set| device.destroy_descriptor_set_layout(set.handle, None));
        }
    }
}

pub(crate) struct PipelineRequest<'a> {
    pub info: &'a GraphicsPipelineInfo,
    pub reflections: &'a [Reflection],
    pub vertex: &'a VertexLayout,
    pub state: &'a PipelineState,
    pub layout: vk::PipelineLayout,
}

pub(crate) fn create_pipelines(
    device: &ash::Device, requests: &[PipelineRequest<'_>],
) -> Result<Vec<vk::Pipeline>, vk::Result> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let mut modules: Vec<Vec<vk::ShaderModule>> = Vec::with_capacity(requests.len());
    let destroy_modules = |modules: &[Vec<vk::ShaderModule>]| {
        for module in modules.iter().flatten() {
            unsafe { device.destroy_shader_module(*module, None) };
        }
    };

    for request in requests {
        let mut created = Vec::with_capacity(request.info.shaders.len());
        for spirv in &request.info.shaders {
            let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
            match unsafe { device.create_shader_module(&create_info, None) } {
                Ok(module) => created.push(module),
                Err(err) => {
                    modules.push(created);
                    destroy_modules(&modules);
                    return Err(err);
                },
            }
        }
        modules.push(created);
    }

    let stages = requests
        .iter()
        .zip(&modules)
        .map(|(request, modules)| {
            request
                .reflections
                .iter()
                .zip(modules)
                .map(|(reflection, module)| {
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(reflection.stage)
                        .module(*module)
                        .name(&reflection.entry_point)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let blend_attachments = requests
        .iter()
        .map(|request| {
            request
                .state
                .blend
                .iter()
                .copied()
                .map(vk::PipelineColorBlendAttachmentState::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let viewports = requests
        .iter()
        .map(|request| {
            request
                .state
                .viewports
                .iter()
                .copied()
                .map(vk::Viewport::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let scissors = requests
        .iter()
        .map(|request| request.state.scissors.clone())
        .collect::<Vec<_>>();
    let dynamic_states = requests
        .iter()
        .map(|request| request.state.dynamic_states())
        .collect::<Vec<_>>();

    let vertex_bindings = requests
        .iter()
        .map(|request| {
            Vec::from_iter((!request.vertex.is_empty()).then(|| {
                vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(request.vertex.stride)
                    .input_rate(vk::VertexInputRate::VERTEX)
            }))
        })
        .collect::<Vec<_>>();

    let vertex_attributes = requests
        .iter()
        .map(|request| {
            request
                .vertex
                .attributes
                .iter()
                .map(|attribute| {
                    vk::VertexInputAttributeDescription::default()
                        .binding(0)
                        .location(attribute.location)
                        .format(attribute.format)
                        .offset(attribute.offset)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let vertex_input = vertex_bindings
        .iter()
        .zip(&vertex_attributes)
        .map(|(bindings, attributes)| {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(bindings)
                .vertex_attribute_descriptions(attributes)
        })
        .collect::<Vec<_>>();

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

    let viewport = requests
        .iter()
        .enumerate()
        .map(|(index, request)| {
            let mut info = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(request.state.viewport_count)
                .scissor_count(request.state.viewport_count);
            if !viewports[index].is_empty() {
                info = info.viewports(&viewports[index]);
            }
            if !scissors[index].is_empty() {
                info = info.scissors(&scissors[index]);
            }
            info
        })
        .collect::<Vec<_>>();

    let dynamic_state = dynamic_states
        .iter()
        .map(|states| vk::PipelineDynamicStateCreateInfo::default().dynamic_states(states))
        .collect::<Vec<_>>();

    let input_assembly = requests
        .iter()
        .map(|request| vk::PipelineInputAssemblyStateCreateInfo::default().topology(request.state.topology))
        .collect::<Vec<_>>();

    let rasterization = requests
        .iter()
        .map(|request| vk::PipelineRasterizationStateCreateInfo::from(request.state.rasterization))
        .collect::<Vec<_>>();

    let multisample = requests
        .iter()
        .map(|request| {
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(request.state.rendering.samples)
        })
        .collect::<Vec<_>>();

    let color_blend = blend_attachments
        .iter()
        .map(|attachments| vk::PipelineColorBlendStateCreateInfo::default().attachments(attachments))
        .collect::<Vec<_>>();

    let mut rendering = requests
        .iter()
        .map(|request| {
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&request.state.rendering.color_formats)
        })
        .collect::<Vec<_>>();

    let create_infos = rendering
        .iter_mut()
        .enumerate()
        .map(|(index, rendering)| {
            vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages[index])
                .vertex_input_state(&vertex_input[index])
                .input_assembly_state(&input_assembly[index])
                .viewport_state(&viewport[index])
                .rasterization_state(&rasterization[index])
                .multisample_state(&multisample[index])
                .depth_stencil_state(&depth_stencil)
                .color_blend_state(&color_blend[index])
                .dynamic_state(&dynamic_state[index])
                .layout(requests[index].layout)
                .push_next(rendering)
        })
        .collect::<Vec<_>>();

    let result = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None) };
    destroy_modules(&modules);

    result.map_err(|(created, err)| {
        for pipeline in created.iter().filter(|p| !p.is_null()) {
            unsafe { device.destroy_pipeline(*pipeline, None) };
        }
        err
    })
}

pub(crate) struct ComputePipelineRequest<'a> {
    pub info: &'a ComputePipelineInfo,
    pub reflection: &'a Reflection,
    pub layout: vk::PipelineLayout,
}

pub(crate) fn create_compute_pipelines(
    device: &ash::Device, requests: &[ComputePipelineRequest<'_>],
) -> Result<Vec<vk::Pipeline>, vk::Result> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let mut modules: Vec<vk::ShaderModule> = Vec::with_capacity(requests.len());
    let destroy_modules = |modules: &[vk::ShaderModule]| {
        for module in modules {
            unsafe { device.destroy_shader_module(*module, None) };
        }
    };

    for request in requests {
        let create_info = vk::ShaderModuleCreateInfo::default().code(&request.info.shader);
        match unsafe { device.create_shader_module(&create_info, None) } {
            Ok(module) => modules.push(module),
            Err(err) => {
                destroy_modules(&modules);
                return Err(err);
            },
        }
    }

    let create_infos = requests
        .iter()
        .zip(&modules)
        .map(|(request, module)| {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(*module)
                .name(&request.reflection.entry_point);

            vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(request.layout)
        })
        .collect::<Vec<_>>();

    let result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None) };
    destroy_modules(&modules);

    result.map_err(|(created, err)| {
        for pipeline in created.iter().filter(|p| !p.is_null()) {
            unsafe { device.destroy_pipeline(*pipeline, None) };
        }
        err
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;
    use crate::resource::shader::VertexInput;

    fn reflection(stage: vk::ShaderStageFlags, vertex_inputs: Vec<VertexInput>) -> Reflection {
        Reflection {
            stage,
            entry_point: CString::new("main").unwrap(),
            bindings: Vec::new(),
            push_constant_offset: 0,
            push_constant_size: 0,
            vertex_inputs,
            local_size: [1, 1, 1],
        }
    }

    fn with_push_constants(stage: vk::ShaderStageFlags, offset: u32, size: u32) -> Reflection {
        Reflection {
            push_constant_offset: offset,
            push_constant_size: size,
            ..reflection(stage, Vec::new())
        }
    }

    fn layout(ranges: Vec<vk::PushConstantRange>) -> PipelineLayout {
        PipelineLayout {
            push_constant_ranges: ranges,
            ..Default::default()
        }
    }

    #[test]
    fn packs_vertex_inputs_into_one_tightly_interleaved_binding() {
        let reflections = [
            reflection(vk::ShaderStageFlags::VERTEX, vec![
                VertexInput {
                    location: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    size: 8,
                },
                VertexInput {
                    location: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    size: 12,
                },
            ]),
            reflection(vk::ShaderStageFlags::FRAGMENT, Vec::new()),
        ];

        let layout = VertexLayout::interleaved(&reflections);
        assert_eq!(layout.stride, 20);
        assert_eq!(layout.attributes, vec![
            VertexAttribute {
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            VertexAttribute {
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 8,
            },
        ]);
    }

    #[test]
    fn a_shader_with_no_vertex_inputs_yields_an_empty_layout() {
        let reflections = [reflection(vk::ShaderStageFlags::VERTEX, Vec::new())];
        assert_eq!(VertexLayout::interleaved(&reflections), VertexLayout::default());
    }

    #[test]
    fn stages_that_read_the_same_block_share_one_range() {
        let ranges = push_constant_ranges(&[
            with_push_constants(vk::ShaderStageFlags::VERTEX, 0, 16),
            with_push_constants(vk::ShaderStageFlags::FRAGMENT, 0, 16),
        ]);

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].offset, 0);
        assert_eq!(ranges[0].size, 16);
        assert_eq!(
            ranges[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
    }

    #[test]
    fn stages_that_disagree_get_a_range_each() {
        let ranges = push_constant_ranges(&[
            with_push_constants(vk::ShaderStageFlags::VERTEX, 0, 16),
            with_push_constants(vk::ShaderStageFlags::FRAGMENT, 16, 8),
        ]);

        assert_eq!(
            ranges
                .iter()
                .map(|range| (range.stage_flags, range.offset, range.size))
                .collect::<Vec<_>>(),
            vec![
                (vk::ShaderStageFlags::VERTEX, 0, 16),
                (vk::ShaderStageFlags::FRAGMENT, 16, 8),
            ]
        );
    }

    #[test]
    fn stages_without_push_constants_contribute_no_range() {
        let ranges = push_constant_ranges(&[
            reflection(vk::ShaderStageFlags::VERTEX, Vec::new()),
            reflection(vk::ShaderStageFlags::FRAGMENT, Vec::new()),
        ]);
        assert!(ranges.is_empty());
    }

    #[test]
    fn a_push_is_clipped_to_the_ranges_that_cover_it() {
        let layout = layout(push_constant_ranges(&[
            with_push_constants(vk::ShaderStageFlags::VERTEX, 0, 16),
            with_push_constants(vk::ShaderStageFlags::FRAGMENT, 16, 16),
        ]));

        // a push spanning both ranges is split at the boundary
        assert_eq!(layout.cover(0, 32).collect::<Vec<_>>(), vec![
            (vk::ShaderStageFlags::VERTEX, 0, 16),
            (vk::ShaderStageFlags::FRAGMENT, 16, 16),
        ]);

        // one that lands inside a single range keeps its own bounds
        assert_eq!(layout.cover(20, 4).collect::<Vec<_>>(), vec![(
            vk::ShaderStageFlags::FRAGMENT,
            20,
            4
        )]);

        // and one that lands past every range covers nothing
        assert_eq!(layout.cover(64, 4).count(), 0);
    }
}
