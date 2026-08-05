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

#[derive(Debug, Default)]
pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl PipelineLayout {
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
        let mut push_constant_size = 0;
        let mut push_constant_stages = vk::ShaderStageFlags::empty();

        for reflection in reflections {
            if reflection.push_constant_size > 0 {
                push_constant_size = push_constant_size.max(reflection.push_constant_size);
                push_constant_stages |= reflection.stage;
            }

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
        let mut set_layouts = Vec::with_capacity(set_count as usize);

        for set in 0..set_count {
            let entries = merged.range((set, 0)..(set + 1, 0));

            let mut bindings = Vec::new();
            let mut binding_flags = Vec::new();
            let mut any_variable = false;

            for ((_, binding), info) in entries {
                let count = if info.variable_count {
                    any_variable = true;
                    max_variable_descriptor_count
                } else {
                    info.count
                };

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
            }

            let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
            let mut create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            if any_variable {
                create_info = create_info
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut flags_info);
            }

            match unsafe { device.create_descriptor_set_layout(&create_info, None) } {
                Ok(layout) => set_layouts.push(layout),
                Err(err) => {
                    set_layouts
                        .iter()
                        .for_each(|layout| unsafe { device.destroy_descriptor_set_layout(*layout, None) });
                    return Err(err);
                },
            }
        }

        let push_constant_ranges = Vec::from_iter((push_constant_size > 0).then(|| {
            vk::PushConstantRange::default()
                .stage_flags(push_constant_stages)
                .offset(0)
                .size(push_constant_size)
        }));

        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let handle = match unsafe { device.create_pipeline_layout(&create_info, None) } {
            Ok(handle) => handle,
            Err(err) => {
                set_layouts
                    .iter()
                    .for_each(|layout| unsafe { device.destroy_descriptor_set_layout(*layout, None) });
                return Err(err);
            },
        };

        Ok(Self { handle, set_layouts })
    }

    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.handle, None);
            self.set_layouts
                .iter()
                .for_each(|layout| device.destroy_descriptor_set_layout(*layout, None));
        }
    }
}

pub(crate) struct PipelineRequest<'a> {
    pub info: &'a GraphicsPipelineInfo,
    pub reflections: &'a [Reflection],
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

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
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
                .vertex_input_state(&vertex_input)
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
