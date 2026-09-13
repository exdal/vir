pub mod state;

use std::collections::BTreeMap;

use ash::vk::{self, Handle};
pub use state::{
    BlendPreset,
    ColorBlendAttachmentState,
    DepthState,
    DynamicStateFlags,
    DynamicValues,
    PassState,
    PipelineState,
    PushConstants,
    RasterizationState,
    Rect2D,
    RenderingState,
    ResolvedViewport,
    SpecValue,
    Specializable,
    Specialization,
    StateChange,
    Viewport,
};

use crate::resource::shader::Reflection;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub(crate) u32);

impl PipelineId {
    pub const INVALID: Self = Self(u32::MAX);

    pub const fn is_valid(self) -> bool { self.0 != u32::MAX }

    pub const fn is_invalid(self) -> bool { self.0 == u32::MAX }
}

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
    pub bindless: Option<BindlessDescriptorSet>,
}

impl GraphicsPipelineInfo {
    pub fn new() -> Self { Self::default() }

    pub fn with_shader(mut self, spirv: &[u32]) -> Self {
        self.shaders.push(spirv.to_vec());
        self
    }

    pub fn with_bindless_set(mut self, index: u32, layout: vk::DescriptorSetLayout, set: vk::DescriptorSet) -> Self {
        self.bindless = Some(BindlessDescriptorSet { index, layout, set });
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct ComputePipelineInfo {
    pub shader: Vec<u32>,
    pub bindless: Option<BindlessDescriptorSet>,
}

impl ComputePipelineInfo {
    pub fn new(spirv: &[u32]) -> Self {
        Self {
            shader: spirv.to_vec(),
            bindless: None,
        }
    }

    pub fn with_bindless_set(mut self, index: u32, layout: vk::DescriptorSetLayout, set: vk::DescriptorSet) -> Self {
        self.bindless = Some(BindlessDescriptorSet { index, layout, set });
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindlessDescriptorSet {
    pub index: u32,
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
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

pub(crate) fn validate_descriptor_bindings(
    reflections: &[Reflection], bindless: Option<BindlessDescriptorSet>,
) -> Result<(), vk::Result> {
    if let Some(external) = bindless
        && (external.layout.is_null() || external.set.is_null())
    {
        tracing::error!(set = external.index, "external bindless handles must not be null");
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }

    let mut seen: BTreeMap<(u32, u32), (vk::DescriptorType, u32, bool)> = BTreeMap::new();
    for reflection in reflections {
        for binding in &reflection.bindings {
            let shape = (binding.descriptor_type, binding.count, binding.variable_count);
            if let Some(existing) = seen.insert((binding.set, binding.binding), shape)
                && existing != shape
            {
                tracing::error!(
                    set = binding.set,
                    binding = binding.binding,
                    "shader stages disagree on descriptor shape"
                );
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }

            if bindless.is_some_and(|external| external.index == binding.set) {
                continue;
            }

            let supported_type = matches!(
                binding.descriptor_type,
                vk::DescriptorType::SAMPLER
                    | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    | vk::DescriptorType::SAMPLED_IMAGE
                    | vk::DescriptorType::STORAGE_IMAGE
                    | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    | vk::DescriptorType::STORAGE_TEXEL_BUFFER
                    | vk::DescriptorType::UNIFORM_BUFFER
                    | vk::DescriptorType::STORAGE_BUFFER
                    | vk::DescriptorType::INPUT_ATTACHMENT
                    | vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
            );
            if !supported_type || binding.count != 1 || binding.variable_count {
                tracing::error!(
                    set = binding.set,
                    binding = binding.binding,
                    descriptor_type = ?binding.descriptor_type,
                    count = binding.count,
                    variable_count = binding.variable_count,
                    "ordinary descriptors only support reflected scalar descriptor types"
                );
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
        }
    }

    Ok(())
}

#[derive(Debug, Default, Clone)]
pub struct SetLayout {
    pub handle: vk::DescriptorSetLayout,
    pub sizes: Vec<(vk::DescriptorType, u32)>,
    owned: bool,
}

#[derive(Debug, Default)]
pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,
    pub sets: Vec<SetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
    pub bindless: Option<BindlessDescriptorSet>,
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
        device: &ash::Device, reflections: &[Reflection], bindless: Option<BindlessDescriptorSet>,
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
                        if existing.count != binding.count || existing.variable_count != binding.variable_count {
                            tracing::error!(
                                set = binding.set,
                                binding = binding.binding,
                                "stages disagree on descriptor count"
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

        let reflected_set_count = merged.keys().map(|(set, _)| set + 1).max().unwrap_or(0);
        let set_count = bindless.map_or(reflected_set_count, |set| reflected_set_count.max(set.index + 1));
        let mut sets: Vec<SetLayout> = Vec::with_capacity(set_count as usize);

        let destroy_sets = |sets: &[SetLayout]| {
            sets.iter()
                .filter(|set| set.owned)
                .for_each(|set| unsafe { device.destroy_descriptor_set_layout(set.handle, None) });
        };

        for set in 0..set_count {
            if let Some(external) = bindless.filter(|external| external.index == set) {
                sets.push(SetLayout {
                    handle: external.layout,
                    sizes: Vec::new(),
                    owned: false,
                });
                continue;
            }

            let entries = merged.range((set, 0)..(set + 1, 0));

            let mut bindings = Vec::new();
            let mut sizes = Vec::new();

            for ((_, binding), info) in entries {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(*binding)
                        .descriptor_type(info.descriptor_type)
                        .descriptor_count(info.count)
                        .stage_flags(info.stages),
                );

                sizes.push((info.descriptor_type, info.count));
            }

            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            match unsafe { device.create_descriptor_set_layout(&create_info, None) } {
                Ok(handle) => sets.push(SetLayout {
                    handle,
                    sizes,
                    owned: true,
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
            bindless,
        })
    }

    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.handle, None);
            self.sets
                .iter()
                .filter(|set| set.owned)
                .for_each(|set| device.destroy_descriptor_set_layout(set.handle, None));
        }
    }
}

/// One value blob for the whole pipeline, and the map entries each stage reads it through.
///
/// A constant declared by more than one stage is written once and named by every stage that
/// declares it, which is what the offsets in the entries are for.
pub(crate) struct PackedSpecialization {
    pub data: Vec<u8>,
    pub entries: Vec<Vec<vk::SpecializationMapEntry>>,
}

pub(crate) fn pack_specialization(
    reflections: &[Reflection], specialization: &Specialization,
) -> Result<PackedSpecialization, vk::Result> {
    let mut merged: BTreeMap<u32, (u32, vk::ShaderStageFlags)> = BTreeMap::new();
    for reflection in reflections {
        for constant in &reflection.spec_constants {
            match merged.entry(constant.id) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let (size, stages) = entry.get_mut();
                    if *size != constant.size {
                        tracing::error!(
                            id = constant.id,
                            declared = *size,
                            found = constant.size,
                            "shader stages disagree on the size of a specialization constant"
                        );
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    *stages |= constant.stage;
                },
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((constant.size, constant.stage));
                },
            }
        }
    }

    // an id the shaders never declare is not an error: the constants in force are whatever the
    // pass set, and a pass sets them for whichever pipeline it goes on to bind
    for (id, value) in specialization.iter() {
        match merged.get(&id) {
            Some((size, _)) if *size != value.size() => {
                tracing::error!(
                    id,
                    declared = *size,
                    given = value.size(),
                    "the value given for a specialization constant is not the size the shader declares"
                );
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            },
            Some(_) => {},
            None => tracing::debug!(
                id,
                "no shader stage declares this specialization constant; it is dropped"
            ),
        }
    }

    let mut data = Vec::new();
    let mut placed: Vec<(u32, u32, u32, vk::ShaderStageFlags)> = Vec::new();
    for (id, (size, stages)) in merged {
        let Some(value) = specialization.get(id) else {
            continue;
        };

        let offset = data.len() as u32;
        data.extend_from_slice(&value.bytes()[..size as usize]);
        placed.push((id, offset, size, stages));
    }

    let entries = reflections
        .iter()
        .map(|reflection| {
            placed
                .iter()
                .filter(|(_, _, _, stages)| stages.contains(reflection.stage))
                .map(|(id, offset, size, _)| {
                    vk::SpecializationMapEntry::default()
                        .constant_id(*id)
                        .offset(*offset)
                        .size(*size as usize)
                })
                .collect()
        })
        .collect();

    Ok(PackedSpecialization { data, entries })
}

/// The `VkSpecializationInfo` each stage is given, or none where the stage specializes nothing.
///
/// Vulkan reads the pointers inside these through the create infos, so both this and the
/// [`PackedSpecialization`] it borrows have to outlive the `vkCreate*Pipelines` call.
fn specialization_infos<'a>(packed: &'a PackedSpecialization) -> Vec<Option<vk::SpecializationInfo<'a>>> {
    packed
        .entries
        .iter()
        .map(|entries| {
            (!entries.is_empty()).then(|| {
                vk::SpecializationInfo::default()
                    .map_entries(entries)
                    .data(&packed.data)
            })
        })
        .collect()
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

    let packed = requests
        .iter()
        .map(|request| pack_specialization(request.reflections, &request.state.specialization))
        .collect::<Result<Vec<_>, _>>()?;
    let specializations = packed.iter().map(specialization_infos).collect::<Vec<_>>();

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
        .zip(&specializations)
        .map(|((request, modules), specializations)| {
            request
                .reflections
                .iter()
                .zip(modules)
                .zip(specializations)
                .map(|((reflection, module), specialization)| {
                    let stage = vk::PipelineShaderStageCreateInfo::default()
                        .stage(reflection.stage)
                        .module(*module)
                        .name(&reflection.entry_point);
                    match specialization {
                        Some(specialization) => stage.specialization_info(specialization),
                        None => stage,
                    }
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

    let depth_stencil = requests
        .iter()
        .map(|request| vk::PipelineDepthStencilStateCreateInfo::from(request.state.depth))
        .collect::<Vec<_>>();

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
            let info = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&request.state.rendering.color_formats);
            match request.state.rendering.depth_format {
                Some(format) => info.depth_attachment_format(format),
                None => info,
            }
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
                .depth_stencil_state(&depth_stencil[index])
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
    pub specialization: &'a Specialization,
    pub layout: vk::PipelineLayout,
}

pub(crate) fn create_compute_pipelines(
    device: &ash::Device, requests: &[ComputePipelineRequest<'_>],
) -> Result<Vec<vk::Pipeline>, vk::Result> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let packed = requests
        .iter()
        .map(|request| pack_specialization(std::slice::from_ref(request.reflection), request.specialization))
        .collect::<Result<Vec<_>, _>>()?;
    let specializations = packed.iter().map(specialization_infos).collect::<Vec<_>>();

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
        .zip(&specializations)
        .map(|((request, module), specialization)| {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(*module)
                .name(&request.reflection.entry_point);
            let stage = match specialization.first().and_then(Option::as_ref) {
                Some(specialization) => stage.specialization_info(specialization),
                None => stage,
            };

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

    use ash::vk::Handle;

    use super::*;
    use crate::{
        Access,
        resource::shader::{DescriptorBinding, SpecConstant, VertexInput},
    };

    fn reflection(stage: vk::ShaderStageFlags, vertex_inputs: Vec<VertexInput>) -> Reflection {
        Reflection {
            stage,
            entry_point: CString::new("main").unwrap(),
            bindings: Vec::new(),
            spec_constants: Vec::new(),
            local_size_spec_ids: [None; 3],
            push_constant_offset: 0,
            push_constant_size: 0,
            vertex_inputs,
            local_size: [1, 1, 1],
        }
    }

    fn with_spec_constants(stage: vk::ShaderStageFlags, constants: &[(u32, u32)]) -> Reflection {
        Reflection {
            spec_constants: constants
                .iter()
                .map(|(id, size)| SpecConstant {
                    id: *id,
                    size: *size,
                    stage,
                })
                .collect(),
            ..reflection(stage, Vec::new())
        }
    }

    fn entry(entry: &vk::SpecializationMapEntry) -> (u32, u32, usize) { (entry.constant_id, entry.offset, entry.size) }

    #[test]
    fn a_constant_two_stages_declare_is_written_once_and_named_by_both() {
        let reflections = [
            with_spec_constants(vk::ShaderStageFlags::VERTEX, &[(0, 4), (1, 4)]),
            with_spec_constants(vk::ShaderStageFlags::FRAGMENT, &[(1, 4)]),
        ];
        let specialization = Specialization::from_iter([(0, SpecValue::U32(7)), (1, SpecValue::F32(1.0f32.to_bits()))]);

        let packed = pack_specialization(&reflections, &specialization).expect("the values fit the reflection");
        assert_eq!(packed.data.len(), 8);
        assert_eq!(&packed.data[..4], &7u32.to_le_bytes());
        assert_eq!(&packed.data[4..], &1.0f32.to_bits().to_le_bytes());
        assert_eq!(
            packed.entries[0].iter().map(entry).collect::<Vec<_>>(),
            vec![(0, 0, 4), (1, 4, 4)]
        );
        assert_eq!(packed.entries[1].iter().map(entry).collect::<Vec<_>>(), vec![(1, 4, 4)]);
    }

    #[test]
    fn a_constant_left_unset_contributes_no_bytes_and_no_entry() {
        let reflections = [with_spec_constants(vk::ShaderStageFlags::COMPUTE, &[(0, 4), (1, 4)])];
        let specialization = Specialization::from_iter([(1, SpecValue::U32(3))]);

        let packed = pack_specialization(&reflections, &specialization).expect("the values fit the reflection");
        assert_eq!(packed.data, 3u32.to_le_bytes());
        assert_eq!(packed.entries[0].iter().map(entry).collect::<Vec<_>>(), vec![(1, 0, 4)]);
    }

    /// A pass sets its constants for whichever pipeline it goes on to bind, so one the shader
    /// never declares is nothing to complain about.
    #[test]
    fn a_constant_no_stage_declares_is_dropped() {
        let reflections = [with_spec_constants(vk::ShaderStageFlags::COMPUTE, &[(0, 4)])];
        let specialization = Specialization::from_iter([(9, SpecValue::U32(3))]);

        let packed = pack_specialization(&reflections, &specialization).expect("an unknown id is not an error");
        assert!(packed.data.is_empty());
        assert!(packed.entries[0].is_empty());
    }

    #[test]
    fn a_value_of_the_wrong_width_is_rejected() {
        let reflections = [with_spec_constants(vk::ShaderStageFlags::COMPUTE, &[(0, 4)])];
        let specialization = Specialization::from_iter([(0, SpecValue::F64(0))]);
        assert!(pack_specialization(&reflections, &specialization).is_err());
    }

    #[test]
    fn stages_that_disagree_on_the_width_of_a_constant_are_rejected() {
        let reflections = [
            with_spec_constants(vk::ShaderStageFlags::VERTEX, &[(0, 4)]),
            with_spec_constants(vk::ShaderStageFlags::FRAGMENT, &[(0, 8)]),
        ];
        assert!(pack_specialization(&reflections, &Specialization::default()).is_err());
    }

    fn with_push_constants(stage: vk::ShaderStageFlags, offset: u32, size: u32) -> Reflection {
        Reflection {
            push_constant_offset: offset,
            push_constant_size: size,
            ..reflection(stage, Vec::new())
        }
    }

    fn with_bindings(stage: vk::ShaderStageFlags, bindings: Vec<DescriptorBinding>) -> Reflection {
        Reflection {
            bindings,
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
            reflection(
                vk::ShaderStageFlags::VERTEX,
                vec![
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
                ],
            ),
            reflection(vk::ShaderStageFlags::FRAGMENT, Vec::new()),
        ];

        let layout = VertexLayout::interleaved(&reflections);
        assert_eq!(layout.stride, 20);
        assert_eq!(
            layout.attributes,
            vec![
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
            ]
        );
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
        assert_eq!(
            layout.cover(0, 32).collect::<Vec<_>>(),
            vec![
                (vk::ShaderStageFlags::VERTEX, 0, 16),
                (vk::ShaderStageFlags::FRAGMENT, 16, 16),
            ]
        );

        // one that lands inside a single range keeps its own bounds
        assert_eq!(
            layout.cover(20, 4).collect::<Vec<_>>(),
            vec![(vk::ShaderStageFlags::FRAGMENT, 20, 4)]
        );

        // and one that lands past every range covers nothing
        assert_eq!(layout.cover(64, 4).count(), 0);
    }

    #[test]
    fn ordinary_descriptors_accept_every_reflected_scalar_type() {
        let descriptor_types = [
            vk::DescriptorType::SAMPLER,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::INPUT_ATTACHMENT,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        ];
        let reflection = with_bindings(
            vk::ShaderStageFlags::FRAGMENT,
            descriptor_types
                .into_iter()
                .enumerate()
                .map(|(binding, descriptor_type)| DescriptorBinding {
                    set: 1,
                    binding: binding as u32,
                    descriptor_type,
                    count: 1,
                    variable_count: false,
                    stages: vk::ShaderStageFlags::FRAGMENT,
                    access: Access::None,
                })
                .collect(),
        );
        assert!(validate_descriptor_bindings(&[reflection], None).is_ok());
    }

    #[test]
    fn descriptor_arrays_require_an_external_bindless_set() {
        let reflection = with_bindings(
            vk::ShaderStageFlags::FRAGMENT,
            vec![DescriptorBinding {
                set: 2,
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 0,
                variable_count: true,
                stages: vk::ShaderStageFlags::FRAGMENT,
                access: Access::FragmentSampled,
            }],
        );
        assert!(validate_descriptor_bindings(std::slice::from_ref(&reflection), None).is_err());

        let bindless = BindlessDescriptorSet {
            index: 2,
            layout: vk::DescriptorSetLayout::from_raw(1),
            set: vk::DescriptorSet::from_raw(2),
        };
        assert!(validate_descriptor_bindings(&[reflection], Some(bindless)).is_ok());
    }
}
