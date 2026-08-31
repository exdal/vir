use std::collections::BTreeMap;

use ash::vk::{self, Handle};

use crate::{
    DescriptorBinding,
    IR,
    PipelineId,
    Program,
    ValueId,
    graph::ir::{Descriptor, Instr, MAX_RESOLVE_DEPTH, UnderlyingObject, underlying_object},
};

/// Where the analyzer reads what a program's pipelines declare, as their shaders were reflected.
///
/// `None` from [`Self::bindings`] means this source knows nothing about that pipeline. Descriptor-free
/// modules may still compile, but descriptor writes need reflected bindings to infer their type and access.
pub trait PipelineBindings {
    fn bindings(&self, pipeline: PipelineId) -> Option<&[DescriptorBinding]>;

    /// The set handed to the pipeline whole. Nothing in the module may write into it.
    fn bindless_set(&self, pipeline: PipelineId) -> Option<u32>;
}

/// Compiles a descriptor-free module without a graph to ask about its pipelines.
pub struct Unchecked;

impl PipelineBindings for Unchecked {
    fn bindings(&self, _: PipelineId) -> Option<&[DescriptorBinding]> { None }

    fn bindless_set(&self, _: PipelineId) -> Option<u32> { None }
}

/// Pipeline bindings written out by tests rather than reflected from shaders.
#[cfg(test)]
pub(super) struct Declared {
    bindings: Vec<DescriptorBinding>,
    bindless: Option<u32>,
}

#[cfg(test)]
impl Declared {
    pub(super) fn new(bindings: &[(u32, u32, vk::DescriptorType)]) -> Self {
        Self::in_stages(bindings, vk::ShaderStageFlags::FRAGMENT)
    }

    pub(super) fn in_stages(bindings: &[(u32, u32, vk::DescriptorType)], stages: vk::ShaderStageFlags) -> Self {
        Self::with_access(bindings, stages, crate::Access::sampled_by(stages))
    }

    pub(super) fn with_access(
        bindings: &[(u32, u32, vk::DescriptorType)], stages: vk::ShaderStageFlags, access: crate::Access,
    ) -> Self {
        Self {
            bindings: bindings
                .iter()
                .map(|(set, binding, descriptor_type)| DescriptorBinding {
                    set: *set,
                    binding: *binding,
                    descriptor_type: *descriptor_type,
                    count: 1,
                    variable_count: false,
                    stages,
                    access,
                })
                .collect(),
            bindless: None,
        }
    }

    pub(super) fn with_bindless(mut self, set: u32) -> Self {
        self.bindless = Some(set);
        self
    }
}

#[cfg(test)]
impl PipelineBindings for Declared {
    fn bindings(&self, _: PipelineId) -> Option<&[DescriptorBinding]> { Some(&self.bindings) }

    fn bindless_set(&self, _: PipelineId) -> Option<u32> { self.bindless }
}

#[derive(Default)]
struct Table {
    open: bool,
    written: BTreeMap<(u32, u32), (ValueId, Descriptor, Option<vk::DescriptorType>)>,
}

impl Table {
    fn open(&mut self) {
        self.open = true;
        self.written.clear();
    }

    fn close(&mut self) {
        self.open = false;
        self.written.clear();
    }
}

pub(crate) fn analyze_descriptors(program: &Program, pipelines: &impl PipelineBindings) -> Result<(), vk::Result> {
    let mut table = Table::default();
    let mut failed = false;

    for (value_id, ir) in program.instructions() {
        match ir {
            IR::BeginRendering { .. } | IR::BeginCompute { .. } => table.open(),
            IR::EndRendering { .. } | IR::EndCompute { .. } => table.close(),

            IR::WriteDescriptor {
                set,
                binding,
                descriptor,
                descriptor_type,
                ..
            } => {
                failed |= !write_is_sound(program, value_id, &table, *set, *binding, descriptor);
                table
                    .written
                    .insert((*set, *binding), (*value_id, *descriptor, *descriptor_type));
            },

            // a callback binds whatever it draws with itself, so there is nothing here to check
            // it against
            IR::CallOpaque { .. } => {},

            IR::Draw { pipeline, .. } | IR::DrawIndexed { pipeline, .. } | IR::Dispatch { pipeline, .. } => {
                failed |= !descriptors_are_compatible_with_pipeline(value_id, &table, *pipeline, pipelines);
            },

            _ => {},
        }
    }

    match failed {
        true => Err(vk::Result::ERROR_INITIALIZATION_FAILED),
        false => Ok(()),
    }
}

fn write_is_sound(
    program: &Program, value_id: &ValueId, table: &Table, set: u32, binding: u32, descriptor: &Descriptor,
) -> bool {
    if !table.open {
        tracing::error!(%value_id, set, binding, "a descriptor is written outside of any pass");
        return false;
    }

    if let Some(image) = descriptor.image()
        && !names_an_image(program.instructions(), image)
    {
        tracing::error!(%value_id, set, binding, %image, "an image descriptor names a value that is not an image");
        return false;
    }

    if let Some(buffer) = descriptor.buffer()
        && !names_a_buffer(program.instructions(), buffer)
    {
        tracing::error!(
            %value_id,
            set,
            binding,
            %buffer,
            "a buffer-backed descriptor names a value that is not a buffer"
        );
        return false;
    }

    if descriptor.sampler().is_some_and(|sampler| sampler.is_null()) {
        tracing::error!(%value_id, set, binding, "a sampler descriptor is written with a null sampler");
        return false;
    }

    match descriptor {
        Descriptor::TexelBuffer { view, .. } if view.is_null() => {
            tracing::error!(%value_id, set, binding, "a texel buffer descriptor is written with a null view");
            return false;
        },
        Descriptor::Buffer { offset, range, .. }
            if *range == 0 || (*range != vk::WHOLE_SIZE && offset.checked_add(*range).is_none()) =>
        {
            tracing::error!(%value_id, set, binding, offset, range, "a buffer descriptor has an invalid range");
            return false;
        },
        Descriptor::AccelerationStructure {
            acceleration_structure, ..
        } if acceleration_structure.is_null() => {
            tracing::error!(%value_id, set, binding, "an acceleration structure descriptor is written with a null handle");
            return false;
        },
        _ => {},
    }

    true
}

fn descriptors_are_compatible_with_pipeline(
    value_id: &ValueId, table: &Table, pipeline: PipelineId, pipelines: &impl PipelineBindings,
) -> bool {
    // there is nothing to check the writes against without one. the graph refuses to record it
    // either way, so this is left as the warning the inference already made it
    if pipeline.is_invalid() {
        return true;
    }

    let Some(bindings) = pipelines.bindings(pipeline) else {
        return true;
    };
    let bindless = pipelines.bindless_set(pipeline);
    let mut sound = true;

    for binding in bindings {
        if bindless == Some(binding.set) {
            continue;
        }

        let Some((written_at, descriptor, resolved_type)) = table.written.get(&(binding.set, binding.binding)) else {
            tracing::error!(
                %value_id,
                %pipeline,
                set = binding.set,
                binding = binding.binding,
                descriptor_type = ?binding.descriptor_type,
                "the pipeline declares a descriptor nothing has written at this point"
            );
            sound = false;
            continue;
        };

        if !descriptor.supports_type(binding.descriptor_type) {
            tracing::error!(
                %value_id,
                %pipeline,
                written_at = %written_at,
                set = binding.set,
                binding = binding.binding,
                declared = ?binding.descriptor_type,
                payload = ?descriptor,
                "the descriptor payload written here is incompatible with the type the pipeline declares"
            );
            sound = false;
        } else if *resolved_type != Some(binding.descriptor_type) {
            tracing::error!(
                %value_id,
                %pipeline,
                written_at = %written_at,
                set = binding.set,
                binding = binding.binding,
                declared = ?binding.descriptor_type,
                resolved = ?resolved_type,
                "the descriptor write was not resolved to the type this pipeline declares"
            );
            sound = false;
        }
    }

    // a bindless set is handed to the pipeline whole, writing into it would go to a set the
    // graph does not own
    for (set, binding) in table.written.keys() {
        if bindless == Some(*set) {
            tracing::error!(
                %value_id,
                %pipeline,
                set,
                binding,
                "a descriptor is written into the pipeline's bindless set"
            );
            sound = false;
        }
    }

    sound
}

fn names_an_image(instructions: &[Instr], id: ValueId) -> bool {
    names_a_resource(instructions, id, |ir| {
        matches!(ir, IR::ConstructImage { .. } | IR::SwapchainImage { .. })
    })
}

fn names_a_buffer(instructions: &[Instr], id: ValueId) -> bool {
    names_a_resource(instructions, id, |ir| matches!(ir, IR::ConstructBuffer { .. }))
}

fn names_a_resource(instructions: &[Instr], id: ValueId, is_base: fn(&IR) -> bool) -> bool {
    let mut id = id;

    for _ in 0..MAX_RESOLVE_DEPTH {
        let Some((_, ir)) = instructions.iter().find(|(instr, _)| *instr == id) else {
            return false;
        };

        id = match underlying_object(ir) {
            UnderlyingObject::Base => return is_base(ir),
            UnderlyingObject::Forwards(next) | UnderlyingObject::Element(next) => next,
            UnderlyingObject::None => return false,
        };
    }

    false
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ash::vk::Handle;

    use super::*;
    use crate::{Access, BufferInfo, ImageInfo, MemoryLocation, Module, ValueId};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

    fn a_sampler() -> vk::Sampler { vk::Sampler::from_raw(1) }

    fn target(module: &mut Module) -> ValueId {
        module.transient_image(&ImageInfo::color_target(
            vk::Extent2D::default().width(4).height(4),
            FORMAT,
        ))
    }

    #[derive(Default)]
    struct PerPipeline {
        bindings: HashMap<PipelineId, Vec<DescriptorBinding>>,
    }

    impl PerPipeline {
        fn with_binding(self, pipeline: PipelineId, stages: vk::ShaderStageFlags) -> Self {
            self.with_typed_binding(
                pipeline,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stages,
                Access::sampled_by(stages),
            )
        }

        fn with_typed_binding(
            mut self, pipeline: PipelineId, descriptor_type: vk::DescriptorType, stages: vk::ShaderStageFlags,
            access: Access,
        ) -> Self {
            self.bindings.insert(
                pipeline,
                vec![DescriptorBinding {
                    set: 0,
                    binding: 0,
                    descriptor_type,
                    count: 1,
                    variable_count: false,
                    stages,
                    access,
                }],
            );
            self
        }
    }

    impl PipelineBindings for PerPipeline {
        fn bindings(&self, pipeline: PipelineId) -> Option<&[DescriptorBinding]> {
            self.bindings.get(&pipeline).map(Vec::as_slice)
        }

        fn bindless_set(&self, _: PipelineId) -> Option<u32> { None }
    }

    /// A pass that writes `writes` and then draws once.
    fn drawing(writes: impl FnOnce(&mut Module, ValueId)) -> (Module, ValueId) {
        let mut module = Module::default();
        let attachment = target(&mut module);
        let texture = target(&mut module);

        module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0));
        writes(&mut module, texture);
        let end = module.draw(3, 1).end_rendering();

        (module, end)
    }

    #[test]
    fn a_write_that_matches_the_reflected_type_is_accepted() {
        let combined = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let (module, end) = drawing(|m, texture| {
            m.bind_texture(0, 0, texture, a_sampler());
        });

        assert!(module.compile(&combined, end).is_ok());
    }

    #[test]
    fn every_reflected_scalar_descriptor_type_can_be_written() {
        fn compile(
            descriptor_type: vk::DescriptorType, access: Access, bind: impl FnOnce(&mut Module, ValueId, ValueId),
        ) {
            let mut module = Module::default();
            let attachment = target(&mut module);
            let image = target(&mut module);
            let buffer = module.transient_buffer(&BufferInfo::new(
                256,
                vk::BufferUsageFlags::empty(),
                MemoryLocation::GpuOnly,
            ));
            module
                .begin_rendering([(attachment, Access::ColorRW)])
                .bind_graphics_pipeline(PipelineId(0));
            bind(&mut module, image, buffer);
            let end = module.draw(3, 1).end_rendering();
            let declared = Declared::with_access(&[(0, 0, descriptor_type)], vk::ShaderStageFlags::FRAGMENT, access);
            assert!(module.compile(&declared, end).is_ok(), "{descriptor_type:?}");
        }

        compile(vk::DescriptorType::SAMPLER, Access::None, |module, _, _| {
            module.bind_sampler(0, 0, a_sampler());
        });
        compile(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Access::FragmentSampled,
            |module, image, _| {
                module.bind_texture(0, 0, image, a_sampler());
            },
        );
        compile(
            vk::DescriptorType::SAMPLED_IMAGE,
            Access::FragmentSampled,
            |module, image, _| {
                module.bind_image(0, 0, image);
            },
        );
        compile(
            vk::DescriptorType::STORAGE_IMAGE,
            Access::FragmentRead,
            |module, image, _| {
                module.bind_image(0, 0, image);
            },
        );
        compile(
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            Access::FragmentSampled,
            |module, _, buffer| {
                module.bind_texel_buffer(0, 0, buffer, vk::BufferView::from_raw(2));
            },
        );
        compile(
            vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            Access::FragmentRead,
            |module, _, buffer| {
                module.bind_texel_buffer(0, 0, buffer, vk::BufferView::from_raw(2));
            },
        );
        compile(
            vk::DescriptorType::UNIFORM_BUFFER,
            Access::FragmentUniformRead,
            |module, _, buffer| {
                module.bind_buffer(0, 0, buffer);
            },
        );
        compile(
            vk::DescriptorType::STORAGE_BUFFER,
            Access::FragmentRead,
            |module, _, buffer| {
                module.bind_buffer_range(0, 0, buffer, 16, 64);
            },
        );
        compile(
            vk::DescriptorType::INPUT_ATTACHMENT,
            Access::InputAttachmentRead,
            |module, image, _| {
                module.bind_image(0, 0, image);
            },
        );
        compile(
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Access::FragmentAccelerationStructureRead,
            |module, _, buffer| {
                module.bind_acceleration_structure(0, 0, buffer, vk::AccelerationStructureKHR::from_raw(3));
            },
        );
    }

    #[test]
    fn descriptor_writes_reject_null_handles_and_empty_buffer_ranges() {
        fn rejects(descriptor_type: vk::DescriptorType, bind: impl FnOnce(&mut Module, ValueId, ValueId)) {
            let mut module = Module::default();
            let attachment = target(&mut module);
            let image = target(&mut module);
            let buffer = module.transient_buffer(&BufferInfo::new(
                256,
                vk::BufferUsageFlags::empty(),
                MemoryLocation::GpuOnly,
            ));
            module
                .begin_rendering([(attachment, Access::ColorRW)])
                .bind_graphics_pipeline(PipelineId(0));
            bind(&mut module, image, buffer);
            let end = module.draw(3, 1).end_rendering();
            let declared =
                Declared::with_access(&[(0, 0, descriptor_type)], vk::ShaderStageFlags::FRAGMENT, Access::None);
            assert!(module.compile(&declared, end).is_err(), "{descriptor_type:?}");
        }

        rejects(vk::DescriptorType::SAMPLER, |module, _, _| {
            module.bind_sampler(0, 0, vk::Sampler::null());
        });
        rejects(vk::DescriptorType::UNIFORM_TEXEL_BUFFER, |module, _, buffer| {
            module.bind_texel_buffer(0, 0, buffer, vk::BufferView::null());
        });
        rejects(vk::DescriptorType::STORAGE_BUFFER, |module, _, buffer| {
            module.bind_buffer_range(0, 0, buffer, 0, 0);
        });
        rejects(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, |module, _, buffer| {
            module.bind_acceleration_structure(0, 0, buffer, vk::AccelerationStructureKHR::null());
        });
    }

    #[test]
    fn a_write_of_the_wrong_type_is_refused() {
        let combined = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let (module, end) = drawing(|m, texture| {
            m.bind_image(0, 0, texture);
        });

        assert!(module.compile(&combined, end).is_err());
    }

    #[test]
    fn a_binding_nothing_wrote_is_refused() {
        let combined = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let (module, end) = drawing(|_, _| {});

        assert!(module.compile(&combined, end).is_err());
    }

    /// The write has to be reached before the draw is, not merely be somewhere in the pass.
    #[test]
    fn a_write_after_the_draw_does_not_cover_it() {
        let combined = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);

        let mut module = Module::default();
        let attachment = target(&mut module);
        let texture = target(&mut module);
        let end = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .bind_texture(0, 0, texture, a_sampler())
            .end_rendering();

        assert!(module.compile(&combined, end).is_err());
    }

    #[test]
    fn a_write_the_pipeline_does_not_declare_is_allowed() {
        let none = Declared::new(&[]);
        let (module, end) = drawing(|m, texture| {
            m.bind_texture(3, 1, texture, a_sampler());
        });

        let compiled = module.compile(&none, end).unwrap();
        let descriptor_type = compiled.instructions().iter().find_map(|(_, ir)| match ir {
            IR::WriteDescriptor { descriptor_type, .. } => Some(*descriptor_type),
            _ => None,
        });
        assert_eq!(descriptor_type, Some(None));
    }

    #[test]
    fn a_write_before_a_pipeline_is_resolved_from_the_consuming_draw() {
        let combined = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let mut module = Module::default();
        let attachment = target(&mut module);
        let texture = target(&mut module);
        let end = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_texture(0, 0, texture, a_sampler())
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&combined, end).unwrap();
        assert_eq!(sampled_barrier(&compiled), Access::FragmentSampled);
    }

    #[test]
    fn a_write_outside_a_pass_is_dropped_without_affecting_unrelated_compilation() {
        let mut module = Module::default();
        let attachment = target(&mut module);
        let texture = target(&mut module);
        module.bind_texture(0, 0, texture, a_sampler());
        let end = module.clear(attachment, crate::clear::f32::BLACK);

        assert!(module.compile(&Unchecked, end).is_ok());
    }

    #[test]
    fn a_descriptor_consumed_without_a_valid_pipeline_is_refused() {
        let mut module = Module::default();
        let attachment = target(&mut module);
        let texture = target(&mut module);
        let end = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId::INVALID)
            .bind_texture(0, 0, texture, a_sampler())
            .draw(3, 1)
            .end_rendering();

        assert!(matches!(
            module.compile(&Unchecked, end),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        ));
    }

    #[test]
    fn a_write_into_the_bindless_set_is_refused() {
        let bindless = Declared::new(&[]).with_bindless(3);
        let (module, end) = drawing(|m, texture| {
            m.bind_texture(3, 1, texture, a_sampler());
        });

        assert!(module.compile(&bindless, end).is_err());
    }

    /// The graph hands a bindless set to the pipeline whole, so a binding inside it is not one
    /// the module has to have written.
    #[test]
    fn a_bindless_binding_needs_no_write() {
        let bindless = Declared::new(&[(3, 1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]).with_bindless(3);
        let (module, end) = drawing(|_, _| {});

        assert!(module.compile(&bindless, end).is_ok());
    }

    #[test]
    fn a_combined_image_sampler_written_with_no_sampler_is_refused() {
        let combined = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let (module, end) = drawing(|m, texture| {
            m.bind_texture(0, 0, texture, vk::Sampler::null());
        });

        assert!(module.compile(&combined, end).is_err());
    }

    /// A pass starts with nothing written, so what a previous one left cannot stand in for it.
    #[test]
    fn a_write_does_not_carry_into_the_next_pass() {
        let combined = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);

        let mut module = Module::default();
        let attachment = target(&mut module);
        let texture = target(&mut module);
        let first = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_texture(0, 0, texture, a_sampler())
            .draw(3, 1)
            .end_rendering();
        let end = module
            .begin_rendering([(first, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();

        assert!(module.compile(&combined, end).is_err());
    }
    /// What the barrier for the written image actually waits on, read back off the emitted
    /// barrier rather than off the write, since the barrier is the thing that has to be right.
    fn sampled_barrier_for(program: &Program, sampled: ValueId) -> Access {
        let instructions = program.instructions();
        let constant = |id: &ValueId| {
            instructions.iter().find_map(|(instr, ir)| match ir {
                IR::Constant(crate::graph::ir::Constant::Access(access)) if instr == id => Some(*access),
                _ => None,
            })
        };

        instructions
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::ImageBarrier { dst_access, value, .. } if *value == sampled => constant(dst_access),
                _ => None,
            })
            .expect("the written image should be transitioned")
    }

    fn sampled_barrier(program: &Program) -> Access {
        let sampled = program
            .instructions()
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::WriteDescriptor { descriptor, .. } => descriptor.image(),
                _ => None,
            })
            .expect("the write should survive");
        sampled_barrier_for(program, sampled)
    }

    /// The whole point of resolving from reflection: a texture the vertex stage reads needs its
    /// transition finished a stage earlier than one the fragment stage reads, and nothing about
    /// the pass it was recorded in tells them apart.
    #[test]
    fn the_stage_that_declares_a_binding_decides_its_access() {
        let write = &[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)];
        let bind = |m: &mut Module, texture| {
            m.bind_texture(0, 0, texture, a_sampler());
        };

        let (module, end) = drawing(bind);
        let fragment = Declared::in_stages(write, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(
            sampled_barrier(&module.compile(&fragment, end).unwrap()),
            Access::FragmentSampled
        );

        let (module, end) = drawing(bind);
        let vertex = Declared::in_stages(write, vk::ShaderStageFlags::VERTEX);
        assert_eq!(
            sampled_barrier(&module.compile(&vertex, end).unwrap()),
            Access::VertexSampled
        );

        // a binding both stages declare is waited on for both
        let (module, end) = drawing(bind);
        let both = Declared::in_stages(write, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(
            sampled_barrier(&module.compile(&both, end).unwrap()),
            Access::VertexSampled | Access::FragmentSampled
        );
    }

    #[test]
    fn an_unchecked_compile_refuses_a_descriptor_write() {
        let (module, end) = drawing(|m, texture| {
            m.bind_texture(0, 0, texture, a_sampler());
        });

        assert!(matches!(
            module.compile(&Unchecked, end),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        ));
    }

    #[test]
    fn a_standing_write_widens_for_every_pipeline_that_consumes_it() {
        let fragment = PipelineId(0);
        let vertex = PipelineId(1);
        let pipelines = PerPipeline::default()
            .with_binding(fragment, vk::ShaderStageFlags::FRAGMENT)
            .with_binding(vertex, vk::ShaderStageFlags::VERTEX);

        let mut module = Module::default();
        let attachment = target(&mut module);
        let texture = target(&mut module);
        let end = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(fragment)
            .bind_texture(0, 0, texture, a_sampler())
            .draw(3, 1)
            .bind_graphics_pipeline(vertex)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&pipelines, end).unwrap();
        assert_eq!(
            sampled_barrier_for(&compiled, texture),
            Access::FragmentSampled | Access::VertexSampled
        );
    }

    #[test]
    fn rewriting_a_descriptor_starts_a_new_access_association() {
        let fragment = PipelineId(0);
        let vertex = PipelineId(1);
        let pipelines = PerPipeline::default()
            .with_binding(fragment, vk::ShaderStageFlags::FRAGMENT)
            .with_binding(vertex, vk::ShaderStageFlags::VERTEX);

        let mut module = Module::default();
        let attachment = target(&mut module);
        let first = target(&mut module);
        let second = target(&mut module);
        let end = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(fragment)
            .bind_texture(0, 0, first, a_sampler())
            .draw(3, 1)
            .bind_graphics_pipeline(vertex)
            .bind_texture(0, 0, second, a_sampler())
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&pipelines, end).unwrap();
        assert_eq!(sampled_barrier_for(&compiled, first), Access::FragmentSampled);
        assert_eq!(sampled_barrier_for(&compiled, second), Access::VertexSampled);
    }

    #[test]
    fn one_standing_write_cannot_be_inferred_as_two_descriptor_types() {
        let sampled = PipelineId(0);
        let storage = PipelineId(1);
        let pipelines = PerPipeline::default()
            .with_typed_binding(
                sampled,
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::ShaderStageFlags::FRAGMENT,
                Access::FragmentSampled,
            )
            .with_typed_binding(
                storage,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::FRAGMENT,
                Access::FragmentRead,
            );

        let mut module = Module::default();
        let attachment = target(&mut module);
        let image = target(&mut module);
        let end = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(sampled)
            .bind_image(0, 0, image)
            .draw(3, 1)
            .bind_graphics_pipeline(storage)
            .draw(3, 1)
            .end_rendering();

        assert!(module.compile(&pipelines, end).is_err());
    }

    #[test]
    fn rebinding_allows_reflection_to_infer_a_new_descriptor_type() {
        let sampled = PipelineId(0);
        let storage = PipelineId(1);
        let pipelines = PerPipeline::default()
            .with_typed_binding(
                sampled,
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::ShaderStageFlags::FRAGMENT,
                Access::FragmentSampled,
            )
            .with_typed_binding(
                storage,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::FRAGMENT,
                Access::FragmentRead,
            );

        let mut module = Module::default();
        let attachment = target(&mut module);
        let sampled_image = target(&mut module);
        let storage_image = target(&mut module);
        let end = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .bind_graphics_pipeline(sampled)
            .bind_image(0, 0, sampled_image)
            .draw(3, 1)
            .bind_graphics_pipeline(storage)
            .bind_image(0, 0, storage_image)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&pipelines, end).unwrap();
        let resolved = compiled
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::WriteDescriptor { descriptor_type, .. } => *descriptor_type,
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            resolved,
            [vk::DescriptorType::SAMPLED_IMAGE, vk::DescriptorType::STORAGE_IMAGE]
        );
    }
}
