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
/// modules may still compile, but descriptor writes need reflected bindings to infer their access.
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
    written: BTreeMap<(u32, u32), (ValueId, Descriptor)>,
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
                ..
            } => {
                failed |= !write_is_sound(program, value_id, &table, *set, *binding, descriptor);
                table.written.insert((*set, *binding), (*value_id, *descriptor));
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

    if !names_an_image(program.instructions(), descriptor.image()) {
        tracing::error!(
            %value_id,
            set,
            binding,
            image = %descriptor.image(),
            "a descriptor is written from a value that is not an image"
        );
        return false;
    }

    if descriptor.sampler().is_some_and(|sampler| sampler.is_null()) {
        tracing::error!(%value_id, set, binding, "a combined image sampler is written with a null sampler");
        return false;
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

        let Some((written_at, descriptor)) = table.written.get(&(binding.set, binding.binding)) else {
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

        if descriptor.descriptor_type() != binding.descriptor_type {
            tracing::error!(
                %value_id,
                %pipeline,
                written_at = %written_at,
                set = binding.set,
                binding = binding.binding,
                declared = ?binding.descriptor_type,
                written = ?descriptor.descriptor_type(),
                "the descriptor written here is not the type the pipeline declares"
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
    let mut id = id;

    for _ in 0..MAX_RESOLVE_DEPTH {
        let Some((_, ir)) = instructions.iter().find(|(instr, _)| *instr == id) else {
            return false;
        };

        id = match underlying_object(ir) {
            UnderlyingObject::Base => return matches!(ir, IR::ConstructImage { .. } | IR::SwapchainImage { .. }),
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
    use crate::{Access, ImageInfo, Module, ValueId};

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
        fn with_binding(mut self, pipeline: PipelineId, stages: vk::ShaderStageFlags) -> Self {
            self.bindings.insert(
                pipeline,
                vec![DescriptorBinding {
                    set: 0,
                    binding: 0,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    count: 1,
                    variable_count: false,
                    stages,
                    access: Access::sampled_by(stages),
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

        assert!(module.compile(&none, end).is_ok());
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
                IR::WriteDescriptor { descriptor, .. } => Some(descriptor.image()),
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
}
