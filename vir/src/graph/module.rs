use std::collections::HashMap;

use ash::vk;

use crate::{
    Access,
    ClearValue,
    DomainFlag,
    IR,
    ImageAttachment,
    PipelineId,
    PipelineState,
    RasterState,
    RasterStateChange,
    RenderingState,
    SwapChain,
    ValueId,
    graph::ir,
};

#[derive(Clone, Copy)]
struct ResourceState {
    layout: vk::ImageLayout,
    last_access: ValueId,
}

#[derive(Default)]
pub struct Module {
    types: HashMap<ir::Type, ValueId>,
    constants: HashMap<ir::Constant, ValueId>,
    instructions: Vec<IR>,
}

impl Module {
    fn get(&self, id: ValueId) -> &IR { &self.instructions[id.0 as usize] }

    fn resolve_access(&self, id: ValueId) -> Access {
        match self.get(id) {
            IR::Constant(ir::Constant::Access(a)) => *a,
            _ => panic!("{id} is not an Access constant"),
        }
    }

    fn emit(&mut self, ir: IR) -> ValueId {
        let id = ValueId(self.instructions.len() as u32);
        self.instructions.push(ir);
        id
    }

    fn lower_type(&mut self, ty: ir::Type) -> ValueId {
        if let Some(&id) = self.types.get(&ty) {
            return id;
        }
        let id = self.emit(IR::Type(ty));
        self.types.insert(ty, id);
        id
    }

    fn lower_constant(&mut self, constant: ir::Constant) -> ValueId {
        if let Some(&id) = self.constants.get(&constant) {
            return id;
        }
        let id = self.emit(IR::Constant(constant));
        self.constants.insert(constant, id);
        id
    }

    fn lower_u32(&mut self, v: u32) -> ValueId { self.lower_constant(ir::Constant::U32(v)) }

    fn lower_i32(&mut self, v: i32) -> ValueId { self.lower_constant(ir::Constant::I32(v)) }

    fn lower_access(&mut self, access: Access) -> ValueId { self.lower_constant(ir::Constant::Access(access)) }

    fn lower_array(&mut self, ty: ValueId, elements: Vec<ValueId>) -> ValueId { self.emit(IR::Array { ty, elements }) }

    fn lower_image_attachment(&mut self, attachment: &ImageAttachment) -> (ValueId, ValueId) {
        let extent = self.lower_constant(ir::Constant::Extent3D(attachment.extent()));
        let base_level = self.lower_u32(attachment.base_level());
        let level_count = self.lower_u32(attachment.level_count());
        let base_layer = self.lower_u32(attachment.base_layer());
        let layer_count = self.lower_u32(attachment.layer_count());

        let ty_instr = self.lower_type(ir::Type::Image {
            format: attachment.format(),
            samples: attachment.samples(),
        });
        let view_type = if attachment.layer_count() > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let construct_instr = self.emit(IR::ConstructImage {
            image: attachment.image().clone(),
            image_view: attachment.image_view(),
            view_type,
            extent,
            format: attachment.format(),
            samples: attachment.samples(),
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: vk::ImageUsageFlags::empty(),
        });

        (ty_instr, construct_instr)
    }

    pub fn acquire_next_image(&mut self, swapchain: &SwapChain) -> ValueId {
        let (ty_instr, attach_instr): (Vec<_>, Vec<_>) = swapchain
            .attachments
            .iter()
            .map(|attach| self.lower_image_attachment(attach))
            .unzip();

        let ty = ty_instr[0];
        assert!(ty_instr.iter().all(|&i| i == ty));
        let attachments = self.lower_array(ty, attach_instr);

        let image_index = self.emit(IR::AcquireNextImage {
            swapchain: swapchain.handle,
            attachments,
            present_semaphores: swapchain.semaphores.clone(),
        });

        self.emit(IR::Index {
            array: attachments,
            index: image_index,
        })
    }

    pub fn release(&mut self, resource: ValueId, access: Access, dst_domain: DomainFlag) -> ValueId {
        let access = self.lower_access(access);
        self.emit(IR::Release {
            resource,
            access,
            dst_domain,
        })
    }

    pub fn present(&mut self, attachment: ValueId) -> ValueId {
        self.release(attachment, Access::Present, DomainFlag::Present)
    }

    pub fn clear(&mut self, attachment: ValueId, color: ClearValue) -> ValueId {
        let color = self.lower_constant(ir::Constant::ClearValue(color));
        self.emit(IR::Clear { attachment, color })
    }

    pub fn begin_rendering(&mut self, color_attachments: &[ValueId]) -> ValueId {
        self.emit(IR::BeginRendering {
            color_attachments: color_attachments.to_vec(),
            render_area: None,
        })
    }

    pub fn begin_rendering_area(&mut self, color_attachments: &[ValueId], render_area: vk::Extent2D) -> ValueId {
        let render_area = self.lower_constant(ir::Constant::Extent2D(render_area));
        self.emit(IR::BeginRendering {
            color_attachments: color_attachments.to_vec(),
            render_area: Some(render_area),
        })
    }

    pub fn bind_pipeline(&mut self, pass: ValueId, pipeline: PipelineId) -> ValueId {
        self.emit(IR::BindPipeline {
            pass,
            pipeline,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
        })
    }

    fn set_raster_state(&mut self, pass: ValueId, change: RasterStateChange) -> ValueId {
        self.emit(IR::SetRasterState { pass, change })
    }

    pub fn set_topology(&mut self, pass: ValueId, topology: vk::PrimitiveTopology) -> ValueId {
        self.set_raster_state(pass, RasterStateChange::Topology(topology))
    }

    pub fn set_polygon_mode(&mut self, pass: ValueId, polygon_mode: vk::PolygonMode) -> ValueId {
        self.set_raster_state(pass, RasterStateChange::PolygonMode(polygon_mode))
    }

    pub fn set_cull_mode(&mut self, pass: ValueId, cull_mode: vk::CullModeFlags) -> ValueId {
        self.set_raster_state(pass, RasterStateChange::CullMode(cull_mode))
    }

    pub fn set_front_face(&mut self, pass: ValueId, front_face: vk::FrontFace) -> ValueId {
        self.set_raster_state(pass, RasterStateChange::FrontFace(front_face))
    }

    pub fn draw(&mut self, pass: ValueId, vertex_count: u32, instance_count: u32) -> ValueId {
        self.draw_range(pass, vertex_count, instance_count, 0, 0)
    }

    pub fn draw_range(
        &mut self, pass: ValueId, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32,
    ) -> ValueId {
        let vertex_count = self.lower_u32(vertex_count);
        let instance_count = self.lower_u32(instance_count);
        let first_vertex = self.lower_u32(first_vertex);
        let first_instance = self.lower_u32(first_instance);
        self.emit(IR::Draw {
            pass,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            pipeline: None,
            state: PipelineState::default(),
        })
    }

    pub fn end_rendering(&mut self, pass: ValueId) -> ValueId { self.emit(IR::EndRendering { pass }) }

    pub fn compile(&self, id: ValueId) -> Vec<ir::Instr> {
        let linearized = self.topo_sort(id);
        let synced = self.sync(linearized);
        self.infer(synced)
    }

    fn topo_sort(&self, value_id: ValueId) -> Vec<ir::Instr> {
        let mut nodes = Vec::new();
        let mut stack = vec![value_id];
        let mut visited = std::collections::HashSet::new();
        let mut processed = std::collections::HashSet::new();

        while let Some(id) = stack.pop() {
            if processed.contains(&id) {
                continue;
            }

            let ir = self.get(id);
            if visited.insert(id) {
                stack.push(id);
                match ir {
                    IR::Constant(_) => {},
                    IR::Type(_) => {},
                    IR::Array { ty, elements } => {
                        stack.push(*ty);
                        elements.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::ConstructBuffer { size, .. } => {
                        stack.push(*size);
                    },
                    IR::ConstructImage {
                        extent,
                        base_level,
                        level_count,
                        base_layer,
                        layer_count,
                        ..
                    } => {
                        stack.push(*layer_count);
                        stack.push(*base_layer);
                        stack.push(*level_count);
                        stack.push(*base_level);
                        stack.push(*extent);
                    },
                    IR::AcquireNextImage { attachments, .. } => {
                        stack.push(*attachments);
                    },
                    IR::Index { array, index } => {
                        stack.push(*index);
                        stack.push(*array);
                    },
                    IR::Acquire { access, resource, .. } => {
                        stack.push(*access);
                        stack.push(*resource);
                    },
                    IR::Release { access, resource, .. } => {
                        stack.push(*access);
                        stack.push(*resource);
                    },
                    IR::CallOpaque { args, returns, .. } => {
                        stack.push(*returns);
                        stack.push(*args);
                    },
                    IR::Clear { color, attachment } => {
                        stack.push(*color);
                        stack.push(*attachment);
                    },
                    IR::BeginRendering {
                        color_attachments,
                        render_area,
                    } => {
                        render_area.iter().for_each(|v| stack.push(*v));
                        color_attachments.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::BindPipeline { pass, .. } | IR::SetRasterState { pass, .. } => {
                        stack.push(*pass);
                    },
                    IR::Draw {
                        pass,
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                        ..
                    } => {
                        stack.push(*first_instance);
                        stack.push(*first_vertex);
                        stack.push(*instance_count);
                        stack.push(*vertex_count);
                        stack.push(*pass);
                    },
                    IR::EndRendering { pass } => {
                        stack.push(*pass);
                    },
                    IR::MemoryBarrier { src_access, dst_access } => {
                        stack.push(*dst_access);
                        stack.push(*src_access);
                    },
                    IR::ImageBarrier {
                        src_access,
                        dst_access,
                        value,
                        ..
                    } => {
                        stack.push(*value);
                        stack.push(*dst_access);
                        stack.push(*src_access);
                    },
                }
            } else {
                processed.insert(id);
                nodes.push((id, ir.clone()));
            }
        }

        nodes
    }

    fn sync(&self, nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        let mut result = Vec::with_capacity(nodes.len() * 2);
        let mut states: HashMap<ValueId, ResourceState> = HashMap::new();
        let mut next_id = self.instructions.len() as u32;

        macro_rules! alloc {
            () => {{
                let id = ValueId(next_id);
                next_id += 1;
                id
            }};
        }
        macro_rules! emit_access {
            ($access:expr) => {{
                let id = alloc!();
                result.push((id, IR::Constant(ir::Constant::Access($access))));
                id
            }};
        }
        macro_rules! emit_barrier {
            ($src:expr, $dst:expr, $old:expr, $new:expr, $res:expr) => {{
                let id = alloc!();
                result.push((
                    id,
                    IR::ImageBarrier {
                        src_access: $src,
                        dst_access: $dst,
                        old_layout: $old,
                        new_layout: $new,
                        value: $res,
                    },
                ));
            }};
        }

        let no_access_id = emit_access!(Access::None);
        let undefined = ResourceState {
            layout: vk::ImageLayout::UNDEFINED,
            last_access: no_access_id,
        };

        for (value_id, ir) in nodes {
            match &ir {
                IR::Index { .. } | IR::ConstructImage { .. } => {
                    states.insert(value_id, undefined);
                },

                IR::Acquire { resource, access } => {
                    let state = states.get(resource).copied().unwrap_or(undefined);
                    let new_layout = self.resolve_access(*access).into();
                    emit_barrier!(state.last_access, *access, state.layout, new_layout, *resource);
                    let new_state = ResourceState {
                        layout: new_layout,
                        last_access: *access,
                    };
                    states.insert(*resource, new_state);
                    states.insert(value_id, new_state);
                },

                IR::Clear { attachment, .. } => {
                    let state = states.get(attachment).copied().unwrap_or(undefined);
                    let access_id = emit_access!(Access::Clear);
                    emit_barrier!(
                        state.last_access,
                        access_id,
                        state.layout,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        *attachment
                    );
                    let new_state = ResourceState {
                        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        last_access: access_id,
                    };
                    states.insert(*attachment, new_state);
                    states.insert(value_id, new_state);
                },

                IR::BeginRendering { color_attachments, .. } => {
                    let access_id = emit_access!(Access::ColorRW);
                    let new_state = ResourceState {
                        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        last_access: access_id,
                    };

                    for attachment in color_attachments {
                        let state = states.get(attachment).copied().unwrap_or(undefined);
                        emit_barrier!(
                            state.last_access,
                            access_id,
                            state.layout,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            *attachment
                        );
                        states.insert(*attachment, new_state);
                    }

                    states.insert(value_id, new_state);
                },

                IR::BindPipeline { pass, .. }
                | IR::SetRasterState { pass, .. }
                | IR::Draw { pass, .. }
                | IR::EndRendering { pass } => {
                    let state = states.get(pass).copied().unwrap_or(undefined);
                    states.insert(value_id, state);
                },

                IR::Release { resource, access, .. } => {
                    let state = states.get(resource).copied().unwrap_or(undefined);
                    let new_layout = self.resolve_access(*access).into();
                    emit_barrier!(state.last_access, *access, state.layout, new_layout, *resource);
                    states.insert(
                        *resource,
                        ResourceState {
                            layout: new_layout,
                            last_access: *access,
                        },
                    );
                },

                _ => {},
            }

            result.push((value_id, ir));
        }

        result
    }

    fn resolve_image_type(&self, id: ValueId) -> Option<(vk::Format, vk::SampleCountFlags)> {
        let mut id = id;

        for _ in 0..64 {
            match self.instructions.get(id.0 as usize)? {
                IR::ConstructImage { format, samples, .. } => return Some((*format, *samples)),
                IR::Type(ir::Type::Image { format, samples }) => return Some((*format, *samples)),
                IR::Array { ty, .. } => id = *ty,
                IR::Index { array, .. } => id = *array,
                IR::Clear { attachment, .. } => id = *attachment,
                IR::BeginRendering { color_attachments, .. } => id = *color_attachments.first()?,
                IR::BindPipeline { pass, .. }
                | IR::SetRasterState { pass, .. }
                | IR::Draw { pass, .. }
                | IR::EndRendering { pass } => id = *pass,
                IR::Acquire { resource, .. } | IR::Release { resource, .. } => id = *resource,
                _ => return None,
            }
        }

        None
    }

    fn infer(&self, mut nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        #[derive(Clone, Default)]
        struct InForce {
            state: PipelineState,
            pipeline: Option<PipelineId>,
        }

        let mut regions: HashMap<ValueId, InForce> = HashMap::new();

        for (value_id, ir) in nodes.iter_mut() {
            match ir {
                IR::BeginRendering { color_attachments, .. } => {
                    let mut rendering = RenderingState::default();

                    for (index, attachment) in color_attachments.iter().enumerate() {
                        let Some((format, samples)) = self.resolve_image_type(*attachment) else {
                            tracing::warn!(%attachment, "cannot infer attachment type; pipelines may not match");
                            continue;
                        };

                        if index == 0 {
                            rendering.samples = samples;
                        } else if samples != rendering.samples {
                            tracing::warn!(
                                %attachment,
                                "attachment sample count differs from the region's first attachment"
                            );
                        }

                        rendering.color_formats.push(format);
                    }

                    regions.insert(
                        *value_id,
                        InForce {
                            state: PipelineState {
                                rendering,
                                raster: RasterState::default(),
                            },
                            pipeline: None,
                        },
                    );
                },

                IR::BindPipeline { pass, pipeline, .. } => {
                    if let Some(mut in_force) = regions.get(pass).cloned() {
                        in_force.pipeline = Some(*pipeline);
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::SetRasterState { pass, change } => {
                    if let Some(mut in_force) = regions.get(pass).cloned() {
                        in_force.state.raster.apply(*change);
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::Draw {
                    pass, pipeline, state, ..
                } => {
                    if let Some(in_force) = regions.get(pass).cloned() {
                        if in_force.pipeline.is_none() {
                            tracing::warn!(%value_id, "draw with no pipeline bound");
                        }
                        *pipeline = in_force.pipeline;
                        *state = in_force.state.clone();
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::EndRendering { pass } => {
                    if let Some(in_force) = regions.get(pass).cloned() {
                        regions.insert(*value_id, in_force);
                    }
                },

                _ => {},
            }
        }

        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Image;

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

    fn module_with_attachment() -> (Module, ValueId) {
        let mut module = Module::default();
        let attachment = ImageAttachment::new(
            Image::new(vk::Image::null(), None),
            FORMAT,
            vk::Extent3D::default().width(4).height(4).depth(1),
            vk::SampleCountFlags::TYPE_1,
            vk::ImageLayout::UNDEFINED,
        );
        let (_, construct) = module.lower_image_attachment(&attachment);
        (module, construct)
    }

    fn draws(instructions: &[ir::Instr]) -> Vec<(Option<PipelineId>, PipelineState)> {
        instructions
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { pipeline, state, .. } => Some((*pipeline, state.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_draw_inherits_the_regions_formats_and_the_default_raster_state() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let pass = module.begin_rendering(&[attachment]);
        let pass = module.bind_pipeline(pass, pipeline);
        let pass = module.draw(pass, 3, 1);
        let end = module.end_rendering(pass);

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].0, Some(pipeline));
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
        assert_eq!(draws[0].1.raster, RasterState::default());
    }

    #[test]
    fn state_set_after_bind_reaches_the_draw() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let pass = module.begin_rendering(&[attachment]);
        let pass = module.bind_pipeline(pass, pipeline);
        let pass = module.set_cull_mode(pass, vk::CullModeFlags::BACK);
        let pass = module.set_polygon_mode(pass, vk::PolygonMode::LINE);
        let pass = module.draw(pass, 3, 1);
        let end = module.end_rendering(pass);

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.raster.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(draws[0].1.raster.polygon_mode, vk::PolygonMode::LINE);
    }

    #[test]
    fn a_state_change_between_draws_splits_them_into_two_permutations() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let pass = module.begin_rendering(&[attachment]);
        let pass = module.bind_pipeline(pass, pipeline);
        let pass = module.draw(pass, 3, 1);
        let pass = module.set_cull_mode(pass, vk::CullModeFlags::FRONT);
        let pass = module.draw(pass, 3, 1);
        let end = module.end_rendering(pass);

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].1.raster.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(draws[1].1.raster.cull_mode, vk::CullModeFlags::FRONT);
        assert_eq!(draws[0].0, draws[1].0);
        assert_ne!(draws[0].1, draws[1].1);
    }

    #[test]
    fn draws_that_agree_on_state_share_one_permutation() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let pass = module.begin_rendering(&[attachment]);
        let pass = module.bind_pipeline(pass, pipeline);
        let pass = module.set_cull_mode(pass, vk::CullModeFlags::BACK);
        let pass = module.draw(pass, 3, 1);
        let pass = module.draw(pass, 6, 1);
        let end = module.end_rendering(pass);

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0], draws[1]);
    }

    #[test]
    fn rebinding_a_declaration_keeps_the_state_accumulated_so_far() {
        let (mut module, attachment) = module_with_attachment();

        let pass = module.begin_rendering(&[attachment]);
        let pass = module.set_cull_mode(pass, vk::CullModeFlags::BACK);
        let pass = module.bind_pipeline(pass, PipelineId(1));
        let pass = module.draw(pass, 3, 1);
        let end = module.end_rendering(pass);

        let draws = draws(&module.compile(end));
        assert_eq!(draws[0].0, Some(PipelineId(1)));
        assert_eq!(draws[0].1.raster.cull_mode, vk::CullModeFlags::BACK);
    }

    #[test]
    fn a_draw_with_no_bound_pipeline_infers_none() {
        let (mut module, attachment) = module_with_attachment();

        let pass = module.begin_rendering(&[attachment]);
        let pass = module.draw(pass, 3, 1);
        let end = module.end_rendering(pass);

        let draws = draws(&module.compile(end));
        assert_eq!(draws[0].0, None);
    }
}
