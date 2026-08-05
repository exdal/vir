use std::collections::HashMap;

use ash::vk;

use crate::{
    Access,
    Buffer,
    ClearValue,
    ColorBlendAttachmentState,
    DomainFlag,
    DynamicStateFlags,
    IR,
    ImageAttachment,
    PassState,
    PipelineId,
    RasterizationState,
    Rect2D,
    RenderingState,
    StateChange,
    SwapChain,
    ValueId,
    Viewport,
    graph::ir,
};

#[derive(Clone, Copy)]
struct ResourceState {
    layout: vk::ImageLayout,
    last_access: ValueId,
}

#[derive(Clone, Copy)]
struct ImageInfo {
    format: vk::Format,
    samples: vk::SampleCountFlags,
    extent: vk::Extent2D,
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

    pub fn import_buffer(&mut self, buffer: &Buffer) -> ValueId {
        self.lower_type(ir::Type::Buffer);
        let size = self.lower_u32(buffer.size() as u32);
        self.emit(IR::ConstructBuffer { buffer: *buffer, size })
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

    pub fn begin_rendering(&mut self, color_attachments: &[ValueId]) -> RenderPass<'_> {
        let id = self.emit(IR::BeginRendering {
            color_attachments: color_attachments.to_vec(),
            render_area: None,
        });
        RenderPass { module: self, id }
    }

    pub fn begin_rendering_area(&mut self, color_attachments: &[ValueId], render_area: vk::Extent2D) -> RenderPass<'_> {
        let render_area = self.lower_constant(ir::Constant::Extent2D(render_area));
        let id = self.emit(IR::BeginRendering {
            color_attachments: color_attachments.to_vec(),
            render_area: Some(render_area),
        });
        RenderPass { module: self, id }
    }

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
                    IR::BindPipeline { pass, .. } | IR::SetState { pass, .. } => {
                        stack.push(*pass);
                    },
                    IR::BindVertexBuffers { pass, buffers, .. } => {
                        buffers.iter().rev().for_each(|v| stack.push(*v));
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
        let mut buffer_states: HashMap<ValueId, ValueId> = HashMap::new();
        let mut next_id = self.instructions.len() as u32;

        let mut region_buffers: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
        let mut open_region: Option<ValueId> = None;
        for (value_id, ir) in &nodes {
            match ir {
                IR::BeginRendering { .. } => open_region = Some(*value_id),
                IR::EndRendering { .. } => open_region = None,
                IR::BindVertexBuffers { buffers, .. } => {
                    let Some(region) = open_region else {
                        continue;
                    };
                    let bound = region_buffers.entry(region).or_default();
                    for buffer in buffers {
                        if !bound.contains(buffer) {
                            bound.push(*buffer);
                        }
                    }
                },
                _ => {},
            }
        }

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
        macro_rules! emit_memory_barrier {
            ($src:expr, $dst:expr) => {{
                let id = alloc!();
                result.push((
                    id,
                    IR::MemoryBarrier {
                        src_access: $src,
                        dst_access: $dst,
                    },
                ));
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
        let mut host_write_id: Option<ValueId> = None;
        let mut attribute_read_id: Option<ValueId> = None;
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

                    for buffer in region_buffers.get(&value_id).into_iter().flatten() {
                        // an imported buffer has no producer inside the graph, so the host is the
                        // only thing that can have written it
                        let last = buffer_states.get(buffer).copied().unwrap_or_else(|| match host_write_id {
                            Some(id) => id,
                            None => {
                                let id = emit_access!(Access::HostWrite);
                                host_write_id = Some(id);
                                id
                            },
                        });

                        let read_id = match attribute_read_id {
                            Some(id) => id,
                            None => {
                                let id = emit_access!(Access::AttributeRead);
                                attribute_read_id = Some(id);
                                id
                            },
                        };

                        // already visible to vertex input from an earlier region
                        if last == read_id {
                            continue;
                        }
                        emit_memory_barrier!(last, read_id);
                        buffer_states.insert(*buffer, read_id);
                    }

                    states.insert(value_id, new_state);
                },

                // the barriers these need were emitted at region entry
                IR::BindPipeline { pass, .. }
                | IR::SetState { pass, .. }
                | IR::BindVertexBuffers { pass, .. }
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

    fn resolve_image(&self, id: ValueId) -> Option<ImageInfo> {
        let mut id = id;

        for _ in 0..256 {
            match self.instructions.get(id.0 as usize)? {
                IR::ConstructImage {
                    format,
                    samples,
                    extent,
                    ..
                } => {
                    let extent = match self.instructions.get(extent.0 as usize)? {
                        IR::Constant(ir::Constant::Extent3D(extent)) => *extent,
                        _ => return None,
                    };

                    return Some(ImageInfo {
                        format: *format,
                        samples: *samples,
                        extent: vk::Extent2D {
                            width: extent.width,
                            height: extent.height,
                        },
                    });
                },
                // every element of an attachment array describes the same image
                IR::Array { elements, .. } => id = *elements.first()?,
                IR::Index { array, .. } => id = *array,
                IR::Clear { attachment, .. } => id = *attachment,
                IR::BeginRendering { color_attachments, .. } => id = *color_attachments.first()?,
                IR::BindPipeline { pass, .. }
                | IR::SetState { pass, .. }
                | IR::BindVertexBuffers { pass, .. }
                | IR::Draw { pass, .. }
                | IR::EndRendering { pass } => id = *pass,
                IR::Acquire { resource, .. } | IR::Release { resource, .. } => id = *resource,
                _ => return None,
            }
        }

        None
    }

    fn resolve_extent_2d(&self, id: ValueId) -> Option<vk::Extent2D> {
        match self.instructions.get(id.0 as usize)? {
            IR::Constant(ir::Constant::Extent2D(extent)) => Some(*extent),
            _ => None,
        }
    }

    fn infer(&self, mut nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        #[derive(Clone)]
        struct InForce {
            state: PassState,
            area: vk::Rect2D,
            pipeline: Option<PipelineId>,
        }

        let mut regions: HashMap<ValueId, InForce> = HashMap::new();

        for (value_id, ir) in nodes.iter_mut() {
            match ir {
                IR::BeginRendering {
                    color_attachments,
                    render_area,
                } => {
                    let mut rendering = RenderingState::default();
                    let mut attachment_extent = None;

                    for (index, attachment) in color_attachments.iter().enumerate() {
                        let Some(image) = self.resolve_image(*attachment) else {
                            tracing::warn!(%attachment, "cannot infer attachment type; pipelines may not match");
                            continue;
                        };

                        if index == 0 {
                            rendering.samples = image.samples;
                            attachment_extent = Some(image.extent);
                        } else if image.samples != rendering.samples {
                            tracing::warn!(
                                %attachment,
                                "attachment sample count differs from the region's first attachment"
                            );
                        }

                        rendering.color_formats.push(image.format);
                    }

                    // framebuffer-relative state resolves against this, so it has to match the
                    // area the region is opened with
                    let extent = match render_area {
                        Some(id) => self.resolve_extent_2d(*id),
                        None => attachment_extent,
                    };
                    if extent.is_none() {
                        tracing::warn!(%value_id, "cannot infer the render area; static viewports may be wrong");
                    }

                    regions.insert(
                        *value_id,
                        InForce {
                            state: PassState::for_rendering(rendering),
                            area: vk::Rect2D::default().extent(extent.unwrap_or_default()),
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

                IR::SetState { pass, change } => {
                    if let Some(mut in_force) = regions.get(pass).cloned() {
                        in_force.state.apply(change.clone());
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::Draw {
                    pass,
                    pipeline,
                    state,
                    dynamic,
                    ..
                } => {
                    if let Some(in_force) = regions.get(pass).cloned() {
                        if in_force.pipeline.is_none() {
                            tracing::warn!(%value_id, "draw with no pipeline bound");
                        }
                        *pipeline = in_force.pipeline;
                        (*state, *dynamic) = in_force.state.resolve(in_force.area);
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::BindVertexBuffers { pass, .. } | IR::EndRendering { pass } => {
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

pub struct RenderPass<'a> {
    module: &'a mut Module,
    id: ValueId,
}

impl RenderPass<'_> {
    pub fn id(&self) -> ValueId { self.id }

    pub fn bind_graphics_pipeline(mut self, pipeline: PipelineId) -> Self {
        self.id = self.module.emit(IR::BindPipeline {
            pass: self.id,
            pipeline,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
        });
        self
    }

    fn set_state(mut self, change: StateChange) -> Self {
        self.id = self.module.emit(IR::SetState { pass: self.id, change });
        self
    }

    pub fn bind_vertex_buffer(self, binding: u32, buffer: ValueId) -> Self {
        self.bind_vertex_buffers(binding, &[buffer], &[0])
    }

    pub fn bind_vertex_buffers(mut self, first_binding: u32, buffers: &[ValueId], offsets: &[u64]) -> Self {
        assert_eq!(
            buffers.len(),
            offsets.len(),
            "every bound vertex buffer needs an offset"
        );

        self.id = self.module.emit(IR::BindVertexBuffers {
            pass: self.id,
            first_binding,
            buffers: buffers.to_vec(),
            offsets: offsets.to_vec(),
        });
        self
    }

    pub fn push_constants<T: Copy>(self, value: &T) -> Self { self.push_constants_at(0, value) }

    pub fn push_constants_at<T: Copy>(self, offset: u32, value: &T) -> Self {
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.push_constant_bytes(offset, bytes)
    }

    pub fn push_constant_bytes(self, offset: u32, data: &[u8]) -> Self {
        self.set_state(StateChange::PushConstants {
            offset,
            data: data.to_vec(),
        })
    }

    pub fn set_primitive_topology(self, topology: vk::PrimitiveTopology) -> Self {
        self.set_state(StateChange::PrimitiveTopology(topology))
    }

    pub fn set_rasterization(self, rasterization: RasterizationState) -> Self {
        self.set_state(StateChange::Rasterization(rasterization))
    }

    pub fn set_dynamic_state(self, dynamic: DynamicStateFlags) -> Self {
        self.set_state(StateChange::DynamicState(dynamic))
    }

    pub fn set_viewport(self, index: u32, viewport: impl Into<Viewport>) -> Self {
        self.set_state(StateChange::Viewport {
            index,
            viewport: viewport.into(),
        })
    }

    pub fn set_scissor(self, index: u32, rect: Rect2D) -> Self { self.set_state(StateChange::Scissor { index, rect }) }

    pub fn set_color_blend(self, index: u32, blend: impl Into<ColorBlendAttachmentState>) -> Self {
        self.set_state(StateChange::ColorBlend {
            index: Some(index),
            blend: blend.into(),
        })
    }

    pub fn broadcast_color_blend(self, blend: impl Into<ColorBlendAttachmentState>) -> Self {
        self.set_state(StateChange::ColorBlend {
            index: None,
            blend: blend.into(),
        })
    }

    pub fn draw(self, vertex_count: u32, instance_count: u32) -> Self {
        self.draw_range(vertex_count, instance_count, 0, 0)
    }

    pub fn draw_range(
        mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32,
    ) -> Self {
        let vertex_count = self.module.lower_u32(vertex_count);
        let instance_count = self.module.lower_u32(instance_count);
        let first_vertex = self.module.lower_u32(first_vertex);
        let first_instance = self.module.lower_u32(first_instance);
        self.id = self.module.emit(IR::Draw {
            pass: self.id,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            pipeline: None,
            state: Default::default(),
            dynamic: Default::default(),
        });
        self
    }

    pub fn end_rendering(self) -> ValueId { self.module.emit(IR::EndRendering { pass: self.id }) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlendPreset, Image, PipelineState, ResolvedViewport};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;

    fn module_with_attachment() -> (Module, ValueId) {
        let mut module = Module::default();
        let attachment = ImageAttachment::new(
            Image::new(vk::Image::null(), None),
            FORMAT,
            vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            vk::SampleCountFlags::TYPE_1,
            vk::ImageLayout::UNDEFINED,
        );
        let (_, construct) = module.lower_image_attachment(&attachment);
        (module, construct)
    }

    fn memory_barriers(module: &Module, instructions: &[ir::Instr]) -> Vec<(Access, Access)> {
        let access = |id: &ValueId| {
            instructions
                .iter()
                .find(|(instr_id, _)| instr_id == id)
                .map(|(_, ir)| ir)
                .or_else(|| module.instructions.get(id.0 as usize))
                .and_then(|ir| match ir {
                    IR::Constant(ir::Constant::Access(a)) => Some(*a),
                    _ => None,
                })
                .expect("barrier operand should be an access constant")
        };

        instructions
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::MemoryBarrier { src_access, dst_access } => Some((access(src_access), access(dst_access))),
                _ => None,
            })
            .collect()
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
    fn binding_a_vertex_buffer_makes_the_host_write_visible_to_vertex_input() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .end_rendering();

        let instructions = module.compile(end);
        assert_eq!(memory_barriers(&module, &instructions), vec![(
            Access::HostWrite,
            Access::AttributeRead
        )]);

        let binds = instructions
            .iter()
            .filter(|(_, ir)| matches!(ir, IR::BindVertexBuffers { .. }))
            .count();
        assert_eq!(binds, 1);

        let position = |predicate: fn(&IR) -> bool| {
            instructions
                .iter()
                .position(|(_, ir)| predicate(ir))
                .expect("instruction should be present")
        };
        assert!(
            position(|ir| matches!(ir, IR::MemoryBarrier { .. }))
                < position(|ir| matches!(ir, IR::BeginRendering { .. }))
        );

        let draws = draws(&instructions);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].0, Some(PipelineId(0)));
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
    }

    #[test]
    fn rebinding_the_same_vertex_buffer_does_not_repeat_the_barrier() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .end_rendering();

        let instructions = module.compile(end);
        assert_eq!(memory_barriers(&module, &instructions).len(), 1);
    }

    #[test]
    fn a_draw_inherits_the_regions_formats_and_the_default_state() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].0, Some(pipeline));
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
        assert_eq!(draws[0].1.rasterization, RasterizationState::default());
        assert_eq!(draws[0].1.topology, vk::PrimitiveTopology::TRIANGLE_LIST);
    }

    #[test]
    fn state_set_after_bind_reaches_the_draw() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::BACK,
                polygon_mode: vk::PolygonMode::LINE,
                ..Default::default()
            })
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.rasterization.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(draws[0].1.rasterization.polygon_mode, vk::PolygonMode::LINE);
    }

    #[test]
    fn a_state_change_between_draws_splits_them_into_two_permutations() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .draw(3, 1)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::FRONT,
                ..Default::default()
            })
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].1.rasterization.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(draws[1].1.rasterization.cull_mode, vk::CullModeFlags::FRONT);
        assert_eq!(draws[0].0, draws[1].0);
        assert_ne!(draws[0].1, draws[1].1);
    }

    #[test]
    fn draws_that_agree_on_state_share_one_permutation() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_primitive_topology(vk::PrimitiveTopology::LINE_STRIP)
            .draw(3, 1)
            .draw(6, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0], draws[1]);
        assert_eq!(draws[0].1.topology, vk::PrimitiveTopology::LINE_STRIP);
    }

    #[test]
    fn rebinding_a_declaration_keeps_the_state_accumulated_so_far() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::BACK,
                ..Default::default()
            })
            .bind_graphics_pipeline(PipelineId(1))
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws[0].0, Some(PipelineId(1)));
        assert_eq!(draws[0].1.rasterization.cull_mode, vk::CullModeFlags::BACK);
    }

    #[test]
    fn a_draw_with_no_bound_pipeline_infers_none() {
        let (mut module, attachment) = module_with_attachment();

        let end = module.begin_rendering(&[attachment]).draw(3, 1).end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws[0].0, None);
    }

    #[test]
    fn a_second_region_over_the_same_attachment_keeps_its_own_state() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let attachment = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .set_rasterization(RasterizationState {
                polygon_mode: vk::PolygonMode::LINE,
                ..Default::default()
            })
            .draw(3, 1)
            .end_rendering();
        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].1.rasterization.polygon_mode, vk::PolygonMode::LINE);
        assert_eq!(draws[1].1.rasterization.polygon_mode, vk::PolygonMode::FILL);
        assert_eq!(draws[1].1.rendering.color_formats, vec![FORMAT]);
    }

    #[test]
    fn a_dynamic_viewport_travels_with_the_draw_instead_of_the_pipeline() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_viewport(0, Rect2D::relative(0.0, 0.0, 0.5, 0.5))
            .draw(3, 1)
            .set_viewport(0, Rect2D::framebuffer())
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(end);
        let dynamic = compiled
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        // both draws share a permutation, only the recorded viewport differs
        let draws = draws(&compiled);
        assert_eq!(draws[0].1, draws[1].1);
        assert!(draws[0].1.viewports.is_empty());
        assert_eq!(dynamic[0].viewports, vec![Rect2D::relative(0.0, 0.0, 0.5, 0.5).into()]);
        assert_eq!(dynamic[1].viewports, vec![Viewport::framebuffer()]);
    }

    #[test]
    fn a_static_viewport_is_resolved_against_the_render_area_and_keys_the_pipeline() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_dynamic_state(DynamicStateFlags::None)
            .set_viewport(0, Rect2D::relative(0.0, 0.0, 0.5, 1.0))
            .set_scissor(0, Rect2D::framebuffer())
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(
            draws[0].1.viewports,
            vec![ResolvedViewport {
                x: 0.0,
                y: 0.0,
                width: (WIDTH / 2) as f32,
                height: HEIGHT as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }]
        );
        assert_eq!(
            draws[0].1.scissors,
            vec![vk::Rect2D::default().extent(vk::Extent2D {
                width: WIDTH,
                height: HEIGHT
            })]
        );
    }

    #[test]
    fn a_static_viewport_follows_the_render_area_a_region_was_opened_with() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering_area(
                &[attachment],
                vk::Extent2D {
                    width: WIDTH / 2,
                    height: HEIGHT / 2,
                },
            )
            .bind_graphics_pipeline(PipelineId(0))
            .set_dynamic_state(DynamicStateFlags::None)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws[0].1.viewports[0].width, (WIDTH / 2) as f32);
        assert_eq!(draws[0].1.viewports[0].height, (HEIGHT / 2) as f32);
    }

    #[test]
    fn push_constants_reach_the_draws_that_follow_them() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .push_constants(&[1.0f32, 2.0])
            .draw(3, 1)
            .push_constants_at(4, &3.0f32)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(end);
        let pushed = compiled
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.push_constants.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        // the second push only touches the tail, so the first four bytes carry over
        assert_eq!(pushed[0].offset, 0);
        assert_eq!(pushed[0].data, [1.0f32, 2.0].map(f32::to_ne_bytes).concat());
        assert_eq!(pushed[1].data, [1.0f32, 3.0].map(f32::to_ne_bytes).concat());

        // and none of it splits the pipeline permutation
        let draws = draws(&compiled);
        assert_eq!(draws[0], draws[1]);
    }

    #[test]
    fn a_draw_recorded_before_a_push_does_not_see_it() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .push_constants(&7u32)
            .draw(3, 1)
            .end_rendering();

        let pushed = module
            .compile(end)
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.push_constants.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(pushed[0].is_empty());
        assert_eq!(pushed[1].data, 7u32.to_ne_bytes());
    }

    #[test]
    fn a_push_before_the_pipeline_bind_still_reaches_the_draw() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .push_constants(&7u32)
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(end);
        let (_, IR::Draw { dynamic, .. }) = compiled
            .iter()
            .find(|(_, ir)| matches!(ir, IR::Draw { .. }))
            .expect("draw should be present")
        else {
            unreachable!()
        };

        assert_eq!(dynamic.push_constants.data, 7u32.to_ne_bytes());
        assert_eq!(draws(&compiled)[0].0, Some(PipelineId(0)));
    }

    #[test]
    fn blend_state_is_per_attachment_and_keys_the_pipeline() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .broadcast_color_blend(BlendPreset::AlphaBlend)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws[0].1.blend, vec![ColorBlendAttachmentState::default()]);
        assert_eq!(draws[1].1.blend, vec![BlendPreset::AlphaBlend.into()]);
        assert_ne!(draws[0].1, draws[1].1);
    }
}
