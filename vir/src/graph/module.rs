use std::{collections::HashMap, sync::Arc};

use ash::vk;

use crate::{
    Access,
    Buffer,
    BufferImageCopy,
    ClearValue,
    ColorBlendAttachmentState,
    DepthState,
    DomainFlag,
    DynamicStateFlags,
    IR,
    Image,
    ImageAttachment,
    ImageInfo,
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
    access: Access,
}

#[derive(Clone, Copy)]
struct ImageType {
    format: vk::Format,
    samples: vk::SampleCountFlags,
    extent: vk::Extent3D,
}

fn name_of(name: &str) -> ir::Name { (!name.is_empty()).then(|| Arc::from(name)) }

/// `count` scaled and rounded up, since a fraction of an invocation still has to be dispatched.
fn scaled(count: u64, scale: f32) -> u32 {
    let scaled = (count as f64 * scale as f64).ceil();
    if !scaled.is_finite() {
        tracing::warn!(
            count,
            scale,
            "invocation count does not scale to a number; it is taken to be none"
        );
        return 0;
    }

    scaled.clamp(0.0, u32::MAX as f64) as u32
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

    fn lower_i32(&mut self, v: i32) -> ValueId { self.lower_constant(ir::Constant::I32(v)) }

    fn lower_u32(&mut self, v: u32) -> ValueId { self.lower_constant(ir::Constant::U32(v)) }

    fn lower_access(&mut self, access: Access) -> ValueId { self.lower_constant(ir::Constant::Access(access)) }

    fn lower_array(&mut self, ty: ValueId, elements: Vec<ValueId>) -> ValueId { self.emit(IR::Array { ty, elements }) }

    fn lower_image_attachment(&mut self, attachment: &ImageAttachment, name: ir::Name) -> (ValueId, ValueId) {
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
            image: *attachment.image(),
            image_view: attachment.image_view(),
            view_type,
            extent,
            format: attachment.format(),
            samples: attachment.samples(),
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: attachment.image().usage(),
            initial_layout: attachment.layout(),
            name,
        });

        (ty_instr, construct_instr)
    }

    pub fn transient_image(&mut self, info: &ImageInfo) -> ValueId {
        let extent = self.lower_constant(ir::Constant::Extent3D(info.extent));
        let base_level = self.lower_u32(0);
        let level_count = self.lower_u32(info.mip_levels);
        let base_layer = self.lower_u32(0);
        let layer_count = self.lower_u32(info.array_layers);

        self.lower_type(ir::Type::Image {
            format: info.format,
            samples: info.samples,
        });
        let view_type = if info.array_layers > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };

        self.emit(IR::ConstructImage {
            image: Image::default(),
            image_view: vk::ImageView::null(),
            view_type,
            extent,
            format: info.format,
            samples: info.samples,
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: info.usage,
            initial_layout: vk::ImageLayout::UNDEFINED,
            name: name_of(&info.name),
        })
    }

    pub fn import_image(&mut self, image: &Image, layout: vk::ImageLayout) -> ValueId {
        self.import_attachment(&ImageAttachment::from_image(image, layout))
    }

    pub fn import_attachment(&mut self, attachment: &ImageAttachment) -> ValueId {
        self.lower_image_attachment(attachment, None).1
    }

    pub fn import_buffer(&mut self, buffer: &Buffer) -> ValueId {
        self.lower_type(ir::Type::Buffer);
        let size = self.lower_u32(buffer.size() as u32);
        self.emit(IR::ConstructBuffer {
            buffer: *buffer,
            size,
            name: None,
        })
    }

    /// Attaches a debug name to a resource or a rendering region, which is what the dump
    /// prints in place of the bare value id.
    pub fn set_name(&mut self, id: ValueId, name: impl Into<Arc<str>>) -> ValueId {
        match self.instructions.get_mut(id.0 as usize) {
            Some(
                IR::ConstructImage { name: slot, .. }
                | IR::ConstructBuffer { name: slot, .. }
                | IR::BeginRendering { name: slot, .. }
                | IR::BeginCompute { name: slot, .. },
            ) => *slot = Some(name.into()),
            _ => tracing::warn!(%id, "value cannot carry a name; the name is dropped"),
        }

        id
    }

    pub fn acquire_next_image(&mut self, swapchain: &SwapChain) -> ValueId {
        let (ty_instr, attach_instr): (Vec<_>, Vec<_>) = swapchain
            .attachments
            .iter()
            .enumerate()
            .map(|(index, attach)| self.lower_image_attachment(attach, Some(format!("swapchain#{index}").into())))
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

    pub fn blit(&mut self, src: ValueId, dst: ValueId) -> ValueId { self.blit_filtered(src, dst, vk::Filter::LINEAR) }

    pub fn blit_filtered(&mut self, src: ValueId, dst: ValueId, filter: vk::Filter) -> ValueId {
        self.emit(IR::Blit { src, dst, filter })
    }

    /// Fills the whole of `image` from `buffer`, which has to hold the image tightly packed.
    pub fn copy_buffer_to_image(&mut self, buffer: ValueId, image: ValueId) -> ValueId {
        let Some(extent) = self.resolve_image(image).map(|image| image.extent) else {
            tracing::warn!(%image, "cannot infer the extent to copy into; the copy is dropped");
            return image;
        };

        self.copy_buffer_to_image_region(buffer, image, BufferImageCopy::whole(extent))
    }

    pub fn copy_buffer_to_image_region(&mut self, buffer: ValueId, image: ValueId, region: BufferImageCopy) -> ValueId {
        if region.is_empty() {
            tracing::warn!(%image, "copy region is empty; the copy is dropped");
            return image;
        }

        self.emit(IR::CopyBufferToImage { buffer, image, region })
    }

    pub fn clear(&mut self, attachment: ValueId, color: ClearValue) -> ValueId {
        let color = self.lower_constant(ir::Constant::ClearValue(color));
        self.emit(IR::Clear { attachment, color })
    }

    pub fn begin_rendering(&mut self, color_attachments: &[ValueId]) -> RenderPass<'_> {
        self.begin_rendering_with(color_attachments, None, None)
    }

    pub fn begin_rendering_depth(&mut self, color_attachments: &[ValueId], depth: ValueId) -> RenderPass<'_> {
        self.begin_rendering_with(color_attachments, Some(depth), None)
    }

    pub fn begin_rendering_area(&mut self, color_attachments: &[ValueId], render_area: vk::Extent2D) -> RenderPass<'_> {
        let render_area = self.lower_constant(ir::Constant::Extent2D(render_area));
        self.begin_rendering_with(color_attachments, None, Some(render_area))
    }

    pub fn begin_rendering_depth_area(
        &mut self, color_attachments: &[ValueId], depth: ValueId, render_area: vk::Extent2D,
    ) -> RenderPass<'_> {
        let render_area = self.lower_constant(ir::Constant::Extent2D(render_area));
        self.begin_rendering_with(color_attachments, Some(depth), Some(render_area))
    }

    fn begin_rendering_with(
        &mut self, color_attachments: &[ValueId], depth_attachment: Option<ValueId>, render_area: Option<ValueId>,
    ) -> RenderPass<'_> {
        let id = self.emit(IR::BeginRendering {
            color_attachments: color_attachments.to_vec(),
            depth_attachment,
            render_area,
            name: None,
        });
        RenderPass {
            module: self,
            id,
            begin: id,
        }
    }

    pub fn begin_compute(&mut self) -> ComputePass<'_> {
        let id = self.emit(IR::BeginCompute {
            resources: Vec::new(),
            name: None,
        });
        ComputePass {
            module: self,
            id,
            begin: id,
        }
    }

    pub fn compile(&self, id: ValueId) -> Vec<ir::Instr> { self.compile_all(&[id]) }

    pub fn compile_all(&self, ids: &[ValueId]) -> Vec<ir::Instr> {
        let linearized = self.topo_sort(ids);
        let mut synced = self.sync(linearized);
        self.infer_usage(&mut synced);
        self.infer(synced)
    }

    fn topo_sort(&self, roots: &[ValueId]) -> Vec<ir::Instr> {
        let mut nodes = Vec::new();
        let mut stack = roots.iter().rev().copied().collect::<Vec<_>>();
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
                    IR::Blit { src, dst, .. } => {
                        stack.push(*dst);
                        stack.push(*src);
                    },
                    IR::CopyBufferToImage { buffer, image, .. } => {
                        stack.push(*image);
                        stack.push(*buffer);
                    },
                    IR::BeginRendering {
                        color_attachments,
                        depth_attachment,
                        render_area,
                        ..
                    } => {
                        render_area.iter().for_each(|v| stack.push(*v));
                        depth_attachment.iter().for_each(|v| stack.push(*v));
                        color_attachments.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::BindPipeline { pass, .. } | IR::SetState { pass, .. } => {
                        stack.push(*pass);
                    },
                    IR::BindVertexBuffers { pass, buffers, .. } => {
                        stack.push(*pass);
                        buffers.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::BindIndexBuffer { pass, buffer, .. } => {
                        stack.push(*pass);
                        stack.push(*buffer);
                    },
                    IR::SampleImage { pass, image } => {
                        stack.push(*pass);
                        stack.push(*image);
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
                    IR::DrawIndexed {
                        pass,
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                        ..
                    } => {
                        stack.push(*first_instance);
                        stack.push(*vertex_offset);
                        stack.push(*first_index);
                        stack.push(*instance_count);
                        stack.push(*index_count);
                        stack.push(*pass);
                    },
                    IR::EndRendering { pass } => {
                        stack.push(*pass);
                    },
                    IR::BeginCompute { resources, .. } => {
                        resources.iter().rev().for_each(|(id, _)| stack.push(*id));
                    },
                    IR::Dispatch { pass, size, .. } => {
                        match size {
                            ir::DispatchSize::Groups { x, y, z } | ir::DispatchSize::Invocations { x, y, z } => {
                                stack.push(*z);
                                stack.push(*y);
                                stack.push(*x);
                            },
                            ir::DispatchSize::Indirect { buffer, .. } => stack.push(*buffer),
                        }
                        stack.push(*pass);
                    },
                    IR::EndCompute { pass } => {
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

        // what a rendering region reads, gathered up front: a layout transition cannot be
        // recorded inside a region, and a host write only has to be made visible once, so both
        // resolve to barriers emitted before the region opens
        let mut region_buffers: HashMap<ValueId, Vec<(ValueId, Access)>> = HashMap::new();
        let mut region_images: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
        let mut open_region: Option<ValueId> = None;
        for (value_id, ir) in &nodes {
            let (buffers, access) = match ir {
                IR::BeginRendering { .. } => {
                    open_region = Some(*value_id);
                    continue;
                },
                IR::EndRendering { .. } => {
                    open_region = None;
                    continue;
                },
                IR::SampleImage { image, .. } => {
                    if let Some(region) = open_region {
                        let sampled = region_images.entry(region).or_default();
                        let root = self.resource_root(*image);
                        if !sampled.contains(&root) {
                            sampled.push(root);
                        }
                    }
                    continue;
                },
                IR::BindVertexBuffers { buffers, .. } => (buffers.as_slice(), Access::AttributeRead),
                IR::BindIndexBuffer { buffer, .. } => (std::slice::from_ref(buffer), Access::IndexRead),
                _ => continue,
            };

            let Some(region) = open_region else {
                continue;
            };
            let bound = region_buffers.entry(region).or_default();
            for buffer in buffers {
                let root = self.resource_root(*buffer);
                if !bound.iter().any(|(bound, _)| *bound == root) {
                    bound.push((root, access));
                }
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
        let mut read_access_ids: HashMap<Access, ValueId> = HashMap::new();

        // the constant for an access a resource is only ever read at is worth sharing, since
        // every barrier that waits on the same read names it
        macro_rules! read_access {
            ($access:expr) => {{
                let access: Access = $access;
                match read_access_ids.get(&access).copied() {
                    Some(id) => id,
                    None => {
                        let id = emit_access!(access);
                        read_access_ids.insert(access, id);
                        id
                    },
                }
            }};
        }

        /// A buffer the host filled is visible to the GPU only after a barrier, and stays
        /// visible until something writes it again, which only a dispatch does today.
        macro_rules! buffer_barrier {
            ($buffer:expr, $access:expr) => {{
                let buffer = $buffer;
                let access: Access = $access;
                let last = match buffer_states.get(&buffer).copied() {
                    Some(id) => id,
                    None => match host_write_id {
                        Some(id) => id,
                        None => {
                            let id = emit_access!(Access::HostWrite);
                            host_write_id = Some(id);
                            id
                        },
                    },
                };

                // a write is never shared, so it always ends up with a barrier of its own
                let next_id = match access.writes() {
                    true => emit_access!(access),
                    false => read_access!(access),
                };
                if last != next_id {
                    emit_memory_barrier!(last, next_id);
                    buffer_states.insert(buffer, next_id);
                }
            }};
        }

        let undefined = ResourceState {
            layout: vk::ImageLayout::UNDEFINED,
            last_access: no_access_id,
            access: Access::None,
        };

        macro_rules! transition {
            ($resource:expr, $access_id:expr, $access:expr) => {{
                let resource = $resource;
                let access: Access = $access;
                let root = self.resource_root(resource);
                let state = states.get(&root).copied().unwrap_or(undefined);
                let new_layout: vk::ImageLayout = access.into();

                let new_state = if state.layout == new_layout && !state.access.writes() && !access.writes() {
                    let merged = state.access | access;
                    let last_access = if merged == state.access {
                        state.last_access
                    } else {
                        emit_access!(merged)
                    };
                    ResourceState {
                        layout: new_layout,
                        last_access,
                        access: merged,
                    }
                } else {
                    emit_barrier!(state.last_access, $access_id, state.layout, new_layout, resource);
                    ResourceState {
                        layout: new_layout,
                        last_access: $access_id,
                        access,
                    }
                };

                states.insert(root, new_state);
            }};
        }

        for (value_id, ir) in nodes {
            match &ir {
                IR::ConstructImage { initial_layout, .. } => {
                    states.insert(
                        value_id,
                        ResourceState {
                            layout: *initial_layout,
                            last_access: no_access_id,
                            access: Access::None,
                        },
                    );
                },

                IR::Acquire { resource, access } => {
                    transition!(*resource, *access, self.resolve_access(*access));
                },

                IR::Clear { attachment, .. } => {
                    let access_id = emit_access!(Access::Clear);
                    transition!(*attachment, access_id, Access::Clear);
                },

                IR::Blit { src, dst, .. } => {
                    let read_id = emit_access!(Access::BlitRead);
                    transition!(*src, read_id, Access::BlitRead);
                    let write_id = emit_access!(Access::BlitWrite);
                    transition!(*dst, write_id, Access::BlitWrite);
                },

                IR::CopyBufferToImage { buffer, image, .. } => {
                    buffer_barrier!(self.resource_root(*buffer), Access::CopyRead);
                    let write_id = emit_access!(Access::CopyWrite);
                    transition!(*image, write_id, Access::CopyWrite);
                },

                IR::BeginRendering {
                    color_attachments,
                    depth_attachment,
                    ..
                } => {
                    for image in region_images.get(&value_id).cloned().into_iter().flatten() {
                        let sampled_id = read_access!(Access::FragmentSampled);
                        transition!(image, sampled_id, Access::FragmentSampled);
                    }

                    let access_id = emit_access!(Access::ColorRW);
                    for attachment in color_attachments {
                        transition!(*attachment, access_id, Access::ColorRW);
                    }

                    if let Some(depth) = depth_attachment {
                        let depth_id = emit_access!(Access::DepthStencilRW);
                        transition!(*depth, depth_id, Access::DepthStencilRW);
                    }

                    for (buffer, access) in region_buffers.get(&value_id).cloned().into_iter().flatten() {
                        buffer_barrier!(buffer, access);
                    }
                },

                IR::BeginCompute { resources, .. } => {
                    for (resource, access) in resources {
                        if self.is_buffer(*resource) {
                            buffer_barrier!(self.resource_root(*resource), *access);
                            continue;
                        }

                        let access_id = match access.writes() {
                            true => emit_access!(*access),
                            false => read_access!(*access),
                        };
                        transition!(*resource, access_id, *access);
                    }
                },

                IR::Dispatch {
                    size: ir::DispatchSize::Indirect { buffer, .. },
                    ..
                } => {
                    buffer_barrier!(self.resource_root(*buffer), Access::IndirectRead);
                },

                IR::Release { resource, access, .. } => {
                    transition!(*resource, *access, self.resolve_access(*access));
                },

                _ => {},
            }

            result.push((value_id, ir));
        }

        result
    }

    fn is_buffer(&self, id: ValueId) -> bool {
        matches!(
            self.instructions.get(self.resource_root(id).0 as usize),
            Some(IR::ConstructBuffer { .. })
        )
    }

    fn resource_root(&self, id: ValueId) -> ValueId {
        let mut id = id;

        for _ in 0..256 {
            let Some(ir) = self.instructions.get(id.0 as usize) else {
                return id;
            };

            id = match ir {
                IR::Clear { attachment, .. } => *attachment,
                IR::Blit { dst, .. } => *dst,
                IR::CopyBufferToImage { image, .. } => *image,
                IR::BeginRendering {
                    color_attachments,
                    depth_attachment,
                    ..
                } => match color_attachments.first().or(depth_attachment.as_ref()) {
                    Some(first) => *first,
                    None => return id,
                },
                IR::BeginCompute { resources, .. } => match resources.first() {
                    Some((first, _)) => *first,
                    None => return id,
                },
                IR::BindPipeline { pass, .. }
                | IR::SetState { pass, .. }
                | IR::BindVertexBuffers { pass, .. }
                | IR::BindIndexBuffer { pass, .. }
                | IR::SampleImage { pass, .. }
                | IR::Draw { pass, .. }
                | IR::DrawIndexed { pass, .. }
                | IR::EndRendering { pass }
                | IR::Dispatch { pass, .. }
                | IR::EndCompute { pass } => *pass,
                IR::Acquire { resource, .. } | IR::Release { resource, .. } => *resource,
                _ => return id,
            };
        }

        id
    }

    fn infer_usage(&self, nodes: &mut [ir::Instr]) {
        let access_of = |id: &ValueId| {
            nodes
                .iter()
                .find(|(instr_id, _)| instr_id == id)
                .map(|(_, ir)| ir)
                .or_else(|| self.instructions.get(id.0 as usize))
                .and_then(|ir| match ir {
                    IR::Constant(ir::Constant::Access(access)) => Some(*access),
                    _ => None,
                })
        };

        let mut usages: HashMap<ValueId, vk::ImageUsageFlags> = HashMap::new();
        for (_, ir) in nodes.iter() {
            let IR::ImageBarrier { dst_access, value, .. } = ir else {
                continue;
            };
            let Some(access) = access_of(dst_access) else {
                continue;
            };

            *usages.entry(self.resource_root(*value)).or_default() |= vk::ImageUsageFlags::from(access);
        }

        for (value_id, ir) in nodes.iter_mut() {
            let IR::ConstructImage { image, usage, .. } = ir else {
                continue;
            };
            if !image.is_null() {
                continue;
            }

            *usage |= usages.get(value_id).copied().unwrap_or_default();
            if usage.is_empty() {
                tracing::warn!(%value_id, "transient image is never used; it will be created with no usage");
            }
        }
    }

    fn resolve_resource(&self, id: ValueId) -> Option<&IR> {
        let mut id = id;

        for _ in 0..256 {
            let ir = self.instructions.get(id.0 as usize)?;
            id = match ir {
                IR::ConstructImage { .. } | IR::ConstructBuffer { .. } => return Some(ir),
                // every element of an attachment array describes the same image
                IR::Array { elements, .. } => *elements.first()?,
                IR::Index { array, .. } => *array,
                IR::Clear { attachment, .. } => *attachment,
                IR::Blit { dst, .. } => *dst,
                IR::CopyBufferToImage { image, .. } => *image,
                IR::BeginRendering {
                    color_attachments,
                    depth_attachment,
                    ..
                } => match color_attachments.first() {
                    Some(first) => *first,
                    None => (*depth_attachment)?,
                },
                IR::BeginCompute { resources, .. } => resources.first()?.0,
                IR::BindPipeline { pass, .. }
                | IR::SetState { pass, .. }
                | IR::BindVertexBuffers { pass, .. }
                | IR::BindIndexBuffer { pass, .. }
                | IR::SampleImage { pass, .. }
                | IR::Draw { pass, .. }
                | IR::DrawIndexed { pass, .. }
                | IR::EndRendering { pass }
                | IR::Dispatch { pass, .. }
                | IR::EndCompute { pass } => *pass,
                IR::Acquire { resource, .. } | IR::Release { resource, .. } => *resource,
                _ => return None,
            };
        }

        None
    }

    fn resolve_image(&self, id: ValueId) -> Option<ImageType> {
        let IR::ConstructImage {
            format,
            samples,
            extent,
            ..
        } = self.resolve_resource(id)?
        else {
            return None;
        };

        match self.instructions.get(extent.0 as usize)? {
            IR::Constant(ir::Constant::Extent3D(extent)) => Some(ImageType {
                format: *format,
                samples: *samples,
                extent: *extent,
            }),
            _ => None,
        }
    }

    fn resolve_buffer_size(&self, id: ValueId) -> Option<u64> {
        match self.resolve_resource(id)? {
            IR::ConstructBuffer { buffer, .. } => Some(buffer.size()),
            _ => None,
        }
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
                    depth_attachment,
                    render_area,
                    ..
                } => {
                    let mut rendering = RenderingState::default();
                    let mut attachment_extent = None;
                    let mut samples_from_color = false;

                    for (index, attachment) in color_attachments.iter().enumerate() {
                        let Some(image) = self.resolve_image(*attachment) else {
                            tracing::warn!(%attachment, "cannot infer attachment type; pipelines may not match");
                            continue;
                        };

                        if index == 0 {
                            rendering.samples = image.samples;
                            samples_from_color = true;
                            attachment_extent = Some(vk::Extent2D {
                                width: image.extent.width,
                                height: image.extent.height,
                            });
                        } else if image.samples != rendering.samples {
                            tracing::warn!(
                                %attachment,
                                "attachment sample count differs from the region's first attachment"
                            );
                        }

                        rendering.color_formats.push(image.format);
                    }

                    if let Some(attachment) = depth_attachment {
                        match self.resolve_image(*attachment) {
                            Some(image) => {
                                rendering.depth_format = Some(image.format);

                                // a depth-only region has nothing else to take these from
                                if !samples_from_color {
                                    rendering.samples = image.samples;
                                    attachment_extent = Some(vk::Extent2D {
                                        width: image.extent.width,
                                        height: image.extent.height,
                                    });
                                } else if image.samples != rendering.samples {
                                    tracing::warn!(
                                        %attachment,
                                        "depth attachment sample count differs from the color attachments"
                                    );
                                }
                            },
                            None => {
                                tracing::warn!(%attachment, "cannot infer depth attachment type; pipelines may not match")
                            },
                        }
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

                IR::BeginCompute { .. } => {
                    regions.insert(
                        *value_id,
                        InForce {
                            state: PassState::default(),
                            area: vk::Rect2D::default(),
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
                }
                | IR::DrawIndexed {
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

                IR::Dispatch {
                    pass,
                    pipeline,
                    push_constants,
                    ..
                } => {
                    if let Some(in_force) = regions.get(pass).cloned() {
                        if in_force.pipeline.is_none() {
                            tracing::warn!(%value_id, "dispatch with no pipeline bound");
                        }
                        *pipeline = in_force.pipeline;
                        *push_constants = in_force.state.push_constants.clone();
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::BindVertexBuffers { pass, .. }
                | IR::BindIndexBuffer { pass, .. }
                | IR::SampleImage { pass, .. }
                | IR::EndRendering { pass }
                | IR::EndCompute { pass } => {
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
    begin: ValueId,
}

impl RenderPass<'_> {
    pub fn id(&self) -> ValueId { self.id }

    pub fn with_name(self, name: impl Into<Arc<str>>) -> Self {
        self.module.set_name(self.begin, name);
        self
    }

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

    pub fn bind_index_buffer(self, buffer: ValueId, index_type: vk::IndexType) -> Self {
        self.bind_index_buffer_at(buffer, 0, index_type)
    }

    pub fn bind_index_buffer_at(mut self, buffer: ValueId, offset: u64, index_type: vk::IndexType) -> Self {
        self.id = self.module.emit(IR::BindIndexBuffer {
            pass: self.id,
            buffer,
            offset,
            index_type,
        });
        self
    }

    pub fn sample_image(mut self, image: ValueId) -> Self {
        self.id = self.module.emit(IR::SampleImage { pass: self.id, image });
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

    pub fn set_depth(self, depth: DepthState) -> Self { self.set_state(StateChange::Depth(depth)) }

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

    pub fn draw_indexed(self, index_count: u32, instance_count: u32) -> Self {
        self.draw_indexed_range(index_count, instance_count, 0, 0, 0)
    }

    pub fn draw_indexed_range(
        mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32,
    ) -> Self {
        let index_count = self.module.lower_u32(index_count);
        let instance_count = self.module.lower_u32(instance_count);
        let first_index = self.module.lower_u32(first_index);
        let vertex_offset = self.module.lower_i32(vertex_offset);
        let first_instance = self.module.lower_u32(first_instance);
        self.id = self.module.emit(IR::DrawIndexed {
            pass: self.id,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            pipeline: None,
            state: Default::default(),
            dynamic: Default::default(),
        });
        self
    }

    pub fn end_rendering(self) -> ValueId { self.module.emit(IR::EndRendering { pass: self.id }) }
}

pub struct ComputePass<'a> {
    module: &'a mut Module,
    id: ValueId,
    begin: ValueId,
}

impl ComputePass<'_> {
    pub fn id(&self) -> ValueId { self.id }

    pub fn with_name(self, name: impl Into<Arc<str>>) -> Self {
        self.module.set_name(self.begin, name);
        self
    }

    pub fn bind_pipeline(mut self, pipeline: PipelineId) -> Self {
        self.id = self.module.emit(IR::BindPipeline {
            pass: self.id,
            pipeline,
            bind_point: vk::PipelineBindPoint::COMPUTE,
        });
        self
    }

    pub fn access(self, resource: ValueId, access: Access) -> Self {
        let Some(IR::BeginCompute { resources, .. }) = self.module.instructions.get_mut(self.begin.0 as usize) else {
            return self;
        };

        match resources.iter_mut().find(|(id, _)| *id == resource) {
            Some((_, merged)) => *merged |= access,
            None => resources.push((resource, access)),
        }

        self
    }

    pub fn read(self, resource: ValueId) -> Self { self.access(resource, Access::ComputeRead) }

    pub fn write(self, resource: ValueId) -> Self { self.access(resource, Access::ComputeWrite) }

    pub fn read_write(self, resource: ValueId) -> Self { self.access(resource, Access::ComputeRW) }

    pub fn sample_image(self, image: ValueId) -> Self { self.access(image, Access::ComputeSampled) }

    pub fn read_uniform(self, buffer: ValueId) -> Self { self.access(buffer, Access::ComputeUniformRead) }

    pub fn push_constants<T: Copy>(self, value: &T) -> Self { self.push_constants_at(0, value) }

    pub fn push_constants_at<T: Copy>(self, offset: u32, value: &T) -> Self {
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.push_constant_bytes(offset, bytes)
    }

    pub fn push_constant_bytes(mut self, offset: u32, data: &[u8]) -> Self {
        self.id = self.module.emit(IR::SetState {
            pass: self.id,
            change: StateChange::PushConstants {
                offset,
                data: data.to_vec(),
            },
        });
        self
    }

    fn dispatch_size(mut self, size: ir::DispatchSize) -> Self {
        self.id = self.module.emit(IR::Dispatch {
            pass: self.id,
            size,
            pipeline: None,
            push_constants: Default::default(),
        });
        self
    }

    pub fn dispatch(self, groups_x: u32, groups_y: u32, groups_z: u32) -> Self {
        let x = self.module.lower_u32(groups_x);
        let y = self.module.lower_u32(groups_y);
        let z = self.module.lower_u32(groups_z);
        self.dispatch_size(ir::DispatchSize::Groups { x, y, z })
    }

    /// Dispatches at least this many invocations per axis, rounded up to whole workgroups of
    /// whatever the bound pipeline declares.
    pub fn dispatch_invocations(self, invocations_x: u32, invocations_y: u32, invocations_z: u32) -> Self {
        let x = self.module.lower_u32(invocations_x);
        let y = self.module.lower_u32(invocations_y);
        let z = self.module.lower_u32(invocations_z);
        self.dispatch_size(ir::DispatchSize::Invocations { x, y, z })
    }

    /// Dispatches one invocation per pixel of `image`, width along x, height along y and depth
    /// along z.
    pub fn dispatch_invocations_per_pixel(self, image: ValueId) -> Self {
        self.dispatch_invocations_per_pixel_scaled(image, [1.0; 3])
    }

    /// Dispatches [`Self::dispatch_invocations_per_pixel`] scaled per axis: above one is more
    /// invocations than pixels, below one is fewer. Rounding up to whole workgroups happens after
    /// the scaling.
    pub fn dispatch_invocations_per_pixel_scaled(self, image: ValueId, scale: [f32; 3]) -> Self {
        let Some(extent) = self.module.resolve_image(image).map(|image| image.extent) else {
            tracing::warn!(%image, "cannot infer the extent to dispatch over; the dispatch is dropped");
            return self;
        };

        self.dispatch_invocations(
            scaled(extent.width as u64, scale[0]),
            scaled(extent.height as u64, scale[1]),
            scaled(extent.depth as u64, scale[2]),
        )
    }

    /// Dispatches one invocation per `element_size` bytes of `buffer`, along the x-axis only.
    pub fn dispatch_invocations_per_element(self, buffer: ValueId, element_size: u64) -> Self {
        self.dispatch_invocations_per_element_scaled(buffer, element_size, 1.0)
    }

    /// Dispatches [`Self::dispatch_invocations_per_element`] scaled: above one is more invocations
    /// than elements, below one is fewer.
    pub fn dispatch_invocations_per_element_scaled(self, buffer: ValueId, element_size: u64, scale: f32) -> Self {
        if element_size == 0 {
            tracing::warn!(%buffer, "elements of no size have no count to dispatch over; the dispatch is dropped");
            return self;
        }

        let Some(size) = self.module.resolve_buffer_size(buffer) else {
            tracing::warn!(%buffer, "cannot infer the element count to dispatch over; the dispatch is dropped");
            return self;
        };

        self.dispatch_invocations(scaled(size / element_size, scale), 1, 1)
    }

    /// Dispatches the group counts the device reads out of `buffer` itself.
    pub fn dispatch_indirect(self, buffer: ValueId) -> Self { self.dispatch_indirect_at(buffer, 0) }

    pub fn dispatch_indirect_at(self, buffer: ValueId, offset: u64) -> Self {
        self.dispatch_size(ir::DispatchSize::Indirect { buffer, offset })
    }

    pub fn end_compute(self) -> ValueId { self.module.emit(IR::EndCompute { pass: self.id }) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlendPreset, Image, PipelineState, ResolvedViewport};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
    const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;

    fn module_with_attachment() -> (Module, ValueId) {
        let mut module = Module::default();
        let attachment = ImageAttachment::new(
            Image::imported(
                vk::Image::null(),
                FORMAT,
                vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
                vk::SampleCountFlags::TYPE_1,
            ),
            FORMAT,
            vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            vk::SampleCountFlags::TYPE_1,
            vk::ImageLayout::UNDEFINED,
        );
        let (_, construct) = module.lower_image_attachment(&attachment, Some("target".into()));
        (module, construct)
    }

    fn depth_info() -> ImageInfo { ImageInfo::depth_target(extent_2d(), DEPTH_FORMAT) }

    fn access_constant(module: &Module, instructions: &[ir::Instr], id: &ValueId) -> Access {
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
    }

    fn memory_barriers(module: &Module, instructions: &[ir::Instr]) -> Vec<(Access, Access)> {
        instructions
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::MemoryBarrier { src_access, dst_access } => Some((
                    access_constant(module, instructions, src_access),
                    access_constant(module, instructions, dst_access),
                )),
                _ => None,
            })
            .collect()
    }

    struct ImageBarrier {
        src: Access,
        dst: Access,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        resource: ValueId,
    }

    fn image_barriers(module: &Module, instructions: &[ir::Instr]) -> Vec<ImageBarrier> {
        instructions
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::ImageBarrier {
                    src_access,
                    dst_access,
                    old_layout,
                    new_layout,
                    value,
                } => Some(ImageBarrier {
                    src: access_constant(module, instructions, src_access),
                    dst: access_constant(module, instructions, dst_access),
                    old_layout: *old_layout,
                    new_layout: *new_layout,
                    resource: module.resource_root(*value),
                }),
                _ => None,
            })
            .collect()
    }

    fn extent_2d() -> vk::Extent2D {
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        }
    }

    fn transient_info() -> ImageInfo { ImageInfo::color_target(extent_2d(), FORMAT) }

    /// An image that declares no usage at all, so what it ends up created with is entirely
    /// what the graph inferred.
    fn untyped_info() -> ImageInfo {
        ImageInfo {
            extent: vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            format: FORMAT,
            ..Default::default()
        }
    }

    fn image_usage(instructions: &[ir::Instr], id: ValueId) -> vk::ImageUsageFlags {
        instructions
            .iter()
            .find_map(|(instr_id, ir)| match ir {
                IR::ConstructImage { usage, .. } if *instr_id == id => Some(*usage),
                _ => None,
            })
            .expect("the image should still be declared after compiling")
    }

    fn draws(instructions: &[ir::Instr]) -> Vec<(Option<PipelineId>, PipelineState)> {
        instructions
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { pipeline, state, .. } | IR::DrawIndexed { pipeline, state, .. } => {
                    Some((*pipeline, state.clone()))
                },
                _ => None,
            })
            .collect()
    }

    fn u32_constant(module: &Module, id: &ValueId) -> u32 {
        match module.instructions.get(id.0 as usize) {
            Some(IR::Constant(ir::Constant::U32(v))) => *v,
            _ => panic!("{id} is not a u32 constant"),
        }
    }

    fn i32_constant(module: &Module, id: &ValueId) -> i32 {
        match module.instructions.get(id.0 as usize) {
            Some(IR::Constant(ir::Constant::I32(v))) => *v,
            _ => panic!("{id} is not an i32 constant"),
        }
    }

    #[test]
    fn a_transient_image_is_created_with_the_usage_the_graph_implies() {
        let (mut module, swapchain) = module_with_attachment();
        let target = module.transient_image(&transient_info());

        let rendered = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();
        let swapchain = module.blit(rendered, swapchain);
        let end = module.present(swapchain);

        assert_eq!(
            image_usage(&module.compile(end), target),
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC
        );
    }

    #[test]
    fn a_blit_puts_each_side_in_its_own_transfer_layout() {
        let (mut module, destination) = module_with_attachment();
        let source = module.transient_image(&transient_info());
        let end = module.blit(source, destination);

        let compiled = module.compile(end);
        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.len(), 2);

        assert_eq!(barriers[0].resource, source);
        assert_eq!(barriers[0].src, Access::None);
        assert_eq!(barriers[0].dst, Access::BlitRead);
        assert_eq!(barriers[0].old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barriers[0].new_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        assert_eq!(barriers[1].resource, destination);
        assert_eq!(barriers[1].dst, Access::BlitWrite);
        assert_eq!(barriers[1].new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    #[test]
    fn an_imported_image_starts_in_the_layout_its_owner_left_it_in() {
        let mut module = Module::default();
        let image = Image::imported(
            vk::Image::null(),
            FORMAT,
            vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            vk::SampleCountFlags::TYPE_1,
        );

        let imported = module.import_image(&image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let end = module.clear(imported, crate::clear::f32::BLACK);

        let compiled = module.compile(end);
        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].old_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(barriers[0].new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    #[test]
    fn a_second_read_at_the_same_layout_does_not_repeat_the_barrier() {
        let (mut module, first) = module_with_attachment();
        let source = module.transient_image(&transient_info());

        // the second blit writes what the first one produced, so both are in the graph and the
        // source is read twice without ever leaving TRANSFER_SRC_OPTIMAL
        let first = module.blit(source, first);
        let end = module.blit(source, first);

        let compiled = module.compile(end);
        let barriers = image_barriers(&module, &compiled);
        let reads = barriers.iter().filter(|b| b.resource == source).count();
        assert_eq!(reads, 1);

        // the destination is written twice, and a write always waits
        assert_eq!(barriers.len(), 3);
    }

    #[test]
    fn a_release_to_the_layout_an_image_already_rests_in_costs_nothing() {
        let (mut module, destination) = module_with_attachment();
        let source = module.transient_image(&transient_info());
        let blit = module.blit(source, destination);

        let present = module.present(blit);
        let resting = module.release(source, Access::BlitRead, DomainFlag::Graphics);
        let compiled = module.compile_all(&[present, resting]);

        // nothing consumes the release, so only naming it as a root keeps it
        let position = |id: ValueId| compiled.iter().position(|(instr_id, _)| *instr_id == id);
        assert!(position(resting) > position(blit));
        assert!(module.compile(present).iter().all(|(id, _)| *id != resting));

        // and it asks for the layout the blit already left it in, so it emits no barrier
        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.iter().filter(|b| b.resource == source).count(), 1);
    }

    #[test]
    fn a_release_that_does_change_the_layout_still_emits_its_barrier() {
        let (mut module, destination) = module_with_attachment();
        let source = module.transient_image(&transient_info());
        let blit = module.blit(source, destination);

        let present = module.present(blit);
        let resting = module.release(source, Access::ColorRW, DomainFlag::Graphics);
        let compiled = module.compile_all(&[present, resting]);

        let barriers = image_barriers(&module, &compiled);
        let last = barriers
            .iter()
            .rfind(|b| b.resource == source)
            .expect("the source should have been transitioned");
        assert_eq!(last.old_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(last.new_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
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
        assert_eq!(
            memory_barriers(&module, &instructions),
            vec![(Access::HostWrite, Access::AttributeRead)]
        );

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

    /// A depth attachment reaches the pipeline through the region's rendering state, which is
    /// what makes the created pipeline agree with the attachment it renders into.
    #[test]
    fn a_region_opened_with_depth_infers_the_depth_format() {
        let (mut module, attachment) = module_with_attachment();
        let depth = module.transient_image(&depth_info());

        let end = module
            .begin_rendering_depth(&[attachment], depth)
            .bind_graphics_pipeline(PipelineId(0))
            .set_depth(DepthState::less())
            .draw(3, 1)
            .end_rendering();
        let instructions = module.compile(end);

        let draws = draws(&instructions);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
        assert_eq!(draws[0].1.rendering.depth_format, Some(DEPTH_FORMAT));
        assert_eq!(draws[0].1.depth, DepthState::less());
    }

    /// Depth state a region without a depth attachment cannot honour is dropped, so the draw
    /// does not ask for a pipeline that would fail validation.
    #[test]
    fn depth_state_without_a_depth_attachment_is_dropped() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_depth(DepthState::less())
            .draw(3, 1)
            .end_rendering();
        let instructions = module.compile(end);

        let draws = draws(&instructions);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.rendering.depth_format, None);
        assert_eq!(draws[0].1.depth, DepthState::default());
    }

    /// The region writes the depth attachment, so the graph has to move it into the layout an
    /// attachment write wants before the region opens.
    #[test]
    fn the_depth_attachment_is_transitioned_for_the_region() {
        let (mut module, attachment) = module_with_attachment();
        let depth = module.transient_image(&depth_info());

        let end = module
            .begin_rendering_depth(&[attachment], depth)
            .bind_graphics_pipeline(PipelineId(0))
            .set_depth(DepthState::less())
            .draw(3, 1)
            .end_rendering();
        let instructions = module.compile(end);

        let barrier = image_barriers(&module, &instructions)
            .into_iter()
            .find(|barrier| barrier.resource == module.resource_root(depth))
            .expect("the depth attachment should be transitioned");

        assert_eq!(barrier.dst, Access::DepthStencilRW);
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    /// Nothing declares the usage of a transient depth target, so rendering into it is what has
    /// to put the attachment bit on.
    #[test]
    fn rendering_into_a_transient_depth_target_infers_its_usage() {
        let (mut module, attachment) = module_with_attachment();
        let depth = module.transient_image(&ImageInfo {
            extent: vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            format: DEPTH_FORMAT,
            ..Default::default()
        });

        let end = module
            .begin_rendering_depth(&[attachment], depth)
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();
        let instructions = module.compile(end);

        assert!(
            image_usage(&instructions, depth).contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            "the depth attachment should be created as one"
        );
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
    fn binding_an_index_buffer_makes_the_host_write_visible_to_index_input() {
        let (mut module, attachment) = module_with_attachment();
        let indices = module.import_buffer(&Buffer::default());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .end_rendering();

        let instructions = module.compile(end);
        assert_eq!(
            memory_barriers(&module, &instructions),
            vec![(Access::HostWrite, Access::IndexRead)]
        );

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
    }

    #[test]
    fn vertex_and_index_buffers_each_get_the_barrier_their_stage_needs() {
        let (mut module, attachment) = module_with_attachment();
        let vertices = module.import_buffer(&Buffer::default());
        let indices = module.import_buffer(&Buffer::default());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, vertices)
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .end_rendering();

        let barriers = memory_barriers(&module, &module.compile(end));
        assert_eq!(barriers.len(), 2);
        assert!(barriers.contains(&(Access::HostWrite, Access::AttributeRead)));
        assert!(barriers.contains(&(Access::HostWrite, Access::IndexRead)));
    }

    #[test]
    fn rebinding_the_same_index_buffer_does_not_repeat_the_barrier() {
        let (mut module, attachment) = module_with_attachment();
        let indices = module.import_buffer(&Buffer::default());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .end_rendering();

        assert_eq!(memory_barriers(&module, &module.compile(end)).len(), 1);
    }

    /// An indexed draw goes through the same state inference as a plain one; a variant missing
    /// from that pass would silently come out with no pipeline and default state.
    #[test]
    fn an_indexed_draw_inherits_the_pipeline_and_state_in_force() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);
        let indices = module.import_buffer(&Buffer::default());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .set_primitive_topology(vk::PrimitiveTopology::LINE_STRIP)
            .broadcast_color_blend(BlendPreset::PremultipliedAlphaBlend)
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(6, 1)
            .end_rendering();

        let draws = draws(&module.compile(end));
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].0, Some(pipeline));
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
        assert_eq!(draws[0].1.topology, vk::PrimitiveTopology::LINE_STRIP);
        assert!(draws[0].1.blend.iter().all(|blend| blend.blend_enable));
    }

    #[test]
    fn an_indexed_draw_lowers_its_range_with_a_signed_vertex_offset() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .draw_indexed_range(9, 2, 3, -4, 5)
            .end_rendering();

        let instructions = module.compile(end);
        let (_, ir) = instructions
            .iter()
            .find(|(_, ir)| matches!(ir, IR::DrawIndexed { .. }))
            .expect("the indexed draw should survive compilation");

        let IR::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            ..
        } = ir
        else {
            unreachable!()
        };

        assert_eq!(u32_constant(&module, index_count), 9);
        assert_eq!(u32_constant(&module, instance_count), 2);
        assert_eq!(u32_constant(&module, first_index), 3);
        assert_eq!(i32_constant(&module, vertex_offset), -4);
        assert_eq!(u32_constant(&module, first_instance), 5);
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
    fn a_dispatch_picks_up_the_compute_pipeline_and_the_pushes_in_force() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(1))
            .write(attachment)
            .push_constants(&7u32)
            .dispatch(8, 4, 1)
            .end_compute();

        let compiled = module.compile(end);
        let (
            _,
            IR::Dispatch {
                pipeline,
                push_constants,
                size: ir::DispatchSize::Groups { x, y, z },
                ..
            },
        ) = compiled
            .iter()
            .find(|(_, ir)| matches!(ir, IR::Dispatch { .. }))
            .expect("dispatch should be present")
        else {
            unreachable!()
        };

        assert_eq!(*pipeline, Some(PipelineId(1)));
        assert_eq!(push_constants.data, 7u32.to_ne_bytes());
        assert_eq!(
            [
                u32_constant(&module, x),
                u32_constant(&module, y),
                u32_constant(&module, z)
            ],
            [8, 4, 1]
        );
    }

    fn dispatch_sizes(instructions: &[ir::Instr]) -> Vec<ir::DispatchSize> {
        instructions
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Dispatch { size, .. } => Some(*size),
                _ => None,
            })
            .collect()
    }

    fn invocations(module: &Module, instructions: &[ir::Instr]) -> [u32; 3] {
        match dispatch_sizes(instructions).as_slice() {
            [ir::DispatchSize::Invocations { x, y, z }] => [
                u32_constant(module, x),
                u32_constant(module, y),
                u32_constant(module, z),
            ],
            sizes => panic!("expected one dispatch sized in invocations, got {sizes:?}"),
        }
    }

    /// The workgroup only enters the picture once a pipeline is bound, which is why the count
    /// reaches the compiled module as the invocations the caller asked for.
    #[test]
    fn a_dispatch_sized_in_invocations_keeps_the_count_it_was_given() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(attachment)
            .dispatch_invocations(1920, 1080, 1)
            .end_compute();

        let compiled = module.compile(end);
        assert_eq!(invocations(&module, &compiled), [1920, 1080, 1]);
    }

    #[test]
    fn a_dispatch_per_pixel_covers_the_image_it_was_sized_from() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(attachment)
            .dispatch_invocations_per_pixel(attachment)
            .end_compute();

        let compiled = module.compile(end);
        assert_eq!(invocations(&module, &compiled), [WIDTH, HEIGHT, 1]);
    }

    /// Scaling rounds up, so an axis that scales to a fraction of an invocation still gets one.
    #[test]
    fn a_scaled_dispatch_per_pixel_rounds_every_axis_up() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(attachment)
            .dispatch_invocations_per_pixel_scaled(attachment, [0.5, 2.0, 0.25])
            .end_compute();

        let compiled = module.compile(end);
        assert_eq!(invocations(&module, &compiled), [WIDTH / 2, HEIGHT * 2, 1]);
    }

    #[test]
    fn a_dispatch_per_element_sizes_itself_on_the_x_axis_alone() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::new(vk::Buffer::null(), 800, 0, None));

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(buffer)
            .dispatch_invocations_per_element(buffer, 20)
            .end_compute();

        let compiled = module.compile(end);
        assert_eq!(invocations(&module, &compiled), [40, 1, 1]);
    }

    #[test]
    fn a_scaled_dispatch_per_element_scales_the_element_count() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::new(vk::Buffer::null(), 800, 0, None));

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(buffer)
            .dispatch_invocations_per_element_scaled(buffer, 20, 3.0)
            .end_compute();

        let compiled = module.compile(end);
        assert_eq!(invocations(&module, &compiled), [120, 1, 1]);
    }

    /// A buffer of counts is no use to the device until the write that filled it is visible, and
    /// nothing but the dispatch itself says that it will be read as commands.
    #[test]
    fn an_indirect_dispatch_waits_for_the_buffer_of_counts() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());
        let commands = module.import_buffer(&Buffer::new(vk::Buffer::null(), 12, 0, None));

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(image)
            .dispatch_indirect(commands)
            .end_compute();

        let compiled = module.compile(end);
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::IndirectRead)]
        );
        assert_eq!(
            dispatch_sizes(&compiled),
            vec![ir::DispatchSize::Indirect {
                buffer: commands,
                offset: 0
            }]
        );
    }

    /// The counts are filled once, so the second dispatch reading the same buffer waits on
    /// nothing further.
    #[test]
    fn indirect_dispatches_out_of_one_buffer_wait_on_it_once() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());
        let commands = module.import_buffer(&Buffer::new(vk::Buffer::null(), 24, 0, None));

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(image)
            .dispatch_indirect(commands)
            .dispatch_indirect_at(commands, 12)
            .end_compute();

        let compiled = module.compile(end);
        assert_eq!(
            dispatch_sizes(&compiled),
            vec![
                ir::DispatchSize::Indirect {
                    buffer: commands,
                    offset: 0
                },
                ir::DispatchSize::Indirect {
                    buffer: commands,
                    offset: 12
                },
            ]
        );
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::IndirectRead)]
        );
    }

    #[test]
    fn a_dispatch_that_writes_an_image_puts_it_in_general_with_storage_usage() {
        let (mut module, swapchain) = module_with_attachment();
        let target = module.transient_image(&untyped_info());

        let computed = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(target)
            .dispatch(8, 8, 1)
            .end_compute();
        // the region's value stands in for what it wrote, so the blit reads the dispatch
        let swapchain = module.blit(computed, swapchain);
        let end = module.present(swapchain);

        let compiled = module.compile(end);
        let written = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == target)
            .expect("the written image should have been transitioned");

        assert_eq!(written.dst, Access::ComputeWrite);
        assert_eq!(written.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(written.new_layout, vk::ImageLayout::GENERAL);
        assert_eq!(
            image_usage(&compiled, target),
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC
        );
    }

    #[test]
    fn a_dispatch_that_reads_what_an_earlier_one_wrote_waits_for_it() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());

        let written = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .write(image)
            .dispatch(1, 1, 1)
            .end_compute();
        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(1))
            .read(written)
            .dispatch(1, 1, 1)
            .end_compute();

        let barriers = image_barriers(&module, &module.compile(end));
        assert_eq!(barriers.len(), 2);
        assert_eq!(barriers[1].src, Access::ComputeWrite);
        assert_eq!(barriers[1].dst, Access::ComputeRead);
        assert_eq!(barriers[1].old_layout, vk::ImageLayout::GENERAL);
        assert_eq!(barriers[1].new_layout, vk::ImageLayout::GENERAL);
    }

    #[test]
    fn a_resource_declared_twice_carries_both_accesses() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());

        let end = module
            .begin_compute()
            .bind_pipeline(PipelineId(0))
            .read(image)
            .write(image)
            .dispatch(1, 1, 1)
            .end_compute();

        let barriers = image_barriers(&module, &module.compile(end));
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].dst, Access::ComputeRW);
    }

    #[test]
    fn a_buffer_a_dispatch_wrote_is_made_visible_to_the_draw_that_reads_it() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default());

        let computed = module
            .begin_compute()
            .bind_pipeline(PipelineId(1))
            .write(buffer)
            .dispatch(1, 1, 1)
            .end_compute();
        let rendered = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile_all(&[computed, rendered]);
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![
                (Access::HostWrite, Access::ComputeWrite),
                (Access::ComputeWrite, Access::AttributeRead)
            ]
        );
    }

    /// A dispatch cannot be recorded inside a rendering region, so a buffer the draw binds has
    /// to bring its producer out ahead of the region rather than into it, even when the draw is
    /// the only root and the dispatch is reachable through the bound buffer alone.
    #[test]
    fn a_dispatch_the_draw_binds_the_result_of_stays_outside_the_region() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default());

        let computed = module
            .begin_compute()
            .bind_pipeline(PipelineId(1))
            .write(buffer)
            .dispatch(1, 1, 1)
            .end_compute();
        let rendered = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, computed)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(rendered);
        let position = |predicate: fn(&IR) -> bool| {
            compiled
                .iter()
                .position(|(_, ir)| predicate(ir))
                .expect("instruction should be present")
        };

        let end_compute = position(|ir| matches!(ir, IR::EndCompute { .. }));
        let begin_rendering = position(|ir| matches!(ir, IR::BeginRendering { .. }));
        assert!(
            end_compute < begin_rendering,
            "the compute region closed at {end_compute} but the rendering region opened at {begin_rendering}"
        );

        // and the barrier that hands the buffer over is hoisted out with it
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![
                (Access::HostWrite, Access::ComputeWrite),
                (Access::ComputeWrite, Access::AttributeRead)
            ]
        );
    }

    #[test]
    fn a_copy_waits_for_the_host_write_and_puts_the_image_in_the_transfer_layout() {
        let mut module = Module::default();
        let staging = module.import_buffer(&Buffer::default());
        let texture = module.transient_image(&untyped_info());
        let end = module.copy_buffer_to_image(staging, texture);

        let compiled = module.compile(end);
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::CopyRead)]
        );

        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].resource, texture);
        assert_eq!(barriers[0].dst, Access::CopyWrite);
        assert_eq!(barriers[0].old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barriers[0].new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    /// The whole point of naming a sampled image: a layout transition cannot be recorded
    /// inside a rendering region, so it has to be hoisted out the way an attachment's is.
    #[test]
    fn a_sampled_image_reaches_the_shader_read_layout_before_the_region_opens() {
        let (mut module, attachment) = module_with_attachment();
        let texture = module.transient_image(&untyped_info());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .sample_image(texture)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(end);
        let sampled = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == texture)
            .expect("the sampled image should have been transitioned");
        assert_eq!(sampled.dst, Access::FragmentSampled);
        assert_eq!(sampled.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let last_barrier = compiled
            .iter()
            .rposition(|(_, ir)| matches!(ir, IR::ImageBarrier { .. }))
            .expect("barriers should be present");
        let begin = compiled
            .iter()
            .position(|(_, ir)| matches!(ir, IR::BeginRendering { .. }))
            .expect("the region should be present");
        assert!(last_barrier < begin);
    }

    #[test]
    fn sampling_the_same_image_twice_in_a_region_does_not_repeat_the_barrier() {
        let (mut module, attachment) = module_with_attachment();
        let texture = module.transient_image(&untyped_info());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .sample_image(texture)
            .draw(3, 1)
            .sample_image(texture)
            .draw(3, 1)
            .end_rendering();

        let barriers = image_barriers(&module, &module.compile(end));
        assert_eq!(barriers.iter().filter(|b| b.resource == texture).count(), 1);
    }

    #[test]
    fn an_image_uploaded_and_then_sampled_moves_through_both_layouts_in_order() {
        let (mut module, attachment) = module_with_attachment();
        let staging = module.import_buffer(&Buffer::default());
        let texture = module.transient_image(&untyped_info());
        let uploaded = module.copy_buffer_to_image(staging, texture);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .sample_image(uploaded)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(end);
        let layouts = image_barriers(&module, &compiled)
            .iter()
            .filter(|barrier| barrier.resource == texture)
            .map(|barrier| (barrier.old_layout, barrier.new_layout))
            .collect::<Vec<_>>();

        assert_eq!(
            layouts,
            vec![
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                ),
            ]
        );
    }

    #[test]
    fn an_uploaded_and_sampled_image_is_created_with_both_usages() {
        let (mut module, attachment) = module_with_attachment();
        let staging = module.import_buffer(&Buffer::default());
        let texture = module.transient_image(&untyped_info());
        let uploaded = module.copy_buffer_to_image(staging, texture);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .sample_image(uploaded)
            .draw(3, 1)
            .end_rendering();

        assert_eq!(
            image_usage(&module.compile(end), texture),
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED
        );
    }

    /// A copy takes the extent off the image it is filling, so a region that names its own
    /// has to be the one that survives.
    #[test]
    fn a_copy_region_overrides_the_extent_the_whole_image_would_have_used() {
        let mut module = Module::default();
        let staging = module.import_buffer(&Buffer::default());
        let texture = module.transient_image(&untyped_info());

        let whole = module.copy_buffer_to_image(staging, texture);
        let patch = module.copy_buffer_to_image_region(
            staging,
            whole,
            BufferImageCopy::region(vk::Offset2D { x: 4, y: 8 }, vk::Extent2D { width: 2, height: 3 }),
        );

        let regions = module
            .compile(patch)
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::CopyBufferToImage { region, .. } => Some((region.image_offset, region.image_extent)),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            regions,
            vec![
                (
                    vk::Offset3D::default(),
                    vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1)
                ),
                (
                    vk::Offset3D { x: 4, y: 8, z: 0 },
                    vk::Extent3D::default().width(2).height(3).depth(1)
                ),
            ]
        );
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
