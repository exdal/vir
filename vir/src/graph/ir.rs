use core::fmt;
use std::{collections::HashMap, sync::Arc};

use ash::vk;
use bitflags::Flags;

use crate::{
    Access,
    BlendPreset,
    Buffer,
    BufferImageCopy,
    ClearValue,
    ColorBlendAttachmentState,
    DomainFlag,
    DynamicValues,
    Image,
    PipelineId,
    PipelineState,
    RasterizationState,
    Rect2D,
    ResolvedViewport,
    StateChange,
    ValueId,
    Viewport,
};

pub type Instr = (ValueId, IR);

pub type Name = Option<Arc<str>>;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Type {
    Image {
        format: vk::Format,
        samples: vk::SampleCountFlags,
    },
    Buffer,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Constant {
    I32(i32),
    U32(u32),
    Extent2D(vk::Extent2D),
    Extent3D(vk::Extent3D),
    Access(Access),
    ClearValue(ClearValue),
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum IR {
    Type(Type),
    Constant(Constant),
    Array {
        ty: ValueId,
        elements: Vec<ValueId>,
    },
    Index {
        array: ValueId,
        index: ValueId,
    },

    ConstructBuffer {
        buffer: Buffer,
        size: ValueId,
        name: Name,
    },
    ConstructImage {
        image: Image,
        image_view: vk::ImageView,
        view_type: vk::ImageViewType,
        extent: ValueId,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        base_level: ValueId,
        level_count: ValueId,
        base_layer: ValueId,
        layer_count: ValueId,
        usage: vk::ImageUsageFlags,
        initial_layout: vk::ImageLayout,
        name: Name,
    },

    AcquireNextImage {
        swapchain: vk::SwapchainKHR,
        attachments: ValueId,
        present_semaphores: Vec<vk::Semaphore>,
    },
    Acquire {
        resource: ValueId,
        access: ValueId,
    },
    Release {
        resource: ValueId,
        access: ValueId,
        dst_domain: DomainFlag,
    },

    CallOpaque {
        args: ValueId,
        returns: ValueId,
        domain: DomainFlag,
    },
    Clear {
        attachment: ValueId,
        color: ValueId,
    },
    Blit {
        src: ValueId,
        dst: ValueId,
        filter: vk::Filter,
    },
    CopyBufferToImage {
        buffer: ValueId,
        image: ValueId,
        region: BufferImageCopy,
    },

    BeginRendering {
        color_attachments: Vec<ValueId>,
        render_area: Option<ValueId>,
        name: Name,
    },
    BindPipeline {
        pass: ValueId,
        pipeline: PipelineId,
        bind_point: vk::PipelineBindPoint,
    },
    SetState {
        pass: ValueId,
        change: StateChange,
    },
    BindVertexBuffers {
        pass: ValueId,
        first_binding: u32,
        buffers: Vec<ValueId>,
        offsets: Vec<u64>,
    },
    BindIndexBuffer {
        pass: ValueId,
        buffer: ValueId,
        offset: u64,
        index_type: vk::IndexType,
    },
    SampleImage {
        pass: ValueId,
        image: ValueId,
    },
    Draw {
        pass: ValueId,
        vertex_count: ValueId,
        instance_count: ValueId,
        first_vertex: ValueId,
        first_instance: ValueId,
        pipeline: Option<PipelineId>,
        state: PipelineState,
        dynamic: DynamicValues,
    },
    DrawIndexed {
        pass: ValueId,
        index_count: ValueId,
        instance_count: ValueId,
        first_index: ValueId,
        vertex_offset: ValueId,
        first_instance: ValueId,
        pipeline: Option<PipelineId>,
        state: PipelineState,
        dynamic: DynamicValues,
    },
    EndRendering {
        pass: ValueId,
    },
    MemoryBarrier {
        src_access: ValueId,
        dst_access: ValueId,
    },
    ImageBarrier {
        src_access: ValueId,
        dst_access: ValueId,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        value: ValueId,
    },
}

#[derive(Default)]
pub struct Program<'a> {
    values: HashMap<ValueId, &'a IR>,
}

impl<'a> Program<'a> {
    pub fn new(instructions: &'a [Instr]) -> Self {
        Self {
            values: instructions.iter().map(|(id, ir)| (*id, ir)).collect(),
        }
    }

    fn get(&self, id: ValueId) -> Option<&'a IR> { self.values.get(&id).copied() }

    fn constant(&self, id: ValueId) -> Option<&'a Constant> {
        match self.get(id)? {
            IR::Constant(constant) => Some(constant),
            _ => None,
        }
    }

    fn is_u32(&self, id: ValueId, value: u32) -> bool {
        matches!(self.constant(id), Some(Constant::U32(constant)) if *constant == value)
    }

    fn resource(&self, id: ValueId) -> Option<&'a IR> { self.resolve(id).map(|(ir, _)| ir) }

    fn resolve(&self, id: ValueId) -> Option<(&'a IR, bool)> {
        let mut id = id;
        let mut through_array = false;

        const MAX_RESOLVE_DEPTH: usize = 256;

        for _ in 0..MAX_RESOLVE_DEPTH {
            let ir = self.get(id)?;
            id = match ir {
                IR::ConstructImage { .. } | IR::ConstructBuffer { .. } => return Some((ir, through_array)),
                IR::Array { elements, .. } => {
                    through_array = true;
                    *elements.first()?
                },
                IR::Index { array, .. } => *array,
                IR::Clear { attachment, .. } => *attachment,
                IR::Blit { dst, .. } => *dst,
                IR::CopyBufferToImage { image, .. } => *image,
                IR::BeginRendering { color_attachments, .. } => *color_attachments.first()?,
                IR::BindPipeline { pass, .. }
                | IR::SetState { pass, .. }
                | IR::BindVertexBuffers { pass, .. }
                | IR::BindIndexBuffer { pass, .. }
                | IR::SampleImage { pass, .. }
                | IR::Draw { pass, .. }
                | IR::DrawIndexed { pass, .. }
                | IR::EndRendering { pass } => *pass,
                IR::Acquire { resource, .. } | IR::Release { resource, .. } => *resource,
                _ => return None,
            };
        }

        None
    }

    pub fn name(&self, id: ValueId) -> Option<&str> {
        let (resource, through_array) = self.resolve(id)?;
        let name = match resource {
            IR::ConstructImage { name, .. } | IR::ConstructBuffer { name, .. } => name.as_deref()?,
            _ => return None,
        };

        match through_array {
            true => name.split('#').next(),
            false => Some(name),
        }
    }

    pub fn extent(&self, id: ValueId) -> Option<vk::Extent3D> {
        let IR::ConstructImage { extent, .. } = self.resource(id)? else {
            return None;
        };

        match self.constant(*extent)? {
            Constant::Extent3D(extent) => Some(*extent),
            _ => None,
        }
    }

    fn operand(&self, id: ValueId) -> Operand<'_> { Operand { program: self, id } }
}

pub struct Operand<'a> {
    program: &'a Program<'a>,
    id: ValueId,
}

impl fmt::Display for Operand<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(constant) = self.program.constant(self.id) {
            return write!(f, "{constant}");
        }

        write!(f, "{}", self.id)?;
        match self.program.name(self.id) {
            Some(name) => write!(f, "({name})"),
            None => Ok(()),
        }
    }
}

pub struct Printer<'a> {
    ir: &'a IR,
    program: &'a Program<'a>,
}

impl fmt::Display for Printer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.ir.fmt_with(f, self.program) }
}

fn fmt_flags<F: Flags>(value: F) -> String {
    let names = value.iter_names().map(|(name, _)| name).collect::<Vec<_>>();
    match names.is_empty() {
        true => "none".into(),
        false => names.join("|"),
    }
}

fn fmt_usage(usage: vk::ImageUsageFlags) -> String {
    let flags = [
        (vk::ImageUsageFlags::TRANSFER_SRC, "TRANSFER_SRC"),
        (vk::ImageUsageFlags::TRANSFER_DST, "TRANSFER_DST"),
        (vk::ImageUsageFlags::SAMPLED, "SAMPLED"),
        (vk::ImageUsageFlags::STORAGE, "STORAGE"),
        (vk::ImageUsageFlags::COLOR_ATTACHMENT, "COLOR_ATTACHMENT"),
        (vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, "DEPTH_STENCIL"),
        (vk::ImageUsageFlags::INPUT_ATTACHMENT, "INPUT_ATTACHMENT"),
    ];
    let names = flags
        .iter()
        .filter(|(flag, _)| usage.contains(*flag))
        .map(|(_, name)| *name)
        .collect::<Vec<_>>();

    match names.is_empty() {
        true => "none".into(),
        false => names.join("|"),
    }
}

fn fmt_layout(layout: vk::ImageLayout) -> String { format!("{layout:?}").trim_end_matches("_OPTIMAL").to_string() }

fn fmt_samples(samples: vk::SampleCountFlags) -> String { samples.as_raw().to_string() }

fn fmt_name(name: &Name) -> String {
    match name {
        Some(name) => format!(" \"{name}\""),
        None => String::new(),
    }
}

fn fmt_rect(rect: &Rect2D) -> String {
    match rect {
        Rect2D::Framebuffer => "framebuffer".into(),
        Rect2D::Absolute(rect) => format!(
            "{},{} {}x{}",
            rect.offset.x, rect.offset.y, rect.extent.width, rect.extent.height
        ),
        Rect2D::Relative { x, y, width, height } => format!("{x},{y} {width}x{height} of framebuffer"),
    }
}

fn fmt_vk_rect(rect: &vk::Rect2D) -> String {
    format!(
        "{},{} {}x{}",
        rect.offset.x, rect.offset.y, rect.extent.width, rect.extent.height
    )
}

fn fmt_viewport(viewport: &Viewport) -> String {
    let rect = fmt_rect(&viewport.rect);
    match (viewport.min_depth, viewport.max_depth) {
        (0.0, 1.0) => rect,
        (min, max) => format!("{rect} depth={min}..{max}"),
    }
}

fn fmt_resolved_viewport(viewport: &ResolvedViewport) -> String {
    format!("{},{} {}x{}", viewport.x, viewport.y, viewport.width, viewport.height)
}

fn fmt_cull(cull: vk::CullModeFlags) -> String {
    match cull.is_empty() {
        true => "NONE".into(),
        false => format!("{cull:?}"),
    }
}

fn fmt_rasterization(state: &RasterizationState) -> String {
    let mut text = format!(
        "{:?} cull={} front={:?}",
        state.polygon_mode,
        fmt_cull(state.cull_mode),
        state.front_face
    );
    if state.line_width != 1.0 {
        text += &format!(" line_width={}", state.line_width);
    }
    text
}

fn fmt_blend(state: &ColorBlendAttachmentState) -> String {
    let presets = [
        BlendPreset::Off,
        BlendPreset::AlphaBlend,
        BlendPreset::PremultipliedAlphaBlend,
        BlendPreset::Additive,
    ];
    if let Some(preset) = presets
        .into_iter()
        .find(|preset| *state == ColorBlendAttachmentState::from(*preset))
    {
        return format!("{preset:?}");
    }

    format!(
        "color={:?}*{:?} {:?} alpha={:?}*{:?} {:?} mask={:?}",
        state.src_color_blend_factor,
        state.dst_color_blend_factor,
        state.color_blend_op,
        state.src_alpha_blend_factor,
        state.dst_alpha_blend_factor,
        state.alpha_blend_op,
        state.color_write_mask
    )
}

fn fmt_bytes(data: &[u8]) -> String {
    const SHOWN: usize = 16;

    let head = data
        .iter()
        .take(SHOWN)
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    match data.len() > SHOWN {
        true => format!("{head} .."),
        false => head,
    }
}

fn fmt_list<T>(items: &[T], fmt: impl Fn(&T) -> String) -> String {
    items.iter().map(fmt).collect::<Vec<_>>().join(", ")
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::I32(v) => write!(f, "{v}"),
            Constant::U32(v) => write!(f, "{v}"),
            Constant::Access(v) => write!(f, "{}", fmt_flags(*v)),
            Constant::ClearValue(v) => {
                let color = unsafe { v.0.color.float32 };
                write!(
                    f,
                    "[{:.3}, {:.3}, {:.3}, {:.3}]",
                    color[0], color[1], color[2], color[3]
                )
            },
            Constant::Extent2D(e) => write!(f, "{}x{}", e.width, e.height),
            Constant::Extent3D(e) => match e.depth {
                1 => write!(f, "{}x{}", e.width, e.height),
                depth => write!(f, "{}x{}x{depth}", e.width, e.height),
            },
        }
    }
}

impl IR {
    pub fn display<'a>(&'a self, program: &'a Program<'a>) -> Printer<'a> { Printer { ir: self, program } }

    pub fn draw_state(&self) -> Option<String> {
        let (state, dynamic) = match self {
            IR::Draw { state, dynamic, .. } | IR::DrawIndexed { state, dynamic, .. } => (state, dynamic),
            _ => return None,
        };

        let mut parts = vec![format!(
            "formats=[{}]",
            fmt_list(&state.rendering.color_formats, |format| format!("{format:?}"))
        )];
        if state.rendering.samples != vk::SampleCountFlags::TYPE_1 {
            parts.push(format!("samples={}", fmt_samples(state.rendering.samples)));
        }
        if state.topology != vk::PrimitiveTopology::TRIANGLE_LIST {
            parts.push(format!("topology={:?}", state.topology));
        }
        if state.rasterization != RasterizationState::default() {
            parts.push(format!("raster={{{}}}", fmt_rasterization(&state.rasterization)));
        }
        parts.push(format!("blend=[{}]", fmt_list(&state.blend, fmt_blend)));

        parts.push(match state.viewports.is_empty() {
            true => format!("viewports=dynamic[{}]", fmt_list(&dynamic.viewports, fmt_viewport)),
            false => format!("viewports=[{}]", fmt_list(&state.viewports, fmt_resolved_viewport)),
        });
        parts.push(match state.scissors.is_empty() {
            true => format!("scissors=dynamic[{}]", fmt_list(&dynamic.scissors, fmt_rect)),
            false => format!("scissors=[{}]", fmt_list(&state.scissors, fmt_vk_rect)),
        });

        let push = &dynamic.push_constants;
        if !push.is_empty() {
            parts.push(format!("push_constants=[{}..{}]", push.offset, push.end()));
        }

        Some(parts.join(" "))
    }

    fn fmt_with(&self, f: &mut fmt::Formatter<'_>, p: &Program<'_>) -> fmt::Result {
        match self {
            IR::Type(ty) => match ty {
                Type::Image { format, samples } => {
                    write!(f, "type image {format:?} samples={}", fmt_samples(*samples))
                },
                Type::Buffer => write!(f, "type buffer"),
            },
            IR::Constant(constant) => write!(f, "const {constant}"),
            IR::Array { ty: _, elements } => {
                write!(f, "array [{}]", fmt_list(elements, |id| p.operand(*id).to_string()))
            },
            IR::Index { array, index } => write!(f, "index {}[{}]", p.operand(*array), p.operand(*index)),
            IR::ConstructBuffer { buffer, size, name } => write!(
                f,
                "buffer{} size={} handle={:?}",
                fmt_name(name),
                p.operand(*size),
                buffer.handle()
            ),
            IR::ConstructImage {
                image,
                image_view,
                view_type,
                extent,
                format,
                samples,
                base_level,
                level_count,
                base_layer,
                layer_count,
                usage,
                initial_layout,
                name,
            } => {
                write!(f, "image{} {} {format:?}", fmt_name(name), p.operand(*extent))?;
                if *samples != vk::SampleCountFlags::TYPE_1 {
                    write!(f, " samples={}", fmt_samples(*samples))?;
                }
                if *view_type != vk::ImageViewType::TYPE_2D {
                    write!(f, " view_type={view_type:?}")?;
                }
                if !(p.is_u32(*base_level, 0) && p.is_u32(*level_count, 1)) {
                    write!(
                        f,
                        " base_level={} level_count={}",
                        p.operand(*base_level),
                        p.operand(*level_count)
                    )?;
                }
                if !(p.is_u32(*base_layer, 0) && p.is_u32(*layer_count, 1)) {
                    write!(
                        f,
                        " base_layer={} layer_count={}",
                        p.operand(*base_layer),
                        p.operand(*layer_count)
                    )?;
                }
                write!(f, " usage={} layout={}", fmt_usage(*usage), fmt_layout(*initial_layout))?;

                match image.is_null() {
                    true => write!(f, " transient"),
                    false => write!(f, " handle={:?} view={image_view:?}", image.handle()),
                }
            },
            IR::AcquireNextImage {
                swapchain,
                attachments,
                present_semaphores,
            } => write!(
                f,
                "acquire_next_image swapchain={swapchain:?} attachments={} present_semaphores={}",
                p.operand(*attachments),
                present_semaphores.len()
            ),
            IR::Acquire { resource, access } => {
                write!(f, "acquire {} access={}", p.operand(*resource), p.operand(*access))
            },
            IR::Release {
                resource,
                access,
                dst_domain,
            } => write!(
                f,
                "release {} access={} domain={}",
                p.operand(*resource),
                p.operand(*access),
                fmt_flags(*dst_domain)
            ),
            IR::CallOpaque {
                args, returns, domain, ..
            } => write!(
                f,
                "call.opaque domain={} args={} returns={}",
                fmt_flags(*domain),
                p.operand(*args),
                p.operand(*returns)
            ),
            IR::Clear { attachment, color } => {
                write!(f, "clear {} color={}", p.operand(*attachment), p.operand(*color))
            },
            IR::Blit { src, dst, filter } => {
                write!(f, "blit {} -> {} filter={filter:?}", p.operand(*src), p.operand(*dst))
            },
            IR::CopyBufferToImage { buffer, image, region } => {
                write!(
                    f,
                    "copy_buffer_to_image {} -> {} buffer_offset={} image={},{} {}x{}",
                    p.operand(*buffer),
                    p.operand(*image),
                    region.buffer_offset,
                    region.image_offset.x,
                    region.image_offset.y,
                    region.image_extent.width,
                    region.image_extent.height
                )?;
                if region.image_extent.depth != 1 {
                    write!(f, "x{}", region.image_extent.depth)?;
                }
                match region.mip_level {
                    0 => Ok(()),
                    level => write!(f, " mip={level}"),
                }
            },
            IR::BeginRendering {
                color_attachments,
                render_area,
                name,
            } => {
                write!(
                    f,
                    "begin_rendering{} color=[{}]",
                    fmt_name(name),
                    fmt_list(color_attachments, |id| p.operand(*id).to_string())
                )?;
                match render_area {
                    Some(area) => write!(f, " area={}", p.operand(*area)),
                    None => Ok(()),
                }
            },
            IR::BindPipeline {
                pipeline, bind_point, ..
            } => write!(f, "bind_pipeline {pipeline} {bind_point:?}"),
            IR::BindVertexBuffers {
                first_binding,
                buffers,
                offsets,
                ..
            } => {
                let bound = buffers
                    .iter()
                    .enumerate()
                    .map(|(index, id)| match offsets.get(index).copied().unwrap_or(0) {
                        0 => p.operand(*id).to_string(),
                        offset => format!("{}@{offset}", p.operand(*id)),
                    })
                    .collect::<Vec<_>>();
                write!(
                    f,
                    "bind_vertex_buffers first={first_binding} buffers=[{}]",
                    bound.join(", ")
                )
            },
            IR::BindIndexBuffer {
                buffer,
                offset,
                index_type,
                ..
            } => {
                write!(f, "bind_index_buffer {} type={index_type:?}", p.operand(*buffer))?;
                match offset {
                    0 => Ok(()),
                    offset => write!(f, " offset={offset}"),
                }
            },
            IR::SampleImage { image, .. } => write!(f, "sample_image {}", p.operand(*image)),
            IR::SetState { change, .. } => {
                write!(f, "set_state ")?;
                match change {
                    StateChange::PrimitiveTopology(v) => write!(f, "topology={v:?}"),
                    StateChange::Rasterization(v) => write!(f, "rasterization={{{}}}", fmt_rasterization(v)),
                    StateChange::DynamicState(v) => write!(f, "dynamic_state={}", fmt_flags(*v)),
                    StateChange::Viewport { index, viewport } => {
                        write!(f, "viewport[{index}]={}", fmt_viewport(viewport))
                    },
                    StateChange::Scissor { index, rect } => write!(f, "scissor[{index}]={}", fmt_rect(rect)),
                    StateChange::ColorBlend { index, blend } => match index {
                        Some(index) => write!(f, "color_blend[{index}]={}", fmt_blend(blend)),
                        None => write!(f, "color_blend[*]={}", fmt_blend(blend)),
                    },
                    StateChange::PushConstants { offset, data } => write!(
                        f,
                        "push_constants[{}..{}]={}",
                        offset,
                        offset + data.len() as u32,
                        fmt_bytes(data)
                    ),
                }
            },
            IR::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
                pipeline,
                ..
            } => {
                write!(
                    f,
                    "draw verts={} insts={}",
                    p.operand(*vertex_count),
                    p.operand(*instance_count)
                )?;
                if !p.is_u32(*first_vertex, 0) {
                    write!(f, " first_vert={}", p.operand(*first_vertex))?;
                }
                if !p.is_u32(*first_instance, 0) {
                    write!(f, " first_inst={}", p.operand(*first_instance))?;
                }
                write!(f, " {}", fmt_pipeline(pipeline))
            },
            IR::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
                pipeline,
                ..
            } => {
                write!(
                    f,
                    "draw_indexed indices={} insts={}",
                    p.operand(*index_count),
                    p.operand(*instance_count)
                )?;
                if !p.is_u32(*first_index, 0) {
                    write!(f, " first_index={}", p.operand(*first_index))?;
                }
                write!(f, " vertex_offset={}", p.operand(*vertex_offset))?;
                if !p.is_u32(*first_instance, 0) {
                    write!(f, " first_inst={}", p.operand(*first_instance))?;
                }
                write!(f, " {}", fmt_pipeline(pipeline))
            },
            IR::EndRendering { .. } => write!(f, "end_rendering"),
            IR::MemoryBarrier { src_access, dst_access } => write!(
                f,
                "barrier.memory {} -> {}",
                p.operand(*src_access),
                p.operand(*dst_access)
            ),
            IR::ImageBarrier {
                src_access,
                dst_access,
                old_layout,
                new_layout,
                value,
            } => write!(
                f,
                "barrier.image {} {} -> {} access={} -> {}",
                p.operand(*value),
                fmt_layout(*old_layout),
                fmt_layout(*new_layout),
                p.operand(*src_access),
                p.operand(*dst_access)
            ),
        }
    }
}

fn fmt_pipeline(pipeline: &Option<PipelineId>) -> String {
    match pipeline {
        Some(pipeline) => format!("pipeline={pipeline}"),
        None => "pipeline=none".into(),
    }
}

impl fmt::Display for IR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.fmt_with(f, &Program::default()) }
}
