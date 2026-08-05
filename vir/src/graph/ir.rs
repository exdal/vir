use core::fmt;

use ash::vk;

use crate::{Access, ClearValue, DomainFlag, DynamicValues, Image, PipelineId, PipelineState, StateChange, ValueId};

pub type Instr = (ValueId, IR);

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
        buffer: vk::Buffer,
        size: ValueId,
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

    BeginRendering {
        color_attachments: Vec<ValueId>,
        render_area: Option<ValueId>,
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
    let s: Vec<_> = flags
        .iter()
        .filter(|(f, _)| usage.contains(*f))
        .map(|(_, n)| *n)
        .collect();
    if s.is_empty() { "NONE".into() } else { s.join(" | ") }
}

impl fmt::Display for IR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IR::Type(ty) => match ty {
                Type::Image { format, samples } => write!(f, "type image format={format:?} samples={samples:?}"),
                Type::Buffer => write!(f, "type buffer"),
            },
            IR::Constant(c) => match c {
                Constant::I32(v) => write!(f, "const i32 {v}"),
                Constant::U32(v) => write!(f, "const u32 {v}"),
                Constant::Access(v) => write!(f, "const access {v:?}"),
                Constant::ClearValue(v) => write!(f, "const clear_value {v:?}"),
                Constant::Extent2D(e) => write!(f, "const Extent2D {{{}x{}}}", e.width, e.height),
                Constant::Extent3D(e) => write!(f, "const Extent3D {{{}x{}x{}}}", e.width, e.height, e.depth),
            },
            IR::Array { ty, elements } => {
                write!(f, "{ty} [")?;
                for (i, id) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{id}")?;
                }
                write!(f, "]")
            },
            IR::Index { array, index } => write!(f, "index {array}[{index}]"),
            IR::ConstructBuffer { buffer, size } => write!(f, "declare buffer={{mem: {buffer:?}}} size={size}"),
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
            } => write!(
                f,
                "declare image={{mem: {:?}}} view={{mem: {image_view:?}}} view_type={view_type:?} extent={extent} \
                 format={format:?} samples={samples:?} levels=[{base_level}..{level_count}] \
                 layers=[{base_layer}..{layer_count}] usage={}",
                image.handle,
                fmt_usage(*usage)
            ),
            IR::AcquireNextImage {
                swapchain,
                attachments,
                present_semaphores,
            } => write!(
                f,
                "acquire_next_image swapchain={{mem: {swapchain:?}}} attachments={attachments} \
                 present_semaphores={present_semaphores:?}"
            ),
            IR::Acquire { resource, access } => write!(f, "acquire resource={resource} access={access:?}"),
            IR::Release {
                resource,
                access,
                dst_domain,
            } => write!(f, "release resource={resource} access={access} domain={dst_domain:?}"),
            IR::CallOpaque {
                args, returns, domain, ..
            } => write!(f, "call.opaque domain={domain:?} args={args} returns={returns}"),
            IR::Clear { attachment, color } => write!(f, "clear attachment={attachment} color={color}"),
            IR::BeginRendering {
                color_attachments,
                render_area,
            } => {
                write!(f, "begin_rendering color=[")?;
                for (i, id) in color_attachments.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{id}")?;
                }
                match render_area {
                    Some(area) => write!(f, "] area={area}"),
                    None => write!(f, "] area=attachments"),
                }
            },
            IR::BindPipeline {
                pass,
                pipeline,
                bind_point,
            } => write!(f, "bind_pipeline {pass} pipeline={pipeline} bind_point={bind_point:?}"),
            IR::SetState { pass, change } => {
                write!(f, "set_state {pass} ")?;
                match change {
                    StateChange::PrimitiveTopology(v) => write!(f, "topology={v:?}"),
                    StateChange::Rasterization(v) => write!(
                        f,
                        "rasterization={{polygon={:?} cull={:?} front={:?} line_width={}}}",
                        v.polygon_mode, v.cull_mode, v.front_face, v.line_width
                    ),
                    StateChange::DynamicState(v) => write!(f, "dynamic_state={v:?}"),
                    StateChange::Viewport { index, viewport } => write!(f, "viewport[{index}]={viewport:?}"),
                    StateChange::Scissor { index, rect } => write!(f, "scissor[{index}]={rect:?}"),
                    StateChange::ColorBlend { index, blend } => match index {
                        Some(index) => write!(f, "color_blend[{index}]={blend:?}"),
                        None => write!(f, "color_blend[*]={blend:?}"),
                    },
                }
            },
            IR::Draw {
                pass,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
                pipeline,
                state,
                dynamic,
            } => {
                write!(
                    f,
                    "draw {pass} verts={vertex_count} insts={instance_count} first_vert={first_vertex} \
                     first_inst={first_instance} pipeline="
                )?;
                match pipeline {
                    Some(pipeline) => write!(f, "{pipeline}")?,
                    None => write!(f, "none")?,
                }
                write!(
                    f,
                    " formats={:?} samples={:?} topology={:?} polygon={:?} cull={:?} front={:?} blend={:?}",
                    state.rendering.color_formats,
                    state.rendering.samples,
                    state.topology,
                    state.rasterization.polygon_mode,
                    state.rasterization.cull_mode,
                    state.rasterization.front_face,
                    state.blend.iter().map(|blend| blend.blend_enable).collect::<Vec<_>>()
                )?;

                match state.viewports.is_empty() {
                    true => write!(f, " viewports=dynamic{:?}", dynamic.viewports)?,
                    false => write!(f, " viewports={:?}", state.viewports)?,
                }
                match state.scissors.is_empty() {
                    true => write!(f, " scissors=dynamic{:?}", dynamic.scissors),
                    false => write!(f, " scissors={:?}", state.scissors),
                }
            },
            IR::EndRendering { pass } => write!(f, "end_rendering {pass}"),
            IR::MemoryBarrier { src_access, dst_access } => {
                write!(f, "barrier.memory src={src_access} dst={dst_access}")
            },
            IR::ImageBarrier {
                src_access,
                dst_access,
                old_layout,
                new_layout,
                value,
            } => write!(
                f,
                "barrier.image src={src_access} dst={dst_access} old_layout={old_layout:?} new_layout={new_layout:?} \
                 value={value}"
            ),
        }
    }
}
