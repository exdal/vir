use core::fmt;

use ash::vk;

use crate::{Access, ClearValue, DomainFlag, Image};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "%{}", self.0) }
}

impl fmt::Debug for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
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
    Constant(Constant),

    Array(Vec<ValueId>),

    // Construct ops
    DeclareBuffer {
        buffer: vk::Buffer,
        size: ValueId,
    },

    ConstructBuffer {
        buffer: ValueId,
    },

    DeclareImage {
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

    ConstructImage {
        image: ValueId,
    },

    // AcqRel ops
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

    // Pass ops
    CallOpaque {
        args: ValueId,
        returns: ValueId,
        // callback: PassCallback,
        domain: DomainFlag,
    },

    Clear {
        attachment: ValueId,
        color: ValueId,
    },

    // Sync ops
    MemoryBarrier {
        src_access_flags: ValueId,
        dst_access_flags: ValueId,
    },

    ImageBarrier {
        src_access_flags: ValueId,
        dst_access_flags: ValueId,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        value: ValueId,
    },
}

impl std::fmt::Display for IR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IR::Constant(c) => match c {
                Constant::I32(v) => write!(f, "const i32 {v}"),
                Constant::U32(v) => write!(f, "const u32 {v}"),
                Constant::Access(v) => write!(f, "const access {v:?}"),
                Constant::ClearValue(v) => write!(f, "const clear_value {v:?}"),
                Constant::Extent2D(e) => write!(f, "const Extent2D {{{}x{}}}", e.width, e.height),
                Constant::Extent3D(e) => write!(f, "const Extent3D {{{}x{}x{}}}", e.width, e.height, e.depth),
            },
            IR::Array(ids) => {
                write!(f, "[")?;
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{id}")?;
                }
                write!(f, "]")
            },
            IR::DeclareBuffer { buffer, size } => {
                write!(f, "declare buffer={{mem: {buffer:?}}} size={size}")
            },
            IR::ConstructBuffer { buffer } => {
                write!(f, "construct buffer={buffer}")
            },
            IR::DeclareImage {
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
            } => {
                write!(
                    f,
                    "declare image={{mem: {:?}}} view={{mem: {image_view:?}}} view_type=${view_type:?} \
                     extent={extent} format={format:?} samples={samples:?} levels=[{base_level}..{level_count}] \
                     layers=[{base_layer}..{layer_count}] usage={usage:?}",
                    image.handle
                )
            },
            IR::ConstructImage { image } => {
                write!(f, "construct image={image}")
            },
            IR::AcquireNextImage {
                swapchain,
                attachments,
                present_semaphores,
            } => {
                write!(
                    f,
                    "acquire_next_image swapchain={{mem: {swapchain:?}}} attachments={attachments} \
                     present_semaphores={present_semaphores:?}",
                )
            },
            IR::Acquire { resource, access } => {
                write!(f, "acquire resource={resource} access={access:?}")
            },
            IR::Release {
                resource,
                access,
                dst_domain,
            } => {
                write!(f, "release resource={resource} access={access} domain={dst_domain:?}")
            },
            IR::CallOpaque {
                args, returns, domain, ..
            } => {
                write!(f, "call.opaque domain={domain:?} args={args} returns={returns}")
            },
            IR::Clear { attachment, color } => {
                write!(f, "clear attachment={attachment} color={color}")
            },
            IR::MemoryBarrier {
                src_access_flags,
                dst_access_flags,
            } => {
                write!(f, "barrier.memory src={src_access_flags} dst={dst_access_flags}")
            },
            IR::ImageBarrier {
                src_access_flags,
                dst_access_flags,
                old_layout,
                new_layout,
                value,
            } => {
                write!(
                    f,
                    "barrier.image src={src_access_flags} dst={dst_access_flags} old_layout={old_layout:?} \
                     new_layout={new_layout:?} value={value}",
                )
            },
        }
    }
}
