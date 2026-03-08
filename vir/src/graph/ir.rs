use ash::vk;
use bitflags::bitflags;

use crate::{DomainFlag, PassCallback};

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Access: u64 {
        const None = 0;
        const ColorRead = 1 << 0;
        const ColorWrite = 1 << 1;
        const ColorRW = Self::ColorRead.bits() | Self::ColorWrite.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Constant {
    I32(i32),
    U32(u32),
    Extent2D(vk::Extent2D),
    Extent3D(vk::Extent3D),
}

#[derive(Clone)]
pub enum IR {
    Constant(Constant),

    Array(Vec<ValueId>),

    // Construct ops
    ConstructBuffer {
        buffer: vk::Buffer,
        size: ValueId,
    },

    ConstructImage {
        image: vk::Image,
        image_view: vk::ImageView,
        extent: ValueId,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        base_level: ValueId,
        level_count: ValueId,
        base_layer: ValueId,
        layer_count: ValueId,
    },

    // Acquire ops
    AcquireNextImage {
        swapchain: vk::SwapchainKHR,
        attachments: ValueId,
    },

    Acquire {
        resource: ValueId,
        access: Access,
    },

    Release {
        resource: ValueId,
        access: Access,
        dst_domain: DomainFlag,
    },

    // Pass ops
    CallOpaque {
        args: Vec<ValueId>,
        returns: Vec<ValueId>,
        callback: PassCallback,
        domain: DomainFlag,
    },

    Clear {
        attachment: ValueId,
        color: vk::ClearValue,
    },
}

impl std::fmt::Display for IR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IR::Constant(c) => match c {
                Constant::I32(v) => write!(f, "const i32 {}", v),
                Constant::U32(v) => write!(f, "const u32 {}", v),
                Constant::Extent2D(e) => write!(f, "const Extent2D {{{}x{}}}", e.width, e.height),
                Constant::Extent3D(e) => write!(f, "const Extent3D {{{}x{}x{}}}", e.width, e.height, e.depth),
            },
            IR::Array(ids) => {
                write!(f, "[")?;
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", id.0)?;
                }
                write!(f, "]")
            },
            IR::ConstructBuffer { buffer, size } => {
                write!(f, "construct buffer={{mem: {:?}}} size=%{}", buffer, size.0)
            },
            IR::ConstructImage {
                image,
                image_view,
                extent,
                format,
                samples,
                base_level,
                level_count,
                base_layer,
                layer_count,
            } => {
                write!(
                    f,
                    "construct image={{mem: {:?}}} view={{mem: {:?}}} extent=%{} format={:?} samples={:?} \
                     levels=[%{}..%{}] layers=[%{}..%{}]",
                    image,
                    image_view,
                    extent.0,
                    format,
                    samples,
                    base_level.0,
                    level_count.0,
                    base_layer.0,
                    layer_count.0,
                )
            },
            IR::AcquireNextImage { swapchain, attachments } => {
                write!(
                    f,
                    "acquire_next_image swapchain={{mem: {:?}}} attachments=%{}",
                    swapchain, attachments.0
                )
            },
            IR::Acquire { resource, access } => {
                write!(f, "acquire resource=%{} access={:?}", resource.0, access)
            },
            IR::Release {
                resource,
                access,
                dst_domain,
            } => {
                write!(
                    f,
                    "release resource=%{} access={:?} domain={:?}",
                    resource.0, access, dst_domain
                )
            },
            IR::CallOpaque {
                args, returns, domain, ..
            } => {
                write!(f, "call.opaque domain={:?} args=[", domain)?;
                for (i, id) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", id.0)?;
                }
                write!(f, "] returns=[")?;
                for (i, id) in returns.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", id.0)?;
                }
                write!(f, "]")
            },
            IR::Clear { attachment, color } => {
                let rgba = unsafe { color.color.float32 };
                write!(
                    f,
                    "clear attachment=%{} color=({:.2}, {:.2}, {:.2}, {:.2})",
                    attachment.0, rgba[0], rgba[1], rgba[2], rgba[3]
                )
            },
        }
    }
}
