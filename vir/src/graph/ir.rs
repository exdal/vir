use ash::vk;

use crate::{Access, DomainFlag, Image, PassCallback};

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

    // Sync ops
    MemoryBarrier {
        src_access_flags: Access,
        dst_access_flags: Access,
    },

    ImageBarrier {
        src_access_flags: Access,
        dst_access_flags: Access,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        value: ValueId,
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
            IR::DeclareBuffer { buffer, size } => {
                write!(f, "declare buffer={{mem: {:?}}} size=%{}", buffer, size.0)
            },
            IR::ConstructBuffer { buffer } => {
                write!(f, "construct buffer=%{}", buffer.0)
            },
            IR::DeclareImage {
                image,
                image_view,
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
                    "declare image={{mem: {:?}}} view={{mem: {:?}}} extent=%{} format={:?} samples={:?} \
                     levels=[%{}..%{}] layers=[%{}..%{}] usage={:?}",
                    image.handle,
                    image_view,
                    extent.0,
                    format,
                    samples,
                    base_level.0,
                    level_count.0,
                    base_layer.0,
                    layer_count.0,
                    usage,
                )
            },
            IR::ConstructImage { image } => {
                write!(f, "construct image=%{}", image.0)
            },
            IR::AcquireNextImage {
                swapchain,
                attachments,
                present_semaphores,
            } => {
                write!(
                    f,
                    "acquire_next_image swapchain={{mem: {:?}}} attachments=%{} present_semaphores=%{:?}",
                    swapchain, attachments.0, present_semaphores
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
            IR::MemoryBarrier {
                src_access_flags,
                dst_access_flags,
            } => write!(
                f,
                "barrier.memory src={:?} dst={:?}",
                src_access_flags, dst_access_flags
            ),
            IR::ImageBarrier {
                src_access_flags,
                dst_access_flags,
                old_layout,
                new_layout,
                value,
            } => write!(
                f,
                "barrier.image src={:?} dst={:?} old_layout={:?} new_layout={:?} value=%{:?}",
                src_access_flags, dst_access_flags, old_layout, new_layout, value.0
            ),
        }
    }
}
