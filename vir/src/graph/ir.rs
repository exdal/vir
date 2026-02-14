use ash::vk;
use bitflags::bitflags;

use crate::{DomainFlag, Image, PassCallback};

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

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum Constant {
    I32(i32),
    U32(u32),
    Extent2D(vk::Extent2D),
    Extent3D(vk::Extent3D),
}

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
    AcquireSwapChain {
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
        src_domain: DomainFlag,
        dst_domain: DomainFlag,
    },

    // Pass ops
    CallOpaque {
        args: Vec<ValueId>,
        returns: Vec<ValueId>,
        callback: PassCallback,
        domain: DomainFlag,
    },
}
