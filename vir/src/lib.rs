pub mod allocator;
pub mod context;
pub mod graph;
pub mod resource;

pub use allocator::{AllocatorKind, FrameAllocator, PersistentAllocator, SuperFrameAllocator};
use ash::vk;
pub use context::{CommandBuffer, Context, DomainFlag};
pub use graph::{Access, IR, ImageAttachment, PassCallback, RenderGraph, ValueId};
pub use resource::{Image, SwapChain};

pub trait ClearColor {
    const WHITE: vk::ClearValue;
    const BLACK: vk::ClearValue;
    const TRANSPARENT: vk::ClearValue;
}

impl ClearColor for f32 {
    const BLACK: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };
    const TRANSPARENT: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    };
    const WHITE: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [1.0, 1.0, 1.0, 1.0],
        },
    };
}

impl ClearColor for u32 {
    const BLACK: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue { uint32: [0, 0, 0, 1] },
    };
    const TRANSPARENT: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue { uint32: [0, 0, 0, 0] },
    };
    const WHITE: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue { uint32: [1, 1, 1, 1] },
    };
}

impl ClearColor for i32 {
    const BLACK: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue { int32: [0, 0, 0, 1] },
    };
    const TRANSPARENT: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue { int32: [0, 0, 0, 0] },
    };
    const WHITE: vk::ClearValue = vk::ClearValue {
        color: vk::ClearColorValue { int32: [1, 1, 1, 1] },
    };
}
