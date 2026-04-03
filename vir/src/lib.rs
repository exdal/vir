pub mod allocator;

pub mod context;
pub mod core;
pub mod graph;
pub mod resource;

pub use allocator::{AllocatorKind, FrameAllocator, PersistentAllocator, SuperFrameAllocator};
pub use context::{Access, CommandBuffer, Context, DomainFlag};
pub use graph::{IR, ImageAttachment, Module, PassCallback, RenderGraph, Value, ValueId};
pub use resource::{Image, SwapChain};

pub mod clear {
    pub mod f32 {
        use ash::vk;
        pub const WHITE: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0, 1.0, 1.0, 1.0],
            },
        };
        pub const BLACK: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        pub const TRANSPARENT: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };
    }

    pub mod u32 {
        use ash::vk;
        pub const WHITE: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue { uint32: [1, 1, 1, 1] },
        };
        pub const BLACK: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue { uint32: [0, 0, 0, 1] },
        };
        pub const TRANSPARENT: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue { uint32: [0, 0, 0, 0] },
        };
    }

    pub mod i32 {
        use ash::vk;
        pub const WHITE: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue { int32: [1, 1, 1, 1] },
        };
        pub const BLACK: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue { int32: [0, 0, 0, 1] },
        };
        pub const TRANSPARENT: vk::ClearValue = vk::ClearValue {
            color: vk::ClearColorValue { int32: [0, 0, 0, 0] },
        };
    }
}
