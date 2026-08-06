pub mod allocator;

pub mod context;
pub mod core;
pub mod graph;
pub mod resource;

use std::{
    fmt,
    hash::{Hash, Hasher},
};

pub use allocator::{AllocatorKind, FrameAllocator, PersistentAllocator, SuperFrameAllocator};
use ash::vk;
pub use context::{Access, CommandBuffer, Context, DomainFlag};
pub use graph::{IR, ImageAttachment, Module, PassCallback, RenderGraph, RenderPass, Value, ValueId};
pub use resource::{
    BlendPreset,
    Buffer,
    BufferImageCopy,
    BufferInfo,
    ColorBlendAttachmentState,
    DescriptorBinding,
    DescriptorSets,
    DynamicStateFlags,
    DynamicValues,
    GraphicsPipelineInfo,
    Image,
    ImageInfo,
    MemoryLocation,
    PassState,
    PipelineId,
    PipelineLayout,
    PipelineState,
    PushConstants,
    RasterizationState,
    Rect2D,
    Reflection,
    RenderingState,
    ResolvedViewport,
    SamplerInfo,
    SetLayout,
    StateChange,
    SwapChain,
    TextureBinding,
    TextureId,
    VertexAttribute,
    VertexLayout,
    Viewport,
};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ClearValue(pub vk::ClearValue);

impl ClearValue {
    #[inline]
    pub const fn rgba_f32(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(vk::ClearValue {
            color: vk::ClearColorValue { float32: [r, g, b, a] },
        })
    }

    #[inline]
    fn bytes(&self) -> &[u8; size_of::<vk::ClearValue>()] {
        unsafe { &*(self as *const Self as *const [u8; size_of::<vk::ClearValue>()]) }
    }
}

impl fmt::Debug for ClearValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let color = unsafe { self.0.color.float32 };
        f.debug_struct("ClearValue").field("float32", &color).finish()
    }
}

impl PartialEq for ClearValue {
    #[inline]
    fn eq(&self, other: &Self) -> bool { self.bytes() == other.bytes() }
}

impl Eq for ClearValue {}

impl Hash for ClearValue {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) { self.bytes().hash(state); }
}

pub mod clear {
    use ash::vk;

    use super::ClearValue;

    pub mod f32 {
        use super::*;
        pub const WHITE: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0, 1.0, 1.0, 1.0],
            },
        });
        pub const BLACK: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        });
        pub const TRANSPARENT: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        });
    }

    pub mod u32 {
        use super::*;
        pub const WHITE: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue { uint32: [1, 1, 1, 1] },
        });
        pub const BLACK: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue { uint32: [0, 0, 0, 1] },
        });
        pub const TRANSPARENT: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue { uint32: [0, 0, 0, 0] },
        });
    }

    pub mod i32 {
        use super::*;
        pub const WHITE: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue { int32: [1, 1, 1, 1] },
        });
        pub const BLACK: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue { int32: [0, 0, 0, 1] },
        });
        pub const TRANSPARENT: ClearValue = ClearValue(vk::ClearValue {
            color: vk::ClearColorValue { int32: [0, 0, 0, 0] },
        });
    }
}
