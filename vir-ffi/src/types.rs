use ash::vk;
use vir::{Access, BlendPreset, DynamicStateFlags, RasterizationState, Rect2D, ValueId, Viewport};

use crate::error::Failure;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirAttachment {
    pub value: u32,
    pub access: u64,
}

impl From<VirAttachment> for (ValueId, Access) {
    fn from(attachment: VirAttachment) -> Self {
        (ValueId(attachment.value), Access::from_bits_retain(attachment.access))
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirRect2D {
    pub tag: u32,
    pub absolute: vk::Rect2D,
    pub relative: [f32; 4],
}

impl VirRect2D {
    pub(crate) fn to_vir(self) -> Result<Rect2D, Failure> {
        match self.tag {
            0 => Ok(Rect2D::Framebuffer),
            1 => Ok(Rect2D::Absolute(self.absolute)),
            2 => {
                let [x, y, width, height] = self.relative;
                Ok(Rect2D::Relative { x, y, width, height })
            },
            tag => Err(Failure::new(format!("unknown rect tag {tag}"))),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirViewport {
    pub rect: VirRect2D,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl VirViewport {
    pub(crate) fn to_vir(self) -> Result<Viewport, Failure> {
        Ok(Viewport {
            rect: self.rect.to_vir()?,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirRasterizationState {
    pub polygon_mode: i32,
    pub cull_mode: u32,
    pub front_face: i32,
    pub line_width: f32,
    pub depth_clamp_enable: u32,
    pub rasterizer_discard_enable: u32,
}

impl From<VirRasterizationState> for RasterizationState {
    fn from(state: VirRasterizationState) -> Self {
        Self {
            polygon_mode: vk::PolygonMode::from_raw(state.polygon_mode),
            cull_mode: vk::CullModeFlags::from_raw(state.cull_mode),
            front_face: vk::FrontFace::from_raw(state.front_face),
            line_width: state.line_width,
            depth_clamp_enable: state.depth_clamp_enable != 0,
            rasterizer_discard_enable: state.rasterizer_discard_enable != 0,
        }
    }
}

pub(crate) fn blend_preset(preset: u32) -> Result<BlendPreset, Failure> {
    match preset {
        0 => Ok(BlendPreset::Off),
        1 => Ok(BlendPreset::AlphaBlend),
        2 => Ok(BlendPreset::PremultipliedAlphaBlend),
        3 => Ok(BlendPreset::Additive),
        preset => Err(Failure::new(format!("unknown blend preset {preset}"))),
    }
}

pub(crate) fn dynamic_state(flags: u32) -> DynamicStateFlags {
    DynamicStateFlags::from_bits_retain(flags)
}

pub(crate) unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        return &[];
    }

    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub(crate) unsafe fn text<'a>(ptr: *const u8, len: usize) -> Result<&'a str, Failure> {
    if ptr.is_null() && len != 0 {
        return Err(Failure::new("string pointer is null"));
    }

    std::str::from_utf8(unsafe { bytes(ptr, len) }).map_err(|err| Failure::new(format!("string is not UTF-8: {err}")))
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirDepthState {
    pub test_enable: u32,
    pub write_enable: u32,
    pub compare_op: i32,
}

impl From<VirDepthState> for vir::DepthState {
    fn from(state: VirDepthState) -> Self {
        Self {
            test_enable: state.test_enable != 0,
            write_enable: state.write_enable != 0,
            compare_op: vk::CompareOp::from_raw(state.compare_op),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_access_image_layout(access: u64) -> i32 {
    vk::ImageLayout::from(Access::from_bits_retain(access)).as_raw()
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirBufferCopy {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

impl From<VirBufferCopy> for vir::BufferCopy {
    fn from(copy: VirBufferCopy) -> Self {
        Self {
            src_offset: copy.src_offset,
            dst_offset: copy.dst_offset,
            size: copy.size,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirBufferImageCopy {
    pub buffer_offset: u64,
    pub mip_level: u32,
    pub offset_x: i32,
    pub offset_y: i32,
    pub offset_z: i32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl From<VirBufferImageCopy> for vir::BufferImageCopy {
    fn from(copy: VirBufferImageCopy) -> Self {
        Self {
            buffer_offset: copy.buffer_offset,
            image_offset: vk::Offset3D {
                x: copy.offset_x,
                y: copy.offset_y,
                z: copy.offset_z,
            },
            image_extent: vk::Extent3D {
                width: copy.width,
                height: copy.height,
                depth: copy.depth,
            },
            mip_level: copy.mip_level,
        }
    }
}

pub(crate) unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T], Failure> {
    if len == 0 {
        return Ok(&[]);
    }

    if ptr.is_null() {
        return Err(Failure::new("array pointer is null"));
    }

    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}
