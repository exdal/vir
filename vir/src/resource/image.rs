use std::hash::{Hash, Hasher};

use ash::vk::{self, Handle};
pub use gpu_allocator::MemoryLocation;

/// What an allocator needs to hand back an [`Image`].
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub image_type: vk::ImageType,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub location: MemoryLocation,
    pub name: String,
}

impl Default for ImageInfo {
    fn default() -> Self {
        Self {
            extent: vk::Extent3D::default().depth(1),
            format: vk::Format::UNDEFINED,
            usage: vk::ImageUsageFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            location: MemoryLocation::GpuOnly,
            name: String::new(),
        }
    }
}

impl ImageInfo {
    pub fn color_target(extent: vk::Extent2D, format: vk::Format) -> Self {
        Self {
            extent: vk::Extent3D::default()
                .width(extent.width)
                .height(extent.height)
                .depth(1),
            format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        }
    }

    pub fn depth_target(extent: vk::Extent2D, format: vk::Format) -> Self {
        Self {
            extent: vk::Extent3D::default()
                .width(extent.width)
                .height(extent.height)
                .depth(1),
            format,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        }
    }

    pub fn texture(extent: vk::Extent2D, format: vk::Format) -> Self {
        Self {
            extent: vk::Extent3D::default()
                .width(extent.width)
                .height(extent.height)
                .depth(1),
            format,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ..Default::default()
        }
    }

    pub fn with_usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage |= usage;
        self
    }

    pub fn with_mips(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    pub fn with_layers(mut self, array_layers: u32) -> Self {
        self.array_layers = array_layers;
        self
    }

    pub fn with_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask(self.format))
            .level_count(self.mip_levels)
            .layer_count(self.array_layers)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferImageCopy {
    pub buffer_offset: u64,
    pub image_offset: vk::Offset3D,
    pub image_extent: vk::Extent3D,
    pub mip_level: u32,
}

impl Default for BufferImageCopy {
    fn default() -> Self {
        Self {
            buffer_offset: 0,
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D::default().depth(1),
            mip_level: 0,
        }
    }
}

impl BufferImageCopy {
    pub fn whole(extent: vk::Extent3D) -> Self {
        Self {
            image_extent: extent,
            ..Default::default()
        }
    }

    pub fn region(offset: vk::Offset2D, extent: vk::Extent2D) -> Self {
        Self {
            image_offset: vk::Offset3D {
                x: offset.x,
                y: offset.y,
                z: 0,
            },
            image_extent: vk::Extent3D::default()
                .width(extent.width)
                .height(extent.height)
                .depth(1),
            ..Default::default()
        }
    }

    pub fn with_buffer_offset(mut self, offset: u64) -> Self {
        self.buffer_offset = offset;
        self
    }

    pub fn with_mip_level(mut self, mip_level: u32) -> Self {
        self.mip_level = mip_level;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.image_extent.width == 0 || self.image_extent.height == 0 || self.image_extent.depth == 0
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Image {
    handle: vk::Image,
    extent: vk::Extent3D,
    format: vk::Format,
    samples: vk::SampleCountFlags,
    mip_levels: u32,
    array_layers: u32,
    usage: vk::ImageUsageFlags,
}

impl Image {
    pub(crate) fn new(handle: vk::Image, info: &ImageInfo) -> Self {
        Self {
            handle,
            extent: info.extent,
            format: info.format,
            samples: info.samples,
            mip_levels: info.mip_levels,
            array_layers: info.array_layers,
            usage: info.usage,
        }
    }

    pub fn imported(
        handle: vk::Image, format: vk::Format, extent: vk::Extent3D, samples: vk::SampleCountFlags,
    ) -> Self {
        Self {
            handle,
            extent,
            format,
            samples,
            mip_levels: 1,
            array_layers: 1,
            usage: vk::ImageUsageFlags::empty(),
        }
    }

    pub fn handle(&self) -> vk::Image { self.handle }

    pub fn extent(&self) -> vk::Extent3D { self.extent }

    pub fn format(&self) -> vk::Format { self.format }

    pub fn samples(&self) -> vk::SampleCountFlags { self.samples }

    pub fn mip_levels(&self) -> u32 { self.mip_levels }

    pub fn array_layers(&self) -> u32 { self.array_layers }

    pub fn usage(&self) -> vk::ImageUsageFlags { self.usage }

    pub fn is_null(&self) -> bool { self.handle.is_null() }

    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask(self.format))
            .level_count(self.mip_levels)
            .layer_count(self.array_layers)
    }
}

pub fn aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        },
        _ => vk::ImageAspectFlags::COLOR,
    }
}

impl From<&Image> for vk::Image {
    fn from(image: &Image) -> Self { image.handle }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool { self.handle == other.handle }
}

impl Eq for Image {}

impl Hash for Image {
    fn hash<H: Hasher>(&self, state: &mut H) { self.handle.hash(state); }
}
