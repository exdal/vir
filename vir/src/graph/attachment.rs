#![allow(dead_code)]

use ash::vk;

use crate::Image;

#[derive(Debug, Default, Clone)]
pub struct ImageAttachment {
    image: Image,
    image_view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent3D,
    samples: vk::SampleCountFlags,
    layout: vk::ImageLayout,
    subresource_range: vk::ImageSubresourceRange,
}

impl ImageAttachment {
    pub fn new(
        image: Image, format: vk::Format, extent: vk::Extent3D, samples: vk::SampleCountFlags, layout: vk::ImageLayout,
    ) -> Self {
        Self {
            image,
            image_view: vk::ImageView::null(),
            format,
            extent,
            samples,
            layout,
            subresource_range: vk::ImageSubresourceRange::default(),
        }
    }

    pub fn from_image(image: &Image, layout: vk::ImageLayout) -> Self {
        Self {
            image: *image,
            image_view: vk::ImageView::null(),
            format: image.format(),
            extent: image.extent(),
            samples: image.samples(),
            layout,
            subresource_range: image.subresource_range(),
        }
    }

    pub fn with_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn with_image_view(mut self, image_view: vk::ImageView) -> Self {
        self.image_view = image_view;
        self
    }

    pub fn with_subresource_range(mut self, subresource: vk::ImageSubresourceRange) -> Self {
        self.subresource_range = subresource;
        self
    }

    pub fn image(&self) -> &Image { &self.image }

    pub fn image_view(&self) -> vk::ImageView { self.image_view }

    pub fn format(&self) -> vk::Format { self.format }

    pub fn extent(&self) -> vk::Extent3D { self.extent }

    pub fn samples(&self) -> vk::SampleCountFlags { self.samples }

    pub fn layout(&self) -> vk::ImageLayout { self.layout }

    pub fn base_level(&self) -> u32 { self.subresource_range.base_mip_level }

    pub fn level_count(&self) -> u32 { self.subresource_range.level_count }

    pub fn base_layer(&self) -> u32 { self.subresource_range.base_array_layer }

    pub fn layer_count(&self) -> u32 { self.subresource_range.layer_count }

    pub fn subresource_range(&self) -> vk::ImageSubresourceRange { self.subresource_range }
}
