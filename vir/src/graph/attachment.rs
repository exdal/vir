#![allow(dead_code)]

use ash::vk;

use crate::Image;

#[derive(Default)]
pub struct ImageAttachment {
    image: Image,
    image_view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent3D,
    samples: vk::SampleCountFlags,
    layout: vk::ImageLayout,
    subresource: vk::ImageSubresourceRange,
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
            subresource: vk::ImageSubresourceRange::default(),
        }
    }

    pub fn with_image_view(mut self, image_view: vk::ImageView) -> Self {
        self.image_view = image_view;
        self
    }

    pub fn with_subresource(mut self, subresource: vk::ImageSubresourceRange) -> Self {
        self.subresource = subresource;
        self
    }

    pub fn image(&self) -> &Image { &self.image }

    pub fn image_view(&self) -> vk::ImageView { self.image_view }

    pub fn format(&self) -> vk::Format { self.format }

    pub fn extent(&self) -> vk::Extent3D { self.extent }

    pub fn samples(&self) -> vk::SampleCountFlags { self.samples }

    pub fn base_level(&self) -> u32 { self.subresource.base_mip_level }

    pub fn level_count(&self) -> u32 { self.subresource.level_count }

    pub fn base_layer(&self) -> u32 { self.subresource.base_array_layer }

    pub fn layer_count(&self) -> u32 { self.subresource.layer_count }
}
