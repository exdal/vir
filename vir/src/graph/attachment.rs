#![allow(dead_code)]

use ash::vk;

use crate::resource::image::Image;

#[derive(Default)]
pub struct ImageAttachment {
    image: Image,
    image_view: Option<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent3D,
    layout: vk::ImageLayout,
    subresource: vk::ImageSubresourceRange,
}

impl ImageAttachment {
    pub fn new(image: Image, format: vk::Format, extent: vk::Extent3D, layout: vk::ImageLayout) -> Self {
        Self {
            image,
            image_view: None,
            format,
            extent,
            layout,
            subresource: vk::ImageSubresourceRange::default(),
        }
    }

    pub fn with_image_view(mut self, image_view: vk::ImageView) -> Self {
        self.image_view = Some(image_view);
        self
    }

    pub fn with_subresource(mut self, subresource: vk::ImageSubresourceRange) -> Self {
        self.subresource = subresource;
        self
    }

    pub fn base_level(&self) -> u32 { self.subresource.base_mip_level }

    pub fn level_count(&self) -> u32 { self.subresource.level_count }

    pub fn base_layer(&self) -> u32 { self.subresource.base_array_layer }

    pub fn layer_count(&self) -> u32 { self.subresource.layer_count }
}
