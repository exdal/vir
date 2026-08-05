use std::{
    hash::{Hash, Hasher},
    rc::Rc,
};

use ash::vk::{self, Handle};

use crate::allocator::Allocator;

#[derive(Debug, Default, Clone)]
pub struct Image {
    pub handle: vk::Image,
    pub allocation: Option<Rc<dyn Allocator>>,
}

impl Image {
    pub fn new(handle: vk::Image, allocation: Option<Rc<dyn Allocator>>) -> Self { Self { handle, allocation } }

    pub fn is_null(&self) -> bool { self.handle.is_null() }
}

pub fn aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => {
            vk::ImageAspectFlags::DEPTH
        },
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
