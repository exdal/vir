use std::rc::Rc;

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

impl From<&Image> for vk::Image {
    fn from(image: &Image) -> Self { image.handle }
}
