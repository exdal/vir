use ash::vk::{self, Handle};

use crate::allocator::Allocator;

#[derive(Default)]
pub struct Image {
    pub handle: vk::Image,
    pub allocation: Option<Box<dyn Allocator>>,
}

impl Image {
    pub fn new(handle: vk::Image, allocation: Option<Box<dyn Allocator>>) -> Self { Self { handle, allocation } }

    pub fn is_null(&self) -> bool { self.handle.is_null() }
}
