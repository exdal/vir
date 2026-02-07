use ash::vk::{self};

use crate::allocator::Allocator;

#[derive(Default)]
pub struct Image {
    handle: vk::Image,
    allocation: Option<Box<dyn Allocator>>,
}

impl Image {
    pub fn new(handle: vk::Image, allocation: Option<Box<dyn Allocator>>) -> Self { Self { handle, allocation } }
}
