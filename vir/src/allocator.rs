use ash::vk;

pub mod frame;
pub mod persistent;

pub use self::{
    frame::{FrameAllocator, SuperFrameAllocator},
    persistent::PersistentAllocator,
};
use crate::CommandBuffer;

pub trait Allocator: std::fmt::Debug {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result>;
    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result>;
    fn deallocate_semaphore(&self, semaphore: vk::Semaphore);
    fn allocate_command_buffer(&mut self, queue_family: u32) -> Result<CommandBuffer, vk::Result>;
    fn allocate_image_view(
        &mut self, image: vk::Image, format: vk::Format, view_type: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
    ) -> Result<vk::ImageView, vk::Result>;
    fn deallocate_image_view(&mut self, image_view: vk::ImageView);
}

pub enum AllocatorKind<'a> {
    Persistent(&'a mut PersistentAllocator),
    Frame(&'a mut FrameAllocator),
}

impl<'a> AllocatorKind<'a> {
    pub fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        match self {
            AllocatorKind::Persistent(a) => a.allocate_binary_semaphore(),
            AllocatorKind::Frame(a) => a.allocate_binary_semaphore(),
        }
    }

    pub fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        match self {
            AllocatorKind::Persistent(a) => a.allocate_timeline_semaphore(),
            AllocatorKind::Frame(a) => a.allocate_timeline_semaphore(),
        }
    }

    pub fn deallocate_semaphore(&self, semaphore: vk::Semaphore) {
        match self {
            AllocatorKind::Persistent(a) => a.deallocate_semaphore(semaphore),
            AllocatorKind::Frame(a) => a.deallocate_semaphore(semaphore),
        }
    }

    pub fn allocate_command_buffer(&mut self, queue_family: u32) -> Result<CommandBuffer, vk::Result> {
        match self {
            AllocatorKind::Persistent(a) => a.allocate_command_buffer(queue_family),
            AllocatorKind::Frame(a) => a.allocate_command_buffer(queue_family),
        }
    }

    pub fn allocate_image_view(
        &mut self, image: vk::Image, format: vk::Format, view_type: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
    ) -> Result<vk::ImageView, vk::Result> {
        match self {
            AllocatorKind::Persistent(a) => a.allocate_image_view(image, format, view_type, subresource_range),
            AllocatorKind::Frame(a) => a.allocate_image_view(image, format, view_type, subresource_range),
        }
    }

    pub fn deallocate_image_view(&mut self, image_view: vk::ImageView) {
        match self {
            AllocatorKind::Persistent(a) => a.deallocate_image_view(image_view),
            AllocatorKind::Frame(a) => a.deallocate_image_view(image_view),
        }
    }

    pub fn add_timeline_wait(&mut self, semaphore: vk::Semaphore, value: u64) {
        match self {
            AllocatorKind::Persistent(_) => {},
            AllocatorKind::Frame(a) => a.add_timeline_wait(semaphore, value),
        }
    }
}
