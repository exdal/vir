use std::{cell::RefCell, rc::Rc};

use ash::vk;

pub mod frame;
pub mod persistent;

pub use self::{
    frame::{FrameAllocator, SuperFrameAllocator},
    persistent::PersistentAllocator,
};
use crate::{Buffer, BufferInfo, CommandBuffer, Image, ImageInfo, SamplerInfo};

pub type MemoryAllocator = Rc<RefCell<gpu_allocator::vulkan::Allocator>>;

pub(crate) fn map_allocation_error(err: gpu_allocator::AllocationError) -> vk::Result {
    use gpu_allocator::AllocationError;

    tracing::error!(%err, "gpu memory allocation failed");
    match err {
        AllocationError::OutOfMemory => vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        AllocationError::FailedToMap(_) => vk::Result::ERROR_MEMORY_MAP_FAILED,
        _ => vk::Result::ERROR_INITIALIZATION_FAILED,
    }
}

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
    fn allocate_sampler(&mut self, info: &SamplerInfo) -> Result<vk::Sampler, vk::Result>;
    fn deallocate_sampler(&mut self, sampler: vk::Sampler);
    fn allocate_buffer(&mut self, info: &BufferInfo) -> Result<Buffer, vk::Result>;
    fn deallocate_buffer(&mut self, buffer: Buffer);
    fn allocate_image(&mut self, info: &ImageInfo) -> Result<Image, vk::Result>;
    fn deallocate_image(&mut self, image: Image);
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

    pub fn allocate_sampler(&mut self, info: &SamplerInfo) -> Result<vk::Sampler, vk::Result> {
        match self {
            AllocatorKind::Persistent(a) => a.allocate_sampler(info),
            AllocatorKind::Frame(a) => a.allocate_sampler(info),
        }
    }

    pub fn deallocate_sampler(&mut self, sampler: vk::Sampler) {
        match self {
            AllocatorKind::Persistent(a) => a.deallocate_sampler(sampler),
            AllocatorKind::Frame(a) => a.deallocate_sampler(sampler),
        }
    }

    pub fn allocate_buffer(&mut self, info: &BufferInfo) -> Result<Buffer, vk::Result> {
        match self {
            AllocatorKind::Persistent(a) => a.allocate_buffer(info),
            AllocatorKind::Frame(a) => a.allocate_buffer(info),
        }
    }

    pub fn deallocate_buffer(&mut self, buffer: Buffer) {
        match self {
            AllocatorKind::Persistent(a) => a.deallocate_buffer(buffer),
            AllocatorKind::Frame(a) => a.deallocate_buffer(buffer),
        }
    }

    pub fn allocate_image(&mut self, info: &ImageInfo) -> Result<Image, vk::Result> {
        match self {
            AllocatorKind::Persistent(a) => a.allocate_image(info),
            AllocatorKind::Frame(a) => a.allocate_image(info),
        }
    }

    pub fn deallocate_image(&mut self, image: Image) {
        match self {
            AllocatorKind::Persistent(a) => a.deallocate_image(image),
            AllocatorKind::Frame(a) => a.deallocate_image(image),
        }
    }

    pub fn free_command_buffers(&mut self) {
        match self {
            AllocatorKind::Persistent(a) => a.free_command_buffers(),
            AllocatorKind::Frame(_) => {},
        }
    }

    pub fn add_timeline_wait(&mut self, semaphore: vk::Semaphore, value: u64) {
        match self {
            AllocatorKind::Persistent(_) => {},
            AllocatorKind::Frame(a) => a.add_timeline_wait(semaphore, value),
        }
    }
}
