use ash::vk;

pub mod frame;
pub mod persistent;

use crate::CommandBuffer;

pub use self::{
    frame::{FrameAllocator, SuperFrameAllocator},
    persistent::PersistentAllocator,
};

pub trait Allocator: std::fmt::Debug {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result>;
    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result>;
    fn deallocate_semaphore(&self, semaphore: vk::Semaphore);
    fn allocate_command_buffer(&mut self, queue_family: u32) -> Result<CommandBuffer, vk::Result>;
}

pub enum AllocatorKind<'a> {
    Persistent(&'a mut PersistentAllocator),
    Frame(&'a mut FrameAllocator),
}

impl<'a> AllocatorKind<'a> {
    pub fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        match self {
            AllocatorKind::Persistent(persistent_allocator) => persistent_allocator.allocate_binary_semaphore(),
            AllocatorKind::Frame(frame_allocator) => frame_allocator.allocate_binary_semaphore(),
        }
    }

    pub fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        match self {
            AllocatorKind::Persistent(persistent_allocator) => persistent_allocator.allocate_timeline_semaphore(),
            AllocatorKind::Frame(frame_allocator) => frame_allocator.allocate_timeline_semaphore(),
        }
    }

    pub fn deallocate_semaphore(&self, semaphore: vk::Semaphore) {
        match self {
            AllocatorKind::Persistent(persistent_allocator) => persistent_allocator.deallocate_semaphore(semaphore),
            AllocatorKind::Frame(frame_allocator) => frame_allocator.deallocate_semaphore(semaphore),
        }
    }

    pub fn allocate_command_buffer(&mut self, queue_family: u32) -> Result<CommandBuffer, vk::Result> {
        match self {
            AllocatorKind::Persistent(persistent_allocator) => {
                persistent_allocator.allocate_command_buffer(queue_family)
            },
            AllocatorKind::Frame(frame_allocator) => todo!(),
        }
    }
}
