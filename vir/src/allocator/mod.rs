use ash::vk;

pub mod frame;
pub mod persistent;

use self::{frame::FrameAllocator, persistent::PersistentAllocator};

pub trait Allocator {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result>;
    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result>;
    fn deallocate_semaphore(&self, semaphore: vk::Semaphore);
}

pub enum AllocatorKind {
    Persistent(PersistentAllocator),
    Frame(FrameAllocator),
}

impl AllocatorKind {
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
}
