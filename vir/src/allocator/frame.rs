use std::sync::Arc;

use ash::vk;

use super::{Allocator, persistent::PersistentAllocator};

pub struct FrameAllocator {
    upstream: PersistentAllocator,
    semaphores: Vec<vk::Semaphore>,
}

impl FrameAllocator {
    fn new(upstream: PersistentAllocator) -> Self {
        Self {
            upstream,
            semaphores: Vec::default(),
        }
    }
}

impl Allocator for FrameAllocator {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        self.upstream
            .allocate_binary_semaphore()
            .inspect(|x| self.semaphores.push(*x))
    }

    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        self.upstream
            .allocate_timeline_semaphore()
            .inspect(|x| self.semaphores.push(*x))
    }

    fn deallocate_semaphore(&self, _: vk::Semaphore) {}
}

pub struct SuperFrameAllocator {
    frames: Vec<FrameAllocator>,
    frame_counter: usize,
    frames_in_flight: usize,
}

impl SuperFrameAllocator {
    pub fn new(device: Arc<ash::Device>, frames_in_flight: usize) -> Self {
        let frames = (0..frames_in_flight)
            .map(|_| FrameAllocator::new(PersistentAllocator::new(device.clone())))
            .collect::<Vec<FrameAllocator>>();

        SuperFrameAllocator {
            frames,
            frame_counter: 0,
            frames_in_flight,
        }
    }

    fn get_last_frame(&self) -> &FrameAllocator { self.frames.get(self.frame_counter % self.frames_in_flight).unwrap() }

    fn get_last_frame_mut(&mut self) -> &mut FrameAllocator {
        self.frames.get_mut(self.frame_counter % self.frames_in_flight).unwrap()
    }
}
