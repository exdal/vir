use std::{ops::Add, sync::Arc};

use ash::vk;

use super::{Allocator, persistent::PersistentAllocator};

pub struct FrameAllocator {
    issued_frame: usize,
    upstream: PersistentAllocator,
    semaphores: Vec<vk::Semaphore>,
}

impl FrameAllocator {
    fn new(upstream: PersistentAllocator) -> Self {
        Self {
            issued_frame: 0,
            upstream,
            semaphores: Vec::default(),
        }
    }

    fn deallocate(&mut self, issued_frame: usize) {
        self.semaphores
            .iter()
            .for_each(|sema| self.upstream.deallocate_semaphore(*sema));
        self.semaphores.clear();

        self.issued_frame = issued_frame;
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

        Self {
            frames,
            frame_counter: 0,
            frames_in_flight,
        }
    }

    fn get_last_frame(&mut self) -> &mut FrameAllocator {
        self.frames.get_mut(self.frame_counter % self.frames_in_flight).unwrap()
    }

    pub fn get_next_frame(&mut self) -> &mut FrameAllocator {
        let issued_frame = self.frame_counter.add(1);
        let frame = self.get_last_frame();
        // wait for frame here
        frame.deallocate(issued_frame);

        frame
    }
}
