use std::{ops::Add, ptr::NonNull};

use ash::vk::{self, Handle};

use super::{Allocator, persistent::PersistentAllocator};
use crate::CommandBuffer;

#[derive(Debug)]
pub struct FrameAllocator {
    issued_frame: usize,
    device: NonNull<ash::Device>,
    upstream: PersistentAllocator,
    cmd_pool: vk::CommandPool,
    semaphores: Vec<vk::Semaphore>,
    image_views: Vec<vk::ImageView>,
    cmd_buffers: Vec<vk::CommandBuffer>,
    timeline_waits: Vec<(vk::Semaphore, u64)>,
}

impl FrameAllocator {
    fn new(device: NonNull<ash::Device>, upstream: PersistentAllocator) -> Self {
        Self {
            issued_frame: 0,
            device,
            upstream,
            cmd_pool: vk::CommandPool::null(),
            semaphores: Vec::default(),
            image_views: Vec::default(),
            cmd_buffers: Vec::default(),
            timeline_waits: Vec::default(),
        }
    }

    pub fn add_timeline_wait(&mut self, semaphore: vk::Semaphore, value: u64) {
        self.timeline_waits.push((semaphore, value));
    }

    fn wait_idle(&self) -> Result<(), vk::Result> {
        if self.timeline_waits.is_empty() {
            return Ok(());
        }

        let (semaphores, values): (Vec<_>, Vec<_>) = self.timeline_waits.iter().copied().unzip();
        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);

        unsafe { self.device.as_ref().wait_semaphores(&wait_info, u64::MAX) }
    }

    fn ensure_cmd_pool(&mut self, queue_family: u32) -> Result<(), vk::Result> {
        if !self.cmd_pool.is_null() {
            return Ok(());
        }

        self.cmd_pool = self.upstream.allocate_command_pool(queue_family)?;

        Ok(())
    }

    fn deallocate(&mut self, issued_frame: usize) -> Result<(), vk::Result> {
        self.wait_idle()?;
        self.timeline_waits.clear();

        if !self.cmd_pool.is_null() {
            if !self.cmd_buffers.is_empty() {
                unsafe {
                    self.device
                        .as_ref()
                        .free_command_buffers(self.cmd_pool, &self.cmd_buffers)
                };
                self.cmd_buffers.clear();
            }
            self.upstream.reset_command_pool(self.cmd_pool, false)?;
        }

        self.semaphores
            .iter()
            .for_each(|sema| self.upstream.deallocate_semaphore(*sema));
        self.semaphores.clear();

        self.image_views
            .iter()
            .for_each(|view| self.upstream.deallocate_image_view(*view));
        self.image_views.clear();

        self.issued_frame = issued_frame;

        Ok(())
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

    fn allocate_command_buffer(&mut self, queue_family: u32) -> Result<CommandBuffer, vk::Result> {
        self.ensure_cmd_pool(queue_family)?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        let cmd_buffer = unsafe { self.device.as_ref().allocate_command_buffers(&alloc_info) }?[0];
        self.cmd_buffers.push(cmd_buffer);

        Ok(CommandBuffer::new(self.device, cmd_buffer))
    }

    fn allocate_image_view(
        &mut self, image: vk::Image, format: vk::Format, view_type: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
    ) -> Result<vk::ImageView, vk::Result> {
        self.upstream
            .allocate_image_view(image, format, view_type, subresource_range)
            .inspect(|x| self.image_views.push(*x))
    }

    fn deallocate_image_view(&mut self, _: vk::ImageView) {}
}

pub struct SuperFrameAllocator {
    frames: Vec<FrameAllocator>,
    frame_counter: usize,
    frames_in_flight: usize,
}

impl SuperFrameAllocator {
    pub fn new(device: NonNull<ash::Device>, frames_in_flight: usize) -> Self {
        let frames = (0..frames_in_flight)
            .map(|_| FrameAllocator::new(device, PersistentAllocator::new(device)))
            .collect::<Vec<FrameAllocator>>();

        Self {
            frames,
            frame_counter: 0,
            frames_in_flight,
        }
    }

    fn current_frame(&mut self) -> &mut FrameAllocator {
        self.frames.get_mut(self.frame_counter % self.frames_in_flight).unwrap()
    }

    pub fn get_next_frame(&mut self) -> Result<&mut FrameAllocator, vk::Result> {
        self.frame_counter = self.frame_counter.add(1);
        let issued_frame = self.frame_counter;
        let frame = self.current_frame();
        frame.deallocate(issued_frame)?;

        Ok(frame)
    }
}
