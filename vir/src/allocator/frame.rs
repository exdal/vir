use std::{cell::RefCell, collections::HashMap, ops::Add, ptr::NonNull, rc::Rc};

use ash::vk::{self, Handle};

use super::{
    Allocator,
    MemoryAllocator,
    buffer_pool::align_up,
    image_cache::FrameImageCache,
    persistent::{PersistentAllocator, allocation_alignment},
};
use crate::{Buffer, BufferInfo, CommandBuffer, Image, ImageInfo, MemoryLocation, SamplerInfo};

const FRAME_BLOCK_SIZE: u64 = 16 * 1024 * 1024;
const IDLE_FRAMES: usize = 16;

#[derive(Debug)]
struct FrameSegment {
    backing: Buffer,
    used: u64,
    last_use_frame: usize,
}

#[derive(Debug)]
pub struct FrameAllocator {
    issued_frame: usize,
    device: NonNull<ash::Device>,
    upstream: Rc<RefCell<PersistentAllocator>>,
    limits: vk::PhysicalDeviceLimits,
    cmd_pool: vk::CommandPool,
    semaphores: Vec<vk::Semaphore>,
    image_views: Vec<vk::ImageView>,
    samplers: Vec<vk::Sampler>,
    buffer_segments: HashMap<(MemoryLocation, u32), Vec<FrameSegment>>,
    next_buffer_id: u64,
    image_cache: FrameImageCache,
    cmd_buffers: Vec<vk::CommandBuffer>,
    timeline_waits: Vec<(vk::Semaphore, u64)>,
}

impl FrameAllocator {
    fn new(
        device: NonNull<ash::Device>, upstream: Rc<RefCell<PersistentAllocator>>, limits: vk::PhysicalDeviceLimits,
    ) -> Self {
        Self {
            issued_frame: 0,
            device,
            upstream,
            limits,
            cmd_pool: vk::CommandPool::null(),
            semaphores: Vec::new(),
            image_views: Vec::new(),
            samplers: Vec::new(),
            buffer_segments: HashMap::new(),
            next_buffer_id: 1 << 63,
            image_cache: FrameImageCache::default(),
            cmd_buffers: Vec::new(),
            timeline_waits: Vec::new(),
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
        if self.cmd_pool.is_null() {
            self.cmd_pool = self.upstream.borrow().allocate_command_pool(queue_family)?;
        }
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
            self.upstream.borrow().reset_command_pool(self.cmd_pool, false)?;
        }

        for semaphore in self.semaphores.drain(..) {
            self.upstream.borrow().deallocate_semaphore(semaphore);
        }
        for view in self.image_views.drain(..) {
            self.upstream.borrow_mut().deallocate_image_view(view);
        }
        for sampler in self.samplers.drain(..) {
            self.upstream.borrow_mut().deallocate_sampler(sampler);
        }

        let mut idle_buffers = Vec::new();
        for segments in self.buffer_segments.values_mut() {
            segments.retain_mut(|segment| {
                if issued_frame.saturating_sub(segment.last_use_frame) > IDLE_FRAMES {
                    idle_buffers.push(segment.backing);
                    false
                } else {
                    segment.used = 0;
                    true
                }
            });
        }
        self.buffer_segments.retain(|_, segments| !segments.is_empty());
        for buffer in idle_buffers {
            self.upstream.borrow_mut().deallocate_buffer(buffer);
        }

        for image in self.image_cache.retire(issued_frame) {
            self.upstream.borrow_mut().deallocate_image(image);
        }
        self.issued_frame = issued_frame;
        Ok(())
    }
}

impl Drop for FrameAllocator {
    fn drop(&mut self) {
        if let Err(err) = self.wait_idle() {
            tracing::error!(?err, "failed to wait for a frame before tearing it down");
        }
        if !self.cmd_pool.is_null() {
            let device = unsafe { self.device.as_ref() };
            if !self.cmd_buffers.is_empty() {
                unsafe { device.free_command_buffers(self.cmd_pool, &self.cmd_buffers) };
            }
            unsafe { device.destroy_command_pool(self.cmd_pool, None) };
        }
        for semaphore in self.semaphores.drain(..) {
            self.upstream.borrow().deallocate_semaphore(semaphore);
        }
        for view in self.image_views.drain(..) {
            self.upstream.borrow_mut().deallocate_image_view(view);
        }
        for sampler in self.samplers.drain(..) {
            self.upstream.borrow_mut().deallocate_sampler(sampler);
        }
        for image in self.image_cache.drain() {
            self.upstream.borrow_mut().deallocate_image(image);
        }
        for segments in self.buffer_segments.values() {
            for segment in segments {
                self.upstream.borrow_mut().deallocate_buffer(segment.backing);
            }
        }
    }
}

impl Allocator for FrameAllocator {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let semaphore = self.upstream.borrow_mut().allocate_binary_semaphore()?;
        self.semaphores.push(semaphore);
        Ok(semaphore)
    }

    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let semaphore = self.upstream.borrow_mut().allocate_timeline_semaphore()?;
        self.semaphores.push(semaphore);
        Ok(semaphore)
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
        let view = self
            .upstream
            .borrow_mut()
            .allocate_image_view(image, format, view_type, subresource_range)?;
        self.image_views.push(view);
        Ok(view)
    }

    fn deallocate_image_view(&mut self, _: vk::ImageView) {}

    fn allocate_sampler(&mut self, info: &SamplerInfo) -> Result<vk::Sampler, vk::Result> {
        let sampler = self.upstream.borrow_mut().allocate_sampler(info)?;
        self.samplers.push(sampler);
        Ok(sampler)
    }

    fn deallocate_sampler(&mut self, _: vk::Sampler) {}

    fn allocate_buffer(&mut self, info: &BufferInfo) -> Result<Buffer, vk::Result> {
        if info.size == 0 {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let alignment = allocation_alignment(info, &self.limits).ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let id = self.next_buffer_id;
        self.next_buffer_id = id.checked_add(1).ok_or(vk::Result::ERROR_TOO_MANY_OBJECTS)?;
        let key = (info.location, info.usage.as_raw());
        if let Some(segments) = self.buffer_segments.get_mut(&key) {
            for segment in segments {
                let Some(offset) = align_up(segment.used, alignment) else {
                    continue;
                };
                let Some(end) = offset.checked_add(info.size) else {
                    continue;
                };
                if end <= segment.backing.size() {
                    segment.used = end;
                    segment.last_use_frame = self.issued_frame;
                    return Ok(segment
                        .backing
                        .slice(offset, info.size, id)
                        .expect("frame slice fits segment"));
                }
            }
        }

        let size = align_up(info.size.max(FRAME_BLOCK_SIZE), FRAME_BLOCK_SIZE)
            .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
        let backing = self.upstream.borrow_mut().allocate_buffer(&BufferInfo {
            size,
            usage: info.usage,
            location: info.location,
            alignment: 1,
            name: format!("frame buffer segment {:?}", key),
        })?;
        let buffer = backing
            .slice(0, info.size, id)
            .expect("new segment holds first allocation");
        self.buffer_segments.entry(key).or_default().push(FrameSegment {
            backing,
            used: info.size,
            last_use_frame: self.issued_frame,
        });
        Ok(buffer)
    }

    fn deallocate_buffer(&mut self, _: Buffer) {}

    fn allocate_image(&mut self, info: &ImageInfo) -> Result<Image, vk::Result> {
        self.image_cache.acquire(info, self.issued_frame, || {
            self.upstream.borrow_mut().allocate_image(info)
        })
    }

    fn deallocate_image(&mut self, _: Image) {}
}

pub struct SuperFrameAllocator {
    frames: Vec<FrameAllocator>,
    frame_counter: usize,
    frames_in_flight: usize,
}

impl SuperFrameAllocator {
    pub fn new(
        device: NonNull<ash::Device>, memory: MemoryAllocator, limits: vk::PhysicalDeviceLimits,
        queue_families: Vec<u32>, frames_in_flight: usize,
    ) -> Self {
        assert!(frames_in_flight > 0, "a super-frame allocator needs at least one frame");
        let upstream = Rc::new(RefCell::new(PersistentAllocator::new(
            device,
            memory,
            limits,
            queue_families,
        )));
        let frames = (0..frames_in_flight)
            .map(|_| FrameAllocator::new(device, upstream.clone(), limits))
            .collect::<Vec<_>>();
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
