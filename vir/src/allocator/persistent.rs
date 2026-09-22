use std::{collections::HashMap, ptr::NonNull};

use ash::vk::{self, Handle};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use super::{Allocator, MemoryAllocator, buffer_pool::Ranges, map_allocation_error};
use crate::{Buffer, BufferInfo, CommandBuffer, Image, ImageInfo, MemoryLocation, SamplerInfo};

const BUFFER_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferPoolKey {
    location: MemoryLocation,
    usage: u32,
}

struct BufferBlock {
    buffer: Buffer,
    ranges: Ranges,
}

enum BufferAllocationKind {
    Dedicated,
    Pooled { key: BufferPoolKey, backing: vk::Buffer },
}

struct BufferAllocation {
    value: Buffer,
    kind: BufferAllocationKind,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

pub(super) fn allocation_alignment(info: &BufferInfo, limits: &vk::PhysicalDeviceLimits) -> Option<u64> {
    if info.alignment == 0 {
        return None;
    }
    let minimum = 4_u64
        .max(limits.min_uniform_buffer_offset_alignment)
        .max(limits.min_storage_buffer_offset_alignment)
        .max(limits.min_texel_buffer_offset_alignment);
    (minimum / gcd(minimum, info.alignment)).checked_mul(info.alignment)
}

pub struct PersistentAllocator {
    device: NonNull<ash::Device>,
    memory: MemoryAllocator,
    limits: vk::PhysicalDeviceLimits,
    queue_families: Vec<u32>,
    next_allocation_id: u64,
    buffer_allocations: HashMap<u64, BufferAllocation>,
    buffer_blocks: HashMap<BufferPoolKey, Vec<BufferBlock>>,
    cmd_pool: vk::CommandPool,
    cmd_buffers: Vec<vk::CommandBuffer>,
    allocations: HashMap<vk::Buffer, Allocation>,
    image_allocations: HashMap<vk::Image, Allocation>,
}

impl PersistentAllocator {
    pub fn new(
        device: NonNull<ash::Device>, memory: MemoryAllocator, limits: vk::PhysicalDeviceLimits,
        mut queue_families: Vec<u32>,
    ) -> Self {
        queue_families.sort_unstable();
        queue_families.dedup();
        PersistentAllocator {
            device,
            memory,
            limits,
            queue_families,
            next_allocation_id: 1,
            buffer_allocations: HashMap::new(),
            buffer_blocks: HashMap::new(),
            cmd_pool: vk::CommandPool::null(),
            cmd_buffers: Vec::new(),
            allocations: HashMap::new(),
            image_allocations: HashMap::new(),
        }
    }

    pub fn allocate_command_pool(&self, queue_family: u32) -> Result<vk::CommandPool, vk::Result> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);

        unsafe { self.device.as_ref().create_command_pool(&create_info, None) }
    }

    pub fn reset_command_pool(&self, cmd_pool: vk::CommandPool, release_resources: bool) -> Result<(), vk::Result> {
        let flags = if release_resources {
            vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandPoolResetFlags::empty()
        };

        unsafe { self.device.as_ref().reset_command_pool(cmd_pool, flags) }
    }

    /// Hands every command buffer this allocator gave out back to its pool. Only call it once the
    /// GPU is done with them: nothing here waits.
    pub fn free_command_buffers(&mut self) {
        if self.cmd_pool.is_null() || self.cmd_buffers.is_empty() {
            return;
        }

        unsafe {
            self.device
                .as_ref()
                .free_command_buffers(self.cmd_pool, &self.cmd_buffers)
        };
        self.cmd_buffers.clear();
    }

    fn ensure_cmd_pool(&mut self, queue_family: u32) -> Result<(), vk::Result> {
        if !self.cmd_pool.is_null() {
            return Ok(());
        }

        self.cmd_pool = self.allocate_command_pool(queue_family)?;

        Ok(())
    }

    fn take_allocation_id(&mut self) -> Result<u64, vk::Result> {
        let id = self.next_allocation_id;
        self.next_allocation_id = id.checked_add(1).ok_or(vk::Result::ERROR_TOO_MANY_OBJECTS)?;
        Ok(id)
    }

    fn allocate_raw_buffer(&mut self, info: &BufferInfo) -> Result<Buffer, vk::Result> {
        let device = unsafe { self.device.as_ref() };

        let create_info = vk::BufferCreateInfo::default().size(info.size).usage(info.usage);
        let create_info = if self.queue_families.len() > 1 {
            create_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&self.queue_families)
        } else {
            create_info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        let handle = unsafe { device.create_buffer(&create_info, None) }?;

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let allocation = self.memory.borrow_mut().allocate(&AllocationCreateDesc {
            name: &info.name,
            requirements,
            location: info.location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });

        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { device.destroy_buffer(handle, None) };
                return Err(map_allocation_error(err));
            },
        };

        if let Err(err) = unsafe { device.bind_buffer_memory(handle, allocation.memory(), allocation.offset()) } {
            unsafe { device.destroy_buffer(handle, None) };
            let _ = self.memory.borrow_mut().free(allocation);
            return Err(err);
        }

        let device_address = if info.usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            let address_info = vk::BufferDeviceAddressInfo::default().buffer(handle);
            unsafe { device.get_buffer_device_address(&address_info) }
        } else {
            0
        };

        let mapped = allocation.mapped_ptr().map(NonNull::cast::<u8>);
        self.allocations.insert(handle, allocation);

        Ok(Buffer::new(handle, info.size, device_address, mapped))
    }

    fn deallocate_raw_buffer(&mut self, buffer: Buffer) {
        let Some(allocation) = self.allocations.remove(&buffer.handle()) else {
            tracing::error!(handle = ?buffer.handle(), "buffer was not allocated by this allocator");
            return;
        };

        unsafe { self.device.as_ref().destroy_buffer(buffer.handle(), None) };
        if let Err(err) = self.memory.borrow_mut().free(allocation) {
            tracing::error!(%err, "failed to free a buffer allocation");
        }
    }
}

impl Allocator for PersistentAllocator {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let create_info = vk::SemaphoreCreateInfo::default();

        unsafe { self.device.as_ref().create_semaphore(&create_info, None) }
    }

    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        unsafe { self.device.as_ref().create_semaphore(&create_info, None) }
    }

    fn deallocate_semaphore(&self, semaphore: vk::Semaphore) {
        unsafe { self.device.as_ref().destroy_semaphore(semaphore, None) };
    }

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
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .format(format)
            .view_type(view_type)
            .subresource_range(subresource_range);

        unsafe { self.device.as_ref().create_image_view(&create_info, None) }
    }

    fn deallocate_image_view(&mut self, image_view: vk::ImageView) {
        unsafe {
            self.device.as_ref().destroy_image_view(image_view, None);
        }
    }

    fn allocate_sampler(&mut self, info: &SamplerInfo) -> Result<vk::Sampler, vk::Result> {
        let create_info = vk::SamplerCreateInfo::from(info);

        unsafe { self.device.as_ref().create_sampler(&create_info, None) }
    }

    fn deallocate_sampler(&mut self, sampler: vk::Sampler) {
        unsafe { self.device.as_ref().destroy_sampler(sampler, None) };
    }

    fn allocate_buffer(&mut self, info: &BufferInfo) -> Result<Buffer, vk::Result> {
        if info.size == 0 {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let alignment = allocation_alignment(info, &self.limits).ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let id = self.take_allocation_id()?;
        if info.size >= BUFFER_BLOCK_SIZE {
            let raw = self.allocate_raw_buffer(info)?;
            let buffer = raw.slice(0, info.size, id).expect("dedicated buffer matches its size");
            self.buffer_allocations.insert(
                id,
                BufferAllocation {
                    value: buffer,
                    kind: BufferAllocationKind::Dedicated,
                },
            );
            return Ok(buffer);
        }

        let key = BufferPoolKey {
            location: info.location,
            usage: info.usage.as_raw(),
        };
        let existing = self.buffer_blocks.get_mut(&key).and_then(|blocks| {
            blocks.iter_mut().find_map(|block| {
                block
                    .ranges
                    .allocate(info.size, alignment)
                    .map(|offset| (block.buffer, offset))
            })
        });
        let (backing, offset) = match existing {
            Some(pair) => pair,
            None => {
                let raw = self.allocate_raw_buffer(&BufferInfo {
                    size: BUFFER_BLOCK_SIZE,
                    usage: info.usage,
                    location: info.location,
                    alignment: 1,
                    name: format!("buffer pool {:?}", key),
                })?;
                let mut ranges = Ranges::new(BUFFER_BLOCK_SIZE);
                let offset = ranges
                    .allocate(info.size, alignment)
                    .expect("a sub-block allocation must fit an empty block");
                self.buffer_blocks
                    .entry(key)
                    .or_default()
                    .push(BufferBlock { buffer: raw, ranges });
                (raw, offset)
            },
        };
        let buffer = backing
            .slice(offset, info.size, id)
            .expect("suballocation fits the backing buffer");
        self.buffer_allocations.insert(
            id,
            BufferAllocation {
                value: buffer,
                kind: BufferAllocationKind::Pooled {
                    key,
                    backing: backing.handle(),
                },
            },
        );
        Ok(buffer)
    }

    fn deallocate_buffer(&mut self, buffer: Buffer) {
        let id = buffer.allocation_id();
        let Some(record) = self.buffer_allocations.get(&id) else {
            tracing::error!(
                ?buffer,
                "buffer was not allocated by this allocator or was already freed"
            );
            return;
        };
        if record.value != buffer {
            tracing::error!(?buffer, "buffer does not match its live allocation");
            return;
        }
        let record = self.buffer_allocations.remove(&id).unwrap();
        match record.kind {
            BufferAllocationKind::Dedicated => self.deallocate_raw_buffer(record.value),
            BufferAllocationKind::Pooled { key, backing } => {
                let empty_block = {
                    let blocks = self.buffer_blocks.get_mut(&key).expect("live pool key");
                    let at = blocks
                        .iter()
                        .position(|block| block.buffer.handle() == backing)
                        .expect("live backing buffer");
                    blocks[at].ranges.free(record.value.offset(), record.value.size());
                    blocks[at].ranges.is_empty().then(|| blocks.swap_remove(at).buffer)
                };
                if let Some(backing) = empty_block {
                    self.deallocate_raw_buffer(backing);
                }
                if self.buffer_blocks.get(&key).is_some_and(Vec::is_empty) {
                    self.buffer_blocks.remove(&key);
                }
            },
        }
    }

    fn allocate_image(&mut self, info: &ImageInfo) -> Result<Image, vk::Result> {
        let device = unsafe { self.device.as_ref() };

        let create_info = vk::ImageCreateInfo::default()
            .image_type(info.image_type)
            .format(info.format)
            .extent(info.extent)
            .mip_levels(info.mip_levels)
            .array_layers(info.array_layers)
            .samples(info.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let handle = unsafe { device.create_image(&create_info, None) }?;

        let requirements = unsafe { device.get_image_memory_requirements(handle) };
        let allocation = self.memory.borrow_mut().allocate(&AllocationCreateDesc {
            name: &info.name,
            requirements,
            location: info.location,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });

        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { device.destroy_image(handle, None) };
                return Err(map_allocation_error(err));
            },
        };

        if let Err(err) = unsafe { device.bind_image_memory(handle, allocation.memory(), allocation.offset()) } {
            unsafe { device.destroy_image(handle, None) };
            let _ = self.memory.borrow_mut().free(allocation);
            return Err(err);
        }

        self.image_allocations.insert(handle, allocation);

        Ok(Image::new(handle, info))
    }

    fn deallocate_image(&mut self, image: Image) {
        let Some(allocation) = self.image_allocations.remove(&image.handle()) else {
            tracing::error!(handle = ?image.handle(), "image was not allocated by this allocator");
            return;
        };

        unsafe { self.device.as_ref().destroy_image(image.handle(), None) };
        if let Err(err) = self.memory.borrow_mut().free(allocation) {
            tracing::error!(%err, "failed to free an image allocation");
        }
    }
}

impl Drop for PersistentAllocator {
    fn drop(&mut self) {
        let device = unsafe { self.device.as_ref() };

        for (handle, allocation) in self.allocations.drain() {
            unsafe { device.destroy_buffer(handle, None) };
            if let Err(err) = self.memory.borrow_mut().free(allocation) {
                tracing::error!(%err, "failed to free a buffer allocation during teardown");
            }
        }

        for (handle, allocation) in self.image_allocations.drain() {
            unsafe { device.destroy_image(handle, None) };
            if let Err(err) = self.memory.borrow_mut().free(allocation) {
                tracing::error!(%err, "failed to free an image allocation during teardown");
            }
        }

        if !self.cmd_pool.is_null() {
            unsafe { device.destroy_command_pool(self.cmd_pool, None) };
        }
    }
}

impl std::fmt::Debug for PersistentAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentAllocator").finish_non_exhaustive()
    }
}
