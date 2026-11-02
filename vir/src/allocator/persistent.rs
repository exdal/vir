use std::{collections::HashMap, ptr::NonNull};

use ash::vk::{self, Handle};
use gpu_allocator::{
    AllocationError,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use super::{Allocator, MemoryAllocator, buffer_pool::Ranges, map_allocation_error};
use crate::{Buffer, BufferInfo, CommandBuffer, Image, ImageInfo, MemoryLocation, SamplerInfo};

const BUFFER_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
struct HeapFallback {
    full_heap: u32,
    memory_type_bits: u32,
    location: MemoryLocation,
}

fn preferred_flags(location: MemoryLocation) -> vk::MemoryPropertyFlags {
    match location {
        MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryLocation::CpuToGpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL
        },
        MemoryLocation::GpuToCpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
        },
        _ => vk::MemoryPropertyFlags::empty(),
    }
}

fn required_flags(location: MemoryLocation) -> vk::MemoryPropertyFlags {
    match location {
        MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryLocation::CpuToGpu | MemoryLocation::GpuToCpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        },
        _ => vk::MemoryPropertyFlags::empty(),
    }
}

fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties, memory_type_bits: u32, flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|&index| {
        memory_type_bits & (1 << index) != 0 && properties.memory_types[index as usize].property_flags.contains(flags)
    })
}

fn pick_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties, memory_type_bits: u32, location: MemoryLocation,
) -> Option<u32> {
    find_memory_type(properties, memory_type_bits, preferred_flags(location))
        .or_else(|| find_memory_type(properties, memory_type_bits, required_flags(location)))
}

fn heap_fallback(
    properties: &vk::PhysicalDeviceMemoryProperties, memory_type_bits: u32, location: MemoryLocation,
) -> Option<HeapFallback> {
    let full_heap =
        properties.memory_types[pick_memory_type(properties, memory_type_bits, location)? as usize].heap_index;

    let remaining = (0..properties.memory_type_count)
        .filter(|&index| properties.memory_types[index as usize].heap_index != full_heap)
        .fold(0, |bits, index| bits | (1 << index))
        & memory_type_bits;
    if remaining == 0 {
        return None;
    }

    let is_location_satisfied = find_memory_type(properties, remaining, required_flags(location)).is_some();
    Some(HeapFallback {
        full_heap,
        memory_type_bits: remaining,
        location: if is_location_satisfied {
            location
        } else {
            MemoryLocation::Unknown
        },
    })
}

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
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    warned_full_heaps: u32,
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
        memory_properties: vk::PhysicalDeviceMemoryProperties, mut queue_families: Vec<u32>,
    ) -> Self {
        queue_families.sort_unstable();
        queue_families.dedup();
        PersistentAllocator {
            device,
            memory,
            limits,
            memory_properties,
            warned_full_heaps: 0,
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

    /// The memory a buffer from this allocator was bound to, pooled buffers report their backing block.
    pub fn memory_properties(&self, buffer: &Buffer) -> Option<vk::MemoryPropertyFlags> {
        self.allocations
            .get(&buffer.handle())
            .map(|allocation| allocation.memory_properties())
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

    fn allocate_memory(
        &mut self, name: &str, mut requirements: vk::MemoryRequirements, mut location: MemoryLocation, is_linear: bool,
    ) -> Result<Allocation, AllocationError> {
        loop {
            let result = self.memory.borrow_mut().allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: is_linear,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });

            if !matches!(result, Err(AllocationError::OutOfMemory)) {
                return result;
            }

            let Some(fallback) = heap_fallback(&self.memory_properties, requirements.memory_type_bits, location) else {
                return result;
            };

            if self.warned_full_heaps & (1 << fallback.full_heap) == 0 {
                self.warned_full_heaps |= 1 << fallback.full_heap;
                tracing::warn!(
                    name,
                    size = requirements.size,
                    heap = fallback.full_heap,
                    "memory heap is full, falling back to another heap"
                );
            }

            requirements.memory_type_bits = fallback.memory_type_bits;
            location = fallback.location;
        }
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
        let allocation = self.allocate_memory(&info.name, requirements, info.location, true);

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
        let allocation = self.allocate_memory(&info.name, requirements, info.location, false);

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

#[cfg(test)]
mod tests {
    use vk::MemoryPropertyFlags as Flags;

    use super::*;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    const DL: Flags = Flags::DEVICE_LOCAL;
    const HV_HC: Flags = Flags::from_raw(Flags::HOST_VISIBLE.as_raw() | Flags::HOST_COHERENT.as_raw());
    const HV_HC_CACHED: Flags = Flags::from_raw(HV_HC.as_raw() | Flags::HOST_CACHED.as_raw());
    const DL_HV_HC: Flags = Flags::from_raw(DL.as_raw() | HV_HC.as_raw());

    fn properties(heaps: &[(u64, bool)], types: &[(Flags, u32)]) -> vk::PhysicalDeviceMemoryProperties {
        let mut properties = vk::PhysicalDeviceMemoryProperties::default();
        properties.memory_heap_count = heaps.len() as u32;
        for (slot, &(size, is_device_local)) in properties.memory_heaps.iter_mut().zip(heaps) {
            slot.size = size;
            slot.flags = if is_device_local {
                vk::MemoryHeapFlags::DEVICE_LOCAL
            } else {
                vk::MemoryHeapFlags::empty()
            };
        }
        properties.memory_type_count = types.len() as u32;
        for (slot, &(property_flags, heap_index)) in properties.memory_types.iter_mut().zip(types) {
            slot.property_flags = property_flags;
            slot.heap_index = heap_index;
        }
        properties
    }

    fn apu() -> vk::PhysicalDeviceMemoryProperties {
        properties(
            &[(256 * MIB, true), (8 * GIB, false)],
            &[(DL, 0), (DL_HV_HC, 0), (HV_HC, 1), (HV_HC_CACHED, 1)],
        )
    }

    fn discrete() -> vk::PhysicalDeviceMemoryProperties {
        properties(
            &[(8 * GIB, true), (16 * GIB, false), (256 * MIB, true)],
            &[(DL, 0), (HV_HC, 1), (HV_HC_CACHED, 1), (DL_HV_HC, 2)],
        )
    }

    #[test]
    fn apu_gpu_only_falls_to_system_heap() {
        let properties = apu();
        let first = heap_fallback(&properties, 0b1111, MemoryLocation::GpuOnly).unwrap();
        assert_eq!(first.full_heap, 0);
        assert_eq!(first.memory_type_bits, 0b1100);
        assert_eq!(first.location, MemoryLocation::Unknown);
        assert_eq!(
            pick_memory_type(&properties, first.memory_type_bits, first.location),
            Some(2)
        );
    }

    #[test]
    fn discrete_gpu_only_falls_to_bar_then_system() {
        let properties = discrete();
        let first = heap_fallback(&properties, 0b1111, MemoryLocation::GpuOnly).unwrap();
        assert_eq!(first.full_heap, 0);
        assert_eq!(first.memory_type_bits, 0b1110);
        assert_eq!(first.location, MemoryLocation::GpuOnly);
        assert_eq!(
            pick_memory_type(&properties, first.memory_type_bits, first.location),
            Some(3)
        );

        let second = heap_fallback(&properties, first.memory_type_bits, first.location).unwrap();
        assert_eq!(second.full_heap, 2);
        assert_eq!(second.memory_type_bits, 0b0110);
        assert_eq!(second.location, MemoryLocation::Unknown);
        assert_eq!(
            pick_memory_type(&properties, second.memory_type_bits, second.location),
            Some(1)
        );
    }

    #[test]
    fn fallback_stops_at_the_last_heap() {
        let properties = discrete();
        let mut bits = 0b1111;
        let mut location = MemoryLocation::GpuOnly;
        let mut steps = 0;
        while let Some(fallback) = heap_fallback(&properties, bits, location) {
            bits = fallback.memory_type_bits;
            location = fallback.location;
            steps += 1;
        }

        assert_eq!(steps, 2);
        assert_ne!(bits, 0);
        assert_eq!(heap_fallback(&properties, bits, location), None);
    }

    #[test]
    fn masking_respects_requirement_bits() {
        let properties = apu();
        assert_eq!(heap_fallback(&properties, 0b0011, MemoryLocation::GpuOnly), None);
    }
}
