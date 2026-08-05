use std::{
    hash::{Hash, Hasher},
    ptr::NonNull,
};

use ash::vk::{self, Handle};
pub use gpu_allocator::MemoryLocation;

/// What an allocator needs to hand back a [`Buffer`].
#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub location: MemoryLocation,
    pub name: String,
}

impl BufferInfo {
    pub fn new(size: u64, usage: vk::BufferUsageFlags, location: MemoryLocation) -> Self {
        Self {
            size,
            usage,
            location,
            name: String::new(),
        }
    }

    pub fn vertex(size: u64) -> Self {
        Self::new(size, vk::BufferUsageFlags::VERTEX_BUFFER, MemoryLocation::CpuToGpu)
    }

    pub fn with_usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage |= usage;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Buffer {
    handle: vk::Buffer,
    size: u64,
    device_address: vk::DeviceAddress,
    mapped: Option<NonNull<u8>>,
}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            handle: vk::Buffer::null(),
            size: 0,
            device_address: 0,
            mapped: None,
        }
    }
}

impl Buffer {
    pub(crate) fn new(
        handle: vk::Buffer, size: u64, device_address: vk::DeviceAddress, mapped: Option<NonNull<u8>>,
    ) -> Self {
        Self {
            handle,
            size,
            device_address,
            mapped,
        }
    }

    pub fn handle(&self) -> vk::Buffer { self.handle }

    pub fn size(&self) -> u64 { self.size }

    pub fn device_address(&self) -> vk::DeviceAddress { self.device_address }

    pub fn is_null(&self) -> bool { self.handle.is_null() }

    pub fn is_mapped(&self) -> bool { self.mapped.is_some() }

    pub fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        let ptr = self.mapped?;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), self.size as usize) })
    }

    pub fn write<T: Copy>(&mut self, offset: u64, data: &[T]) -> Result<(), vk::Result> {
        let Some(ptr) = self.mapped else {
            tracing::error!("cannot write to a buffer that is not host visible");
            return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
        };

        let bytes = size_of_val(data) as u64;
        if offset.checked_add(bytes).is_none_or(|end| end > self.size) {
            tracing::error!(offset, bytes, size = self.size, "write runs past the end of the buffer");
            return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                ptr.as_ptr().add(offset as usize),
                bytes as usize,
            );
        }

        Ok(())
    }
}

impl From<&Buffer> for vk::Buffer {
    fn from(buffer: &Buffer) -> Self { buffer.handle }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool { self.handle == other.handle }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) { self.handle.hash(state); }
}
