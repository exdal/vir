use std::{
    hash::{Hash, Hasher},
    ptr::NonNull,
};

use ash::vk::{self, Handle};
pub use gpu_allocator::MemoryLocation;

#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub location: MemoryLocation,
    pub alignment: u64,
    pub name: String,
}

impl BufferInfo {
    pub fn new(size: u64, usage: vk::BufferUsageFlags, location: MemoryLocation) -> Self {
        Self {
            size,
            usage,
            location,
            alignment: 1,
            name: String::new(),
        }
    }

    pub fn vertex(size: u64) -> Self { Self::new(size, vk::BufferUsageFlags::VERTEX_BUFFER, MemoryLocation::CpuToGpu) }

    /// A host-visible buffer to stage a copy into device-local memory out of.
    pub fn staging(size: u64) -> Self { Self::new(size, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu) }

    pub fn with_usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage |= usage;
        self
    }

    pub fn with_size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }

    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment;
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
    offset: u64,
    size: u64,
    device_address: vk::DeviceAddress,
    mapped: Option<NonNull<u8>>,
    allocation_id: u64,
}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            handle: vk::Buffer::null(),
            offset: 0,
            size: 0,
            device_address: 0,
            mapped: None,
            allocation_id: 0,
        }
    }
}

impl Buffer {
    pub(crate) fn new(
        handle: vk::Buffer, size: u64, device_address: vk::DeviceAddress, mapped: Option<NonNull<u8>>,
    ) -> Self {
        Self {
            handle,
            offset: 0,
            size,
            device_address,
            mapped,
            allocation_id: 0,
        }
    }

    pub(crate) fn slice(self, offset: u64, size: u64, allocation_id: u64) -> Option<Self> {
        if offset.checked_add(size)? > self.size {
            return None;
        }
        let mapped = match self.mapped {
            Some(ptr) => Some(NonNull::new(unsafe {
                ptr.as_ptr().add(usize::try_from(offset).ok()?)
            })?),
            None => None,
        };
        Some(Self {
            handle: self.handle,
            offset: self.offset.checked_add(offset)?,
            size,
            device_address: if self.device_address == 0 {
                0
            } else {
                self.device_address.checked_add(offset)?
            },
            mapped,
            allocation_id,
        })
    }

    pub fn handle(&self) -> vk::Buffer { self.handle }

    /// Byte offset of this logical buffer within its Vulkan backing buffer.
    pub fn offset(&self) -> u64 { self.offset }

    pub fn size(&self) -> u64 { self.size }

    pub(crate) fn checked_offset(&self, relative: u64) -> Result<u64, vk::Result> {
        if relative > self.size {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        self.offset
            .checked_add(relative)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
    }

    pub fn device_address(&self) -> vk::DeviceAddress { self.device_address }

    pub fn is_null(&self) -> bool { self.handle.is_null() }

    pub fn is_mapped(&self) -> bool { self.mapped.is_some() }

    pub(crate) fn allocation_id(&self) -> u64 { self.allocation_id }

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
    fn eq(&self, other: &Self) -> bool {
        (self.handle, self.offset, self.size, self.allocation_id)
            == (other.handle, other.offset, other.size, other.allocation_id)
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) { (self.handle, self.offset, self.size, self.allocation_id).hash(state); }
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;

    #[test]
    fn slices_carry_base_offsets_and_limit_host_writes() {
        let mut bytes = [0_u8; 64];
        let mapped = NonNull::new(bytes.as_mut_ptr()).unwrap();
        let raw = Buffer::new(vk::Buffer::from_raw(7), 64, 1000, Some(mapped));
        let mut slice = raw.slice(16, 8, 1).unwrap();
        assert_eq!(slice.handle(), raw.handle());
        assert_eq!(slice.offset(), 16);
        assert_eq!(slice.size(), 8);
        assert_eq!(slice.device_address(), 1016);
        assert_eq!(slice.checked_offset(4), Ok(20));
        assert!(slice.checked_offset(9).is_err());
        slice.write(2, &[1_u8, 2, 3]).unwrap();
        assert_eq!(&bytes[18..21], &[1, 2, 3]);
        assert!(slice.write(7, &[1_u8, 2]).is_err());

        let other = raw.slice(24, 8, 2).unwrap();
        assert_ne!(slice, other);
        assert_ne!(slice, raw.slice(16, 8, 3).unwrap());
    }
}
