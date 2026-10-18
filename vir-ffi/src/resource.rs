use ash::{vk, vk::Handle};
use vir::{BufferInfo, Image, ImageInfo, MemoryLocation, SamplerInfo, allocator::Allocator};

use crate::{
    allocator::VirPersistentAllocator,
    context::out_ptr,
    error::{Failure, guard_result, guard_void},
    types::{bytes, text},
};

pub struct VirBuffer(pub vir::Buffer);

pub struct VirImage {
    pub image: Image,
    pub view: vk::ImageView,
}

#[repr(C)]
pub struct VirBufferInfo {
    pub size: u64,
    pub usage: u32,
    pub location: u32,
    pub alignment: u64,
    pub name: *const u8,
    pub name_len: usize,
}

#[repr(C)]
pub struct VirImageInfo {
    pub kind: u32,
    pub width: u32,
    pub height: u32,
    pub format: i32,
    pub usage: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: u32,
    pub name: *const u8,
    pub name_len: usize,
}

#[repr(C)]
pub struct VirSamplerInfo {
    pub mag_filter: i32,
    pub min_filter: i32,
    pub mipmap_mode: i32,
    pub address_mode_u: i32,
    pub address_mode_v: i32,
    pub address_mode_w: i32,
}

fn persistent<'a>(a: *mut VirPersistentAllocator) -> Result<&'a mut vir::PersistentAllocator, Failure> {
    unsafe { a.as_mut() }
        .map(|allocator| &mut allocator.0)
        .ok_or_else(|| Failure::new("persistent allocator is null"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_allocate_buffer(
    a: *mut VirPersistentAllocator, info: *const VirBufferInfo, out: *mut *mut VirBuffer,
) -> vk::Result {
    guard_result(|| {
        let allocator = persistent(a)?;
        let info = unsafe { info.as_ref() }.ok_or_else(|| Failure::new("buffer info is null"))?;
        let out = out_ptr(out)?;
        let location = match info.location {
            0 => MemoryLocation::Unknown,
            1 => MemoryLocation::GpuOnly,
            2 => MemoryLocation::CpuToGpu,
            3 => MemoryLocation::GpuToCpu,
            location => return Err(Failure::new(format!("unknown memory location {location}"))),
        };
        let name = unsafe { text(info.name, info.name_len) }?;

        let buffer = allocator.allocate_buffer(
            &BufferInfo::new(info.size, vk::BufferUsageFlags::from_raw(info.usage), location)
                .with_alignment(info.alignment.max(1))
                .with_name(name),
        )?;
        *out = Box::into_raw(Box::new(VirBuffer(buffer)));

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_deallocate_buffer(a: *mut VirPersistentAllocator, b: *mut VirBuffer) {
    guard_void(|| {
        let allocator = persistent(a)?;
        if !b.is_null() {
            allocator.deallocate_buffer(unsafe { Box::from_raw(b) }.0);
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_buffer_write(b: *mut VirBuffer, offset: u64, data: *const u8, len: usize) -> vk::Result {
    guard_result(|| {
        let buffer = unsafe { b.as_mut() }.ok_or_else(|| Failure::new("buffer is null"))?;
        buffer.0.write(offset, unsafe { bytes(data, len) })?;

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_allocate_image(
    a: *mut VirPersistentAllocator, info: *const VirImageInfo, out: *mut *mut VirImage,
) -> vk::Result {
    guard_result(|| {
        let allocator = persistent(a)?;
        let info = unsafe { info.as_ref() }.ok_or_else(|| Failure::new("image info is null"))?;
        let out = out_ptr(out)?;
        if info.mip_levels == 0 || info.array_layers == 0 || info.samples == 0 {
            return Err(Failure::new("mip_levels, array_layers and samples must be at least 1"));
        }

        let extent = vk::Extent2D {
            width: info.width,
            height: info.height,
        };
        let format = vk::Format::from_raw(info.format);
        let image_info = match info.kind {
            0 => ImageInfo::color_target(extent, format),
            1 => ImageInfo::depth_target(extent, format),
            2 => ImageInfo::texture(extent, format),
            kind => return Err(Failure::new(format!("unknown image kind {kind}"))),
        }
        .with_usage(vk::ImageUsageFlags::from_raw(info.usage))
        .with_mips(info.mip_levels)
        .with_layers(info.array_layers)
        .with_samples(vk::SampleCountFlags::from_raw(info.samples))
        .with_name(unsafe { text(info.name, info.name_len) }?);

        let is_array = info.array_layers > 1;
        let view_type = if is_array {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };

        let image = allocator.allocate_image(&image_info)?;
        let view = match allocator.allocate_image_view(image.handle(), format, view_type, image.subresource_range()) {
            Ok(view) => view,
            Err(err) => {
                allocator.deallocate_image(image);
                return Err(err.into());
            },
        };
        *out = Box::into_raw(Box::new(VirImage { image, view }));

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_deallocate_image(a: *mut VirPersistentAllocator, img: *mut VirImage) {
    guard_void(|| {
        let allocator = persistent(a)?;
        if !img.is_null() {
            let image = unsafe { Box::from_raw(img) };
            allocator.deallocate_image_view(image.view);
            allocator.deallocate_image(image.image);
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_allocate_sampler(
    a: *mut VirPersistentAllocator, info: *const VirSamplerInfo, out: *mut u64,
) -> vk::Result {
    guard_result(|| {
        let allocator = persistent(a)?;
        let info = unsafe { info.as_ref() }.ok_or_else(|| Failure::new("sampler info is null"))?;
        let out = out_ptr(out)?;

        let sampler = allocator.allocate_sampler(&SamplerInfo {
            mag_filter: vk::Filter::from_raw(info.mag_filter),
            min_filter: vk::Filter::from_raw(info.min_filter),
            mipmap_mode: vk::SamplerMipmapMode::from_raw(info.mipmap_mode),
            address_mode_u: vk::SamplerAddressMode::from_raw(info.address_mode_u),
            address_mode_v: vk::SamplerAddressMode::from_raw(info.address_mode_v),
            address_mode_w: vk::SamplerAddressMode::from_raw(info.address_mode_w),
        })?;
        *out = sampler.as_raw();

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_deallocate_sampler(a: *mut VirPersistentAllocator, sampler: u64) {
    guard_void(|| {
        persistent(a)?.deallocate_sampler(vk::Sampler::from_raw(sampler));
        Ok(())
    });
}
