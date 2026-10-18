use ash::vk;
use vir::{FrameAllocator, PersistentAllocator, SuperFrameAllocator};

use crate::{
    context::{VirContext, context, out_ptr},
    error::{Failure, guard_result, guard_void},
};

pub struct VirPersistentAllocator(pub PersistentAllocator);

pub struct VirSuperFrameAllocator(pub SuperFrameAllocator);

#[repr(transparent)]
pub struct VirFrameAllocator(pub FrameAllocator);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_create(
    c: *mut VirContext, out: *mut *mut VirPersistentAllocator,
) -> vk::Result {
    guard_result(|| {
        let allocator = context(c)?.ctx.create_persistent_allocator();
        *out_ptr(out)? = Box::into_raw(Box::new(VirPersistentAllocator(allocator)));
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_persistent_allocator_destroy(a: *mut VirPersistentAllocator) {
    guard_void(|| {
        if !a.is_null() {
            drop(unsafe { Box::from_raw(a) });
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_super_frame_allocator_create(
    c: *mut VirContext, frame_count: u32, out: *mut *mut VirSuperFrameAllocator,
) -> vk::Result {
    guard_result(|| {
        if frame_count == 0 {
            return Err(Failure::new("frame_count must be at least 1"));
        }

        let allocator = context(c)?.ctx.create_super_frame_allocator(frame_count as usize);
        *out_ptr(out)? = Box::into_raw(Box::new(VirSuperFrameAllocator(allocator)));
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_super_frame_allocator_destroy(a: *mut VirSuperFrameAllocator) {
    guard_void(|| {
        if !a.is_null() {
            drop(unsafe { Box::from_raw(a) });
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_super_frame_allocator_next_frame(
    a: *mut VirSuperFrameAllocator, out: *mut *mut VirFrameAllocator,
) -> vk::Result {
    guard_result(|| {
        let allocator = unsafe { a.as_mut() }.ok_or_else(|| Failure::new("super frame allocator is null"))?;
        let out = out_ptr(out)?;
        let frame = allocator.0.get_next_frame()?;
        *out = frame as *mut FrameAllocator as *mut VirFrameAllocator;
        Ok(())
    })
}
