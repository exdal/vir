use std::ptr;

use ash::vk;
use vir_ffi::*;

fn last_error() -> String {
    let len = unsafe { vir_last_error(ptr::null_mut(), 0) };
    let mut buf = vec![0u8; len + 1];
    assert_eq!(unsafe { vir_last_error(buf.as_mut_ptr(), buf.len()) }, len);
    String::from_utf8(buf[..len].to_vec()).unwrap()
}

#[test]
fn enable_timestamps_rejects_null_allocator() {
    let mut out = 0;
    let result = unsafe { vir_super_frame_allocator_enable_timestamps(ptr::null_mut(), ptr::null_mut(), &mut out) };
    assert_ne!(result, vk::Result::SUCCESS);
    assert_eq!(last_error(), "super frame allocator is null");
}

#[test]
fn completed_timings_rejects_null_frame() {
    let mut count = 0;
    let result = unsafe { vir_frame_allocator_completed_timings(ptr::null_mut(), ptr::null_mut(), 0, &mut count) };
    assert_ne!(result, vk::Result::SUCCESS);
    assert_eq!(last_error(), "frame allocator is null");
}

#[test]
fn add_timeline_wait_null_frame_sets_error() {
    unsafe { vir_frame_allocator_add_timeline_wait(ptr::null_mut(), 1, 2) };
    assert_eq!(last_error(), "frame allocator is null");
}
