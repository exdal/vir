use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    cell::Cell,
    ptr::NonNull,
};

const STACK_SIZE: usize = 1024 * 1024; // 1 MiB
const STACK_ALIGN: usize = 16;

struct ThreadStack {
    data: NonNull<u8>,
    ptr: Cell<usize>,
}

impl ThreadStack {
    fn layout() -> Layout { Layout::from_size_align(STACK_SIZE, STACK_ALIGN).unwrap() }

    fn new() -> Self {
        let layout = Self::layout();
        let data = unsafe { alloc_zeroed(layout) };
        Self {
            data: NonNull::new(data).unwrap_or_else(|| handle_alloc_error(layout)),
            ptr: Cell::new(0),
        }
    }

    fn alloc(&self, size: usize, align: usize) -> *mut u8 {
        assert!(align <= STACK_ALIGN, "alignment {align} exceeds stack alignment");
        let aligned = (self.ptr.get() + align - 1) & !(align - 1);
        let end = aligned.checked_add(size).expect("stack allocation overflowed");
        assert!(end <= STACK_SIZE, "thread stack exhausted");
        self.ptr.set(end);
        unsafe { self.data.as_ptr().add(aligned) }
    }
}

impl Drop for ThreadStack {
    fn drop(&mut self) { unsafe { dealloc(self.data.as_ptr(), Self::layout()) } }
}

thread_local! {
    static STACK: ThreadStack = ThreadStack::new();
}

pub struct ScopedStack {
    saved: usize,
}

impl ScopedStack {
    pub fn new() -> Self {
        let saved = STACK.with(|s| s.ptr.get());
        Self { saved }
    }

    pub fn alloc<T>(&self) -> *mut T { STACK.with(|s| s.alloc(size_of::<T>(), align_of::<T>()) as *mut T) }

    #[expect(clippy::mut_from_ref)]
    pub fn alloc_slice<T>(&self, count: usize) -> &mut [T] {
        let size = size_of::<T>().checked_mul(count).expect("stack allocation overflowed");
        let ptr = STACK.with(|s| s.alloc(size, align_of::<T>()) as *mut T);
        unsafe { std::slice::from_raw_parts_mut(ptr, count) }
    }

    pub fn concat_slices<T: Copy>(&self, slices: &[&[T]]) -> &mut [T] {
        let total = slices.iter().map(|s| s.len()).sum();
        let out = self.alloc_slice(total);
        let mut offset = 0;
        for s in slices {
            out[offset..offset + s.len()].copy_from_slice(s);
            offset += s.len();
        }
        out
    }
}

impl Default for ScopedStack {
    fn default() -> Self { Self::new() }
}

impl Drop for ScopedStack {
    fn drop(&mut self) { STACK.with(|s| s.ptr.set(self.saved)); }
}
