use std::cell::Cell;

const STACK_SIZE: usize = 1 * 1024 * 1024; // 1 MiB

struct ThreadStack {
    data: Box<[u8]>,
    ptr: Cell<usize>,
}

impl ThreadStack {
    fn new() -> Self {
        Self {
            data: vec![0u8; STACK_SIZE].into_boxed_slice(),
            ptr: Cell::new(0),
        }
    }

    fn alloc(&self, size: usize, align: usize) -> *mut u8 {
        let offset = self.ptr.get();
        let aligned = (offset + align - 1) & !(align - 1);
        self.ptr.set(aligned + size);
        unsafe { self.data.as_ptr().add(aligned) as *mut u8 }
    }
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

    pub fn alloc_slice<T>(&self, count: usize) -> &mut [T] {
        let ptr = STACK.with(|s| s.alloc(size_of::<T>() * count, align_of::<T>()) as *mut T);
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

impl Drop for ScopedStack {
    fn drop(&mut self) { STACK.with(|s| s.ptr.set(self.saved)); }
}
