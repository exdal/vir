use std::{
    cell::RefCell,
    panic::{AssertUnwindSafe, catch_unwind},
};

use ash::vk;

// uhhhhhhhhh
thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

pub(crate) struct Failure {
    pub code: vk::Result,
    pub message: String,
}

impl Failure {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            code: vk::Result::ERROR_UNKNOWN,
            message: message.into(),
        }
    }
}

impl From<vk::Result> for Failure {
    fn from(code: vk::Result) -> Self {
        Self {
            code,
            message: code.to_string(),
        }
    }
}

fn set_last_error(message: String) {
    LAST_ERROR.with(|last| *last.borrow_mut() = message);
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_owned(),
            Err(_) => "panic with a non-string payload".to_owned(),
        },
    }
}

fn run<T>(f: impl FnOnce() -> Result<T, Failure>) -> Result<T, vk::Result> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(failure)) => {
            set_last_error(failure.message);
            Err(failure.code)
        },
        Err(payload) => {
            set_last_error(panic_message(payload));
            Err(vk::Result::ERROR_UNKNOWN)
        },
    }
}

pub(crate) fn guard_result(f: impl FnOnce() -> Result<(), Failure>) -> vk::Result {
    run(f).map_or_else(|code| code, |()| vk::Result::SUCCESS)
}

pub(crate) fn guard_id(f: impl FnOnce() -> Result<u32, Failure>) -> u32 {
    run(f).unwrap_or(u32::MAX)
}

pub(crate) fn guard_u64(f: impl FnOnce() -> Result<u64, Failure>) -> u64 {
    run(f).unwrap_or(0)
}

pub(crate) fn guard_void(f: impl FnOnce() -> Result<(), Failure>) {
    let _ = run(f);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_last_error(buf: *mut u8, len: usize) -> usize {
    LAST_ERROR.with(|last| {
        let message = last.borrow();
        let bytes = message.as_bytes();
        if buf.is_null() || len == 0 {
            return bytes.len();
        }

        let copied = bytes.len().min(len);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, copied) };
        if copied < len {
            unsafe { *buf.add(copied) = 0 };
        }

        bytes.len()
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_clear_last_error() {
    set_last_error(String::new());
}
