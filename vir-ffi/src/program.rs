use ash::vk;
use vir::{Program, ValueId};

use crate::{
    error::{Failure, guard_result, guard_void},
    types::bytes,
};

pub struct VirProgram(pub Program);

fn program<'a>(p: *mut VirProgram) -> Result<&'a mut Program, Failure> {
    unsafe { p.as_mut() }
        .map(|program| &mut program.0)
        .ok_or_else(|| Failure::new("program is null"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_program_destroy(p: *mut VirProgram) {
    guard_void(|| {
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_program_set_bool(p: *mut VirProgram, variable: u32, value: bool) -> vk::Result {
    guard_result(|| {
        program(p)?.set(ValueId(variable), value);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_program_set_bytes(
    p: *mut VirProgram, variable: u32, data: *const u8, len: usize,
) -> vk::Result {
    guard_result(|| {
        let data = unsafe { bytes(data, len) };
        program(p)?.set(ValueId(variable), data);
        Ok(())
    })
}
