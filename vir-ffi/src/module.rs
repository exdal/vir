use std::ffi::c_void;

use ash::vk;
use vir::{ClearValue, Module, PipelineId, ValueId};

use crate::{
    context::out_ptr,
    error::{Failure, guard_id, guard_result, guard_void},
    graph::{VirRenderGraph, graph},
    program::VirProgram,
    swapchain::VirSwapChain,
    types::{VirAttachment, VirRasterizationState, VirRect2D, VirViewport, blend_preset, bytes, dynamic_state, text},
};

#[repr(transparent)]
pub struct VirModule(pub Module);

pub type VirConditionFn =
    unsafe extern "C" fn(module: *mut VirModule, user_data: *mut c_void, out: *mut u32, capacity: usize) -> usize;

pub(crate) fn module<'a>(m: *mut VirModule) -> Result<&'a mut Module, Failure> {
    unsafe { m.as_mut() }
        .map(|module| &mut module.0)
        .ok_or_else(|| Failure::new("module is null"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_create() -> *mut VirModule {
    std::panic::catch_unwind(|| Box::into_raw(Box::new(VirModule(Module::default())))).unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_destroy(m: *mut VirModule) {
    guard_void(|| {
        if !m.is_null() {
            drop(unsafe { Box::from_raw(m) });
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_declare_bool_var(
    m: *mut VirModule, name: *const u8, name_len: usize, default: bool,
) -> u32 {
    guard_id(|| {
        let name = unsafe { text(name, name_len) }?;
        Ok(module(m)?.declare_bool_var(name, default).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_declare_bytes_var(
    m: *mut VirModule, name: *const u8, name_len: usize, size: u32,
) -> u32 {
    guard_id(|| {
        let name = unsafe { text(name, name_len) }?;
        Ok(module(m)?.declare_bytes_var(name, size).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_clear(m: *mut VirModule, attachment: u32, color: ClearValue) -> u32 {
    guard_id(|| Ok(module(m)?.clear(ValueId(attachment), color).0))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_begin_rendering(
    m: *mut VirModule, attachments: *const VirAttachment, count: usize,
) {
    guard_void(|| {
        let module = module(m)?;
        let attachments = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(attachments, count) }
        };
        module.begin_rendering(attachments.iter().map(|&attachment| attachment.into()));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_with_name(m: *mut VirModule, name: *const u8, name_len: usize) {
    guard_void(|| {
        let name = unsafe { text(name, name_len) }?;
        module(m)?.with_name(name);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_graphics_pipeline(m: *mut VirModule, pipeline: u32) {
    guard_void(|| {
        module(m)?.bind_graphics_pipeline(PipelineId::from_raw(pipeline));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_set_viewport(m: *mut VirModule, index: u32, viewport: VirViewport) {
    guard_void(|| {
        module(m)?.set_viewport(index, viewport.to_vir()?);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_set_scissor(m: *mut VirModule, index: u32, rect: VirRect2D) {
    guard_void(|| {
        module(m)?.set_scissor(index, rect.to_vir()?);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_broadcast_color_blend(m: *mut VirModule, preset: u32) {
    guard_void(|| {
        let preset = blend_preset(preset)?;
        module(m)?.broadcast_color_blend(preset);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_set_rasterization(m: *mut VirModule, state: VirRasterizationState) {
    guard_void(|| {
        module(m)?.set_rasterization(state.into());
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_set_dynamic_state(m: *mut VirModule, flags: u32) {
    guard_void(|| {
        module(m)?.set_dynamic_state(dynamic_state(flags));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_push_constant_bytes(m: *mut VirModule, offset: u32, data: *const u8, len: usize) {
    guard_void(|| {
        let data = unsafe { bytes(data, len) };
        module(m)?.push_constant_bytes(offset, data);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_push_constants_from_at(m: *mut VirModule, offset: u32, variable: u32) {
    guard_void(|| {
        module(m)?.push_constants_from_at(offset, ValueId(variable));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw(m: *mut VirModule, vertex_count: u32, instance_count: u32) {
    guard_void(|| {
        module(m)?.draw(vertex_count, instance_count);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_end_rendering(m: *mut VirModule, out: *mut u32, count: usize) -> vk::Result {
    guard_result(|| {
        let module = module(m)?;
        let mut ids = vec![ValueId::INVALID; count];
        module.end_rendering_into(&mut ids);
        if count != 0 {
            let out = unsafe { std::slice::from_raw_parts_mut(out, count) };
            for (slot, id) in out.iter_mut().zip(&ids) {
                *slot = id.0;
            }
        }

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_present(m: *mut VirModule, attachment: u32) -> u32 {
    guard_id(|| Ok(module(m)?.present(ValueId(attachment)).0))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_set_condition(
    m: *mut VirModule, condition: u32, then: VirConditionFn, otherwise: VirConditionFn, user_data: *mut c_void,
    out: *mut u32, capacity: usize, out_count: *mut usize,
) -> vk::Result {
    guard_result(|| {
        if capacity == 0 || out.is_null() || out_count.is_null() {
            return Err(Failure::new("a condition must produce at least one value"));
        }

        let arm = |callback: VirConditionFn, module: &mut Module| {
            let mut ids = vec![u32::MAX; capacity];
            let written = unsafe {
                callback(
                    module as *mut Module as *mut VirModule,
                    user_data,
                    ids.as_mut_ptr(),
                    capacity,
                )
            };
            assert!(written != 0, "a condition arm produced no values");
            assert!(
                written <= capacity,
                "a condition arm produced {written} values, more than the capacity of {capacity}"
            );
            ids.truncate(written);
            assert!(
                ids.iter().all(|&id| id != u32::MAX),
                "a condition arm left a result unset"
            );
            ids.into_iter().map(ValueId).collect::<Vec<_>>()
        };

        let merged = module(m)?.set_condition(
            ValueId(condition),
            |module| arm(then, module),
            |module| arm(otherwise, module),
        );
        let out = unsafe { std::slice::from_raw_parts_mut(out, capacity) };
        for (slot, id) in out.iter_mut().zip(&merged) {
            *slot = id.0;
        }

        unsafe { *out_count = merged.len() };

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_acquire_next_image(m: *mut VirModule, s: *const VirSwapChain) -> u32 {
    guard_id(|| {
        let swapchain = unsafe { s.as_ref() }.ok_or_else(|| Failure::new("swapchain is null"))?;
        Ok(module(m)?.acquire_next_image(&swapchain.0).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_compile(
    m: *mut VirModule, g: *mut VirRenderGraph, root: u32, out: *mut *mut VirProgram,
) -> vk::Result {
    guard_result(|| {
        let graph = graph(g)?;
        let out = out_ptr(out)?;
        let program = module(m)?.compile(&*graph, ValueId(root))?;
        *out = Box::into_raw(Box::new(VirProgram(program)));
        Ok(())
    })
}
