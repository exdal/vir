use std::ffi::c_void;

use ash::{vk, vk::Handle};
use vir::{Access, BufferCopy, ClearValue, DomainFlag, ImageAttachment, Module, PipelineId, SpecValue, ValueId};

use crate::{
    context::out_ptr,
    error::{Failure, guard_id, guard_result, guard_void},
    graph::{VirRenderGraph, graph},
    program::VirProgram,
    resource::{VirBuffer, VirBufferInfo, VirImage, VirTransientImageInfo, buffer_info, transient_image_info},
    swapchain::VirSwapChain,
    types::{
        VirAttachment, VirBufferCopy, VirBufferImageCopy, VirDepthState, VirRasterizationState, VirRect2D, VirViewport,
        blend_preset, bytes, dynamic_state, slice, text,
    },
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_import_buffer(m: *mut VirModule, b: *const VirBuffer, access: u64) -> u32 {
    guard_id(|| {
        let buffer = unsafe { b.as_ref() }.ok_or_else(|| Failure::new("buffer is null"))?;
        Ok(module(m)?.import_buffer(&buffer.0, Access::from_bits_retain(access)).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_import_image(
    m: *mut VirModule, img: *const VirImage, layout: i32, access: u64,
) -> u32 {
    guard_id(|| {
        let image = unsafe { img.as_ref() }.ok_or_else(|| Failure::new("image is null"))?;
        let attachment =
            ImageAttachment::from_image(&image.image, vk::ImageLayout::from_raw(layout)).with_image_view(image.view);
        Ok(module(m)?
            .import_attachment(&attachment, Access::from_bits_retain(access))
            .0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_set_name(m: *mut VirModule, value: u32, name: *const u8, name_len: usize) -> u32 {
    guard_id(|| {
        let name = unsafe { text(name, name_len) }?;
        Ok(module(m)?.set_name(ValueId(value), name).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_copy_buffer_to_image(m: *mut VirModule, buffer: u32, image: u32) -> u32 {
    guard_id(|| Ok(module(m)?.copy_buffer_to_image(ValueId(buffer), ValueId(image)).0))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_export(m: *mut VirModule, value: u32, access: u64, domain: u32) -> u32 {
    guard_id(|| {
        let access = Access::from_bits_retain(access);
        Ok(module(m)?
            .export(ValueId(value), access, DomainFlag::from_bits_retain(domain))
            .0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_set_depth(m: *mut VirModule, depth: VirDepthState) {
    guard_void(|| {
        module(m)?.set_depth(depth.into());
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_texture(m: *mut VirModule, set: u32, binding: u32, image: u32, sampler: u64) {
    guard_void(|| {
        module(m)?.bind_texture(set, binding, ValueId(image), vk::Sampler::from_raw(sampler));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_vertex_buffer(m: *mut VirModule, binding: u32, buffer: u32) {
    guard_void(|| {
        module(m)?.bind_vertex_buffer(binding, ValueId(buffer));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_index_buffer(m: *mut VirModule, buffer: u32, index_type: i32) {
    guard_void(|| {
        module(m)?.bind_index_buffer(ValueId(buffer), vk::IndexType::from_raw(index_type));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw_indexed(m: *mut VirModule, index_count: u32, instance_count: u32) {
    guard_void(|| {
        module(m)?.draw_indexed(index_count, instance_count);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_compile_all(
    m: *mut VirModule, g: *mut VirRenderGraph, roots: *const u32, count: usize, out: *mut *mut VirProgram,
) -> vk::Result {
    guard_result(|| {
        let graph = graph(g)?;
        let out = out_ptr(out)?;
        let roots = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(roots, count) }
        };
        let roots = roots.iter().map(|&root| ValueId(root)).collect::<Vec<_>>();

        let program = module(m)?.compile_all(&*graph, &roots)?;
        *out = Box::into_raw(Box::new(VirProgram(program)));

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_begin_compute(m: *mut VirModule, attachments: *const VirAttachment, count: usize) {
    guard_void(|| {
        let module = module(m)?;
        let attachments = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(attachments, count) }
        };
        module.begin_compute(attachments.iter().map(|&attachment| attachment.into()));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_end_compute(m: *mut VirModule, out: *mut u32, count: usize) -> vk::Result {
    guard_result(|| {
        let module = module(m)?;
        let mut ids = vec![ValueId::INVALID; count];
        module.end_compute_into(&mut ids);
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
pub unsafe extern "C" fn vir_module_bind_compute_pipeline(m: *mut VirModule, pipeline: u32) {
    guard_void(|| {
        module(m)?.bind_compute_pipeline(PipelineId::from_raw(pipeline));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_dispatch(m: *mut VirModule, x: u32, y: u32, z: u32) {
    guard_void(|| {
        module(m)?.dispatch(x, y, z);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_dispatch_invocations(m: *mut VirModule, x: u32, y: u32, z: u32) {
    guard_void(|| {
        module(m)?.dispatch_invocations(x, y, z);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_dispatch_invocations_per_pixel_scaled(
    m: *mut VirModule, image: u32, scale_x: f32, scale_y: f32, scale_z: f32,
) {
    guard_void(|| {
        module(m)?.dispatch_invocations_per_pixel_scaled(ValueId(image), [scale_x, scale_y, scale_z]);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_dispatch_invocations_per_element_scaled(
    m: *mut VirModule, buffer: u32, element_size: u64, scale: f32,
) {
    guard_void(|| {
        module(m)?.dispatch_invocations_per_element_scaled(ValueId(buffer), element_size, scale);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_dispatch_indirect_at(m: *mut VirModule, buffer: u32, offset: u64) {
    guard_void(|| {
        module(m)?.dispatch_indirect_at(ValueId(buffer), offset);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_image(m: *mut VirModule, set: u32, binding: u32, image: u32) {
    guard_void(|| {
        module(m)?.bind_image(set, binding, ValueId(image));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_buffer_range(
    m: *mut VirModule, set: u32, binding: u32, buffer: u32, offset: u64, range: u64,
) {
    guard_void(|| {
        module(m)?.bind_buffer_range(set, binding, ValueId(buffer), offset, range);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_sampler(m: *mut VirModule, set: u32, binding: u32, sampler: u64) {
    guard_void(|| {
        module(m)?.bind_sampler(set, binding, vk::Sampler::from_raw(sampler));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_push_constant_address(m: *mut VirModule, offset: u32, buffer: u32) {
    guard_void(|| {
        module(m)?.push_constant_address(offset, ValueId(buffer));
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_specialize_constant(m: *mut VirModule, id: u32, kind: u32, bits: u64) {
    guard_void(|| {
        let value = match kind {
            0 => SpecValue::Bool(bits != 0),
            1 => SpecValue::U32(bits as u32),
            2 => SpecValue::I32(bits as u32 as i32),
            3 => SpecValue::F32(bits as u32),
            4 => SpecValue::F64(bits),
            kind => return Err(Failure::new(format!("unknown specialization constant kind {kind}"))),
        };
        module(m)?.specialize_constant(id, value);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_transient_image(m: *mut VirModule, info: *const VirTransientImageInfo) -> u32 {
    guard_id(|| {
        let info = unsafe { info.as_ref() }.ok_or_else(|| Failure::new("image info is null"))?;
        let info = unsafe { transient_image_info(info) }?;
        Ok(module(m)?.transient_image(&info).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_transient_buffer(m: *mut VirModule, info: *const VirBufferInfo) -> u32 {
    guard_id(|| {
        let info = unsafe { info.as_ref() }.ok_or_else(|| Failure::new("buffer info is null"))?;
        let info = unsafe { buffer_info(info) }?;
        Ok(module(m)?.transient_buffer(&info).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_declare_buffer_var(
    m: *mut VirModule, name: *const u8, name_len: usize, access: u64,
) -> u32 {
    guard_id(|| {
        let name = unsafe { text(name, name_len) }?;
        Ok(module(m)?.declare_buffer_var(name, Access::from_bits_retain(access)).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_declare_u32_var(
    m: *mut VirModule, name: *const u8, name_len: usize, default: u32,
) -> u32 {
    guard_id(|| {
        let name = unsafe { text(name, name_len) }?;
        Ok(module(m)?.declare_u32_var(name, default).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_copy_buffer(
    m: *mut VirModule, src: u32, dst: u32, regions: *const VirBufferCopy, count: usize,
) -> u32 {
    guard_id(|| {
        let regions = unsafe { slice(regions, count) }?
            .iter()
            .map(|&region| region.into())
            .collect::<Vec<BufferCopy>>();
        Ok(module(m)?.copy_buffer(ValueId(src), ValueId(dst), &regions).0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_copy_buffer_to_image_region(
    m: *mut VirModule, buffer: u32, image: u32, regions: *const VirBufferImageCopy, count: usize,
) -> u32 {
    guard_id(|| {
        let regions = unsafe { slice(regions, count) }?;
        let module = module(m)?;
        let mut copied = ValueId(image);
        for &region in regions {
            copied = module.copy_buffer_to_image_region(ValueId(buffer), copied, region.into());
        }

        Ok(copied.0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw_range(
    m: *mut VirModule, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32,
) {
    guard_void(|| {
        module(m)?.draw_range(vertex_count, instance_count, first_vertex, first_instance);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw_indexed_range(
    m: *mut VirModule, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32,
) {
    guard_void(|| {
        module(m)?.draw_indexed_range(index_count, instance_count, first_index, vertex_offset, first_instance);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw_indirect_at(
    m: *mut VirModule, buffer: u32, offset: u64, draw_count: u32, stride: u32,
) {
    guard_void(|| {
        module(m)?.draw_indirect_at(ValueId(buffer), offset, draw_count, stride);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw_indexed_indirect_at(
    m: *mut VirModule, buffer: u32, offset: u64, draw_count: u32, stride: u32,
) {
    guard_void(|| {
        module(m)?.draw_indexed_indirect_at(ValueId(buffer), offset, draw_count, stride);
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw_indirect_count_at(
    m: *mut VirModule, buffer: u32, offset: u64, count_buffer: u32, count_offset: u64, max_draw_count: u32, stride: u32,
) {
    guard_void(|| {
        module(m)?.draw_indirect_count_at(
            ValueId(buffer),
            offset,
            ValueId(count_buffer),
            count_offset,
            max_draw_count,
            stride,
        );
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_draw_indexed_indirect_count_at(
    m: *mut VirModule, buffer: u32, offset: u64, count_buffer: u32, count_offset: u64, max_draw_count: u32, stride: u32,
) {
    guard_void(|| {
        module(m)?.draw_indexed_indirect_count_at(
            ValueId(buffer),
            offset,
            ValueId(count_buffer),
            count_offset,
            max_draw_count,
            stride,
        );
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_module_bind_index_buffer_at(m: *mut VirModule, buffer: u32, offset: u64, index_type: i32) {
    guard_void(|| {
        module(m)?.bind_index_buffer_at(ValueId(buffer), offset, vk::IndexType::from_raw(index_type));
        Ok(())
    });
}
