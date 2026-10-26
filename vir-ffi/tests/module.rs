use std::{ffi::c_void, ptr};

use ash::vk;
use vir::{Access, ClearValue, DomainFlag, Image, ImageAttachment, Unchecked, ValueId};
use vir_ffi::*;

const INVALID: u32 = u32::MAX;

fn last_error() -> String {
    let len = unsafe { vir_last_error(ptr::null_mut(), 0) };
    let mut buf = vec![0u8; len + 1];
    assert_eq!(unsafe { vir_last_error(buf.as_mut_ptr(), buf.len()) }, len);
    assert_eq!(buf[len], 0);
    String::from_utf8(buf[..len].to_vec()).unwrap()
}

fn image(module: *mut VirModule) -> u32 {
    let extent = vk::Extent3D::default().width(64).height(32).depth(1);
    let format = vk::Format::R8G8B8A8_SRGB;
    let attachment = ImageAttachment::new(
        Image::imported(vk::Image::null(), format, extent, vk::SampleCountFlags::TYPE_1),
        format,
        extent,
        vk::SampleCountFlags::TYPE_1,
        vk::ImageLayout::UNDEFINED,
    );

    unsafe { &mut (*module).0 }
        .import_attachment(&attachment, Access::None)
        .0
}

fn rect(tag: u32) -> VirRect2D {
    VirRect2D {
        tag,
        absolute: vk::Rect2D {
            offset: vk::Offset2D { x: 1, y: 2 },
            extent: vk::Extent2D { width: 3, height: 4 },
        },
        relative: [0.0, 0.0, 0.5, 0.5],
    }
}

unsafe fn begin_pass(module: *mut VirModule, target: u32) {
    let attachment = VirAttachment {
        value: target,
        access: Access::ColorWrite.bits(),
    };
    unsafe { vir_module_begin_rendering(module, &attachment, 1) };
}

#[test]
fn records_a_pass() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);

        let name = "scale";
        let var = vir_module_declare_bytes_var(module, name.as_ptr(), name.len(), 8);
        assert_ne!(var, INVALID);

        let cleared = vir_module_clear(module, target, ClearValue::rgba_f32(0.0, 0.0, 0.0, 1.0));
        assert_ne!(cleared, INVALID);

        begin_pass(module, cleared);
        let pass_name = "pass";
        vir_module_with_name(module, pass_name.as_ptr(), pass_name.len());
        for tag in 0..3 {
            vir_module_set_viewport(
                module,
                0,
                VirViewport {
                    rect: rect(tag),
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
            );
            vir_module_set_scissor(module, 0, rect(tag));
        }

        vir_module_broadcast_color_blend(module, 1);
        vir_module_set_dynamic_state(module, 3);
        vir_module_push_constant_bytes(module, 0, [0u8; 8].as_ptr(), 8);
        vir_module_draw(module, 3, 1);

        let mut out = [INVALID];
        assert_eq!(
            vir_module_end_rendering(module, out.as_mut_ptr(), 1),
            vk::Result::SUCCESS
        );
        assert_ne!(out[0], INVALID);

        let present = vir_module_present(module, out[0]);
        assert_ne!(present, INVALID);

        vir_module_destroy(module);
    }
}

#[test]
fn bad_inputs_report_errors() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);

        let invalid_utf8 = [0xffu8, 0xfe];
        assert_eq!(
            vir_module_declare_bool_var(module, invalid_utf8.as_ptr(), 2, false),
            INVALID
        );
        assert!(last_error().contains("UTF-8"));

        begin_pass(module, target);
        vir_module_broadcast_color_blend(module, 99);
        assert!(last_error().contains("blend preset"));

        let bad_rect = rect(7);
        vir_module_set_scissor(module, 0, bad_rect);
        assert!(last_error().contains("rect tag"));

        vir_module_destroy(module);
    }
}

#[test]
fn end_rendering_with_wrong_count_is_an_error() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);
        begin_pass(module, target);

        let mut out = [INVALID; 2];
        let result = vir_module_end_rendering(module, out.as_mut_ptr(), 2);
        assert_eq!(result, vk::Result::ERROR_UNKNOWN);
        assert!(last_error().contains("declared 1 resources but its result expects 2"));

        vir_module_destroy(module);
    }
}

#[test]
fn empty_begin_rendering_is_an_error() {
    unsafe {
        let module = vir_module_create();
        vir_module_begin_rendering(module, ptr::null(), 0);
        assert!(last_error().contains("at least one resource"));
        vir_module_destroy(module);
    }
}

unsafe fn declare_values(module: *mut VirModule, user_data: *mut c_void, out: *mut u32, count: usize) -> usize {
    unsafe { *user_data.cast::<u32>() += 1 };
    let name = "inner";
    for index in 0..count {
        unsafe { *out.add(index) = vir_module_declare_bool_var(module, name.as_ptr(), name.len(), true) };
    }

    count
}

unsafe extern "C" fn arm(module: *mut VirModule, user_data: *mut c_void, out: *mut u32, _: usize) -> usize {
    unsafe { declare_values(module, user_data, out, 2) }
}

unsafe extern "C" fn arm_of_three(module: *mut VirModule, user_data: *mut c_void, out: *mut u32, _: usize) -> usize {
    unsafe { declare_values(module, user_data, out, 3) }
}

unsafe extern "C" fn lazy_arm(_: *mut VirModule, user_data: *mut c_void, _: *mut u32, _: usize) -> usize {
    unsafe { *user_data.cast::<u32>() += 1 };
    1
}

unsafe extern "C" fn empty_arm(_: *mut VirModule, _: *mut c_void, _: *mut u32, _: usize) -> usize {
    0
}

unsafe extern "C" fn overflowing_arm(_: *mut VirModule, _: *mut c_void, _: *mut u32, capacity: usize) -> usize {
    capacity + 1
}

unsafe fn condition(
    module: *mut VirModule, then: VirConditionFn, otherwise: VirConditionFn, capacity: usize,
) -> (vk::Result, Vec<u32>, usize, u32) {
    let name = "flag";
    let flag = unsafe { vir_module_declare_bool_var(module, name.as_ptr(), name.len(), true) };
    let mut calls = 0u32;
    let mut out = vec![INVALID; capacity];
    let mut count = 0usize;
    let result = unsafe {
        vir_module_set_condition(
            module,
            flag,
            then,
            otherwise,
            (&raw mut calls).cast(),
            out.as_mut_ptr(),
            capacity,
            &raw mut count,
        )
    };

    (result, out, count, calls)
}

#[test]
fn set_condition_runs_both_arms() {
    unsafe {
        let module = vir_module_create();
        let (result, out, count, calls) = condition(module, arm, arm, 8);
        assert_eq!(result, vk::Result::SUCCESS);
        assert_eq!(calls, 2);
        assert_eq!(count, 2);
        assert!(out[..2].iter().all(|&id| id != INVALID));
        assert!(out[2..].iter().all(|&id| id == INVALID));

        vir_module_destroy(module);
    }
}

#[test]
fn set_condition_rejects_bad_counts_and_unfilled_results() {
    unsafe {
        let module = vir_module_create();

        let (result, ..) = condition(module, arm, arm, 0);
        assert_eq!(result, vk::Result::ERROR_UNKNOWN);
        assert!(last_error().contains("at least one value"));

        let (result, ..) = condition(module, lazy_arm, lazy_arm, 2);
        assert_ne!(result, vk::Result::SUCCESS);
        assert!(last_error().contains("left a result unset"));

        let (result, ..) = condition(module, empty_arm, empty_arm, 2);
        assert_ne!(result, vk::Result::SUCCESS);
        assert!(last_error().contains("produced no values"));

        let (result, ..) = condition(module, overflowing_arm, overflowing_arm, 2);
        assert_ne!(result, vk::Result::SUCCESS);
        assert!(last_error().contains("more than the capacity"));

        let (result, ..) = condition(module, arm, arm_of_three, 8);
        assert_ne!(result, vk::Result::SUCCESS);

        vir_module_destroy(module);
    }
}

#[test]
fn records_an_indexed_draw_over_imported_resources() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);

        let vertices = VirBuffer(vir::Buffer::default());
        let indices = VirBuffer(vir::Buffer::default());
        let vertex_id = vir_module_import_buffer(module, &vertices, Access::HostWrite.bits());
        let index_id = vir_module_import_buffer(module, &indices, Access::HostWrite.bits());
        assert_ne!(vertex_id, INVALID);
        assert_ne!(index_id, INVALID);

        let extent = vk::Extent3D::default().width(8).height(8).depth(1);
        let texture = VirImage {
            image: Image::imported(
                vk::Image::null(),
                vk::Format::R8G8B8A8_SRGB,
                extent,
                vk::SampleCountFlags::TYPE_1,
            ),
            view: vk::ImageView::null(),
        };
        let layout = vir_access_image_layout(Access::FragmentSampled.bits());
        assert_eq!(layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL.as_raw());
        let texture_id = vir_module_import_image(module, &texture, layout, Access::FragmentSampled.bits());
        assert_ne!(texture_id, INVALID);

        let name = "texture";
        assert_eq!(
            vir_module_set_name(module, texture_id, name.as_ptr(), name.len()),
            texture_id
        );

        begin_pass(module, target);
        vir_module_set_depth(
            module,
            VirDepthState {
                test_enable: 1,
                write_enable: 1,
                compare_op: vk::CompareOp::LESS.as_raw(),
            },
        );
        vir_module_bind_texture(module, 0, 0, texture_id, 0);
        vir_module_bind_vertex_buffer(module, 0, vertex_id);
        vir_module_bind_index_buffer(module, index_id, vk::IndexType::UINT32.as_raw());
        vir_module_draw_indexed(module, 6, 1);

        let mut out = [INVALID];
        assert_eq!(
            vir_module_end_rendering(module, out.as_mut_ptr(), 1),
            vk::Result::SUCCESS
        );

        let exported = vir_module_export(module, out[0], Access::Present.bits(), DomainFlag::Present.bits());
        assert_ne!(exported, INVALID);

        vir_module_destroy(module);
    }
}

#[test]
fn null_resources_are_errors() {
    unsafe {
        let module = vir_module_create();
        assert_eq!(vir_module_import_buffer(module, ptr::null(), 0), INVALID);
        assert!(last_error().contains("buffer is null"));
        assert_eq!(vir_module_import_image(module, ptr::null(), 0, 0), INVALID);
        assert!(last_error().contains("image is null"));
        vir_module_destroy(module);
    }
}

#[test]
fn program_set_bytes_with_wrong_size_is_an_error() {
    unsafe {
        let module = vir_module_create();
        let name = "constants";
        let var = vir_module_declare_bytes_var(module, name.as_ptr(), name.len(), 8);
        let flag_name = "flag";
        let flag = vir_module_declare_bool_var(module, flag_name.as_ptr(), flag_name.len(), false);
        let target = image(module);
        let present = vir_module_present(module, target);

        let program = (*module).0.compile(&Unchecked, ValueId(present)).unwrap();
        let program = Box::into_raw(Box::new(VirProgram(program)));

        assert_eq!(
            vir_program_set_bytes(program, var, [0u8; 8].as_ptr(), 8),
            vk::Result::SUCCESS
        );
        assert_eq!(vir_program_set_bool(program, flag, true), vk::Result::SUCCESS);

        assert_eq!(
            vir_program_set_bytes(program, var, [0u8; 4].as_ptr(), 4),
            vk::Result::ERROR_UNKNOWN
        );
        assert!(!last_error().is_empty());

        assert_eq!(
            vir_program_set_bytes(program, 9999, ptr::null(), 0),
            vk::Result::ERROR_UNKNOWN
        );
        assert!(last_error().contains("not a variable"));

        vir_program_destroy(program);
        vir_module_destroy(module);
    }
}

fn transient_buffer(module: *mut VirModule) -> u32 {
    let info = VirBufferInfo {
        size: 256,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER.as_raw(),
        location: 1,
        alignment: 1,
        name: ptr::null(),
        name_len: 0,
    };
    unsafe { vir_module_transient_buffer(module, &info) }
}

unsafe fn begin_compute(module: *mut VirModule, values: &[u32]) {
    let attachments = values
        .iter()
        .map(|&value| VirAttachment {
            value,
            access: Access::ComputeWrite.bits(),
        })
        .collect::<Vec<_>>();
    unsafe { vir_module_begin_compute(module, attachments.as_ptr(), attachments.len()) };
}

#[test]
fn records_a_compute_pass() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);
        let buffer = transient_buffer(module);
        assert_ne!(buffer, INVALID);

        begin_compute(module, &[target, buffer]);
        vir_module_bind_compute_pipeline(module, 0);
        vir_module_bind_image(module, 0, 0, target);
        vir_module_bind_buffer_range(module, 0, 1, buffer, 16, 64);
        vir_module_bind_sampler(module, 0, 2, 0);
        vir_module_push_constant_address(module, 0, buffer);
        for (kind, bits) in [
            (0, 1),
            (1, 7),
            (2, (-3i32) as u32 as u64),
            (3, 1.5f32.to_bits() as u64),
            (4, 2.5f64.to_bits()),
        ] {
            vir_module_specialize_constant(module, 0, kind, bits);
            assert!(last_error().is_empty());
        }

        vir_module_dispatch_invocations_per_pixel_scaled(module, target, 1.0, 0.5, 1.0);
        assert!(last_error().is_empty());

        let mut out = [INVALID; 2];
        assert_eq!(vir_module_end_compute(module, out.as_mut_ptr(), 2), vk::Result::SUCCESS);
        assert_ne!(out[0], INVALID);
        assert_ne!(out[1], INVALID);
        assert_ne!(out[0], out[1]);

        vir_module_destroy(module);
    }
}

#[test]
fn records_every_dispatch_form() {
    unsafe {
        let module = vir_module_create();
        let buffer = transient_buffer(module);
        begin_compute(module, &[buffer]);
        vir_module_dispatch(module, 1, 2, 3);
        vir_module_dispatch_invocations(module, 64, 64, 1);
        vir_module_dispatch_invocations_per_element_scaled(module, buffer, 16, 1.0);
        vir_module_dispatch_indirect_at(module, buffer, 0);
        assert!(last_error().is_empty());

        let mut out = [INVALID];
        assert_eq!(vir_module_end_compute(module, out.as_mut_ptr(), 1), vk::Result::SUCCESS);

        vir_module_destroy(module);
    }
}

#[test]
fn end_compute_with_wrong_count_is_an_error() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);
        begin_compute(module, &[target]);

        let mut out = [INVALID; 2];
        assert_eq!(
            vir_module_end_compute(module, out.as_mut_ptr(), 2),
            vk::Result::ERROR_UNKNOWN
        );
        assert!(last_error().contains("declared 1 resources but its result expects 2"));

        vir_module_destroy(module);
    }
}

#[test]
fn empty_begin_compute_is_an_error() {
    unsafe {
        let module = vir_module_create();
        vir_module_begin_compute(module, ptr::null(), 0);
        assert!(last_error().contains("at least one resource"));
        vir_module_destroy(module);
    }
}

#[test]
fn end_compute_while_rendering_is_an_error() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);
        begin_pass(module, target);

        let mut out = [INVALID];
        assert_eq!(
            vir_module_end_compute(module, out.as_mut_ptr(), 1),
            vk::Result::ERROR_UNKNOWN
        );
        assert!(!last_error().is_empty());

        vir_module_destroy(module);
    }
}

#[test]
fn bad_compute_inputs_report_errors() {
    unsafe {
        let module = vir_module_create();
        let target = image(module);
        begin_compute(module, &[target]);

        vir_module_specialize_constant(module, 0, 9, 0);
        assert!(last_error().contains("specialization constant kind"));

        let mut info = VirTransientImageInfo {
            width: 8,
            height: 8,
            depth: 1,
            format: vk::Format::R8G8B8A8_UNORM.as_raw(),
            usage: vk::ImageUsageFlags::STORAGE.as_raw(),
            mip_levels: 0,
            array_layers: 1,
            samples: 1,
            name: ptr::null(),
            name_len: 0,
        };
        assert_eq!(vir_module_transient_image(module, &info), INVALID);
        assert!(last_error().contains("at least 1"));

        info.mip_levels = 1;
        assert_ne!(vir_module_transient_image(module, &info), INVALID);

        assert_eq!(vir_module_transient_image(module, ptr::null()), INVALID);
        assert!(last_error().contains("image info is null"));
        assert_eq!(vir_module_transient_buffer(module, ptr::null()), INVALID);
        assert!(last_error().contains("buffer info is null"));

        vir_module_destroy(module);
    }
}
