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

unsafe extern "C" fn empty_arm(_: *mut VirModule, _: *mut c_void, _: *mut u32, _: usize) -> usize { 0 }

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
