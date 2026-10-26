use std::{io::Cursor, ptr};

use ash::{vk, vk::Handle};
use vir::{AllocatorKind, ComputePipelineInfo, GraphicsPipelineInfo, RenderGraph};

use crate::{
    allocator::{VirFrameAllocator, VirPersistentAllocator},
    context::{VirContext, context, out_ptr},
    error::{Failure, guard_result, guard_void},
    program::VirProgram,
    types::bytes,
};

pub struct VirRenderGraph(pub RenderGraph);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirBindlessSet {
    pub index: u32,
    pub layout: u64,
    pub set: u64,
}

fn read_spirv(ptr: *const u8, len: usize) -> Result<Vec<u32>, Failure> {
    ash::util::read_spv(&mut Cursor::new(unsafe { bytes(ptr, len) }))
        .map_err(|err| Failure::new(format!("invalid SPIR-V: {err}")))
}

pub(crate) fn graph<'a>(g: *mut VirRenderGraph) -> Result<&'a mut RenderGraph, Failure> {
    unsafe { g.as_mut() }
        .map(|graph| &mut graph.0)
        .ok_or_else(|| Failure::new("render graph is null"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_create(c: *mut VirContext, out: *mut *mut VirRenderGraph) -> vk::Result {
    guard_result(|| {
        let graph = RenderGraph::new(&context(c)?.ctx);
        *out_ptr(out)? = Box::into_raw(Box::new(VirRenderGraph(graph)));
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_destroy(g: *mut VirRenderGraph) {
    guard_void(|| {
        if !g.is_null() {
            drop(unsafe { Box::from_raw(g) });
        }

        Ok(())
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_declare_graphics_pipeline(
    g: *mut VirRenderGraph, vert: *const u8, vert_len: usize, frag: *const u8, frag_len: usize, out: *mut u32,
) -> vk::Result {
    unsafe { vir_render_graph_declare_graphics_pipeline_bindless(g, vert, vert_len, frag, frag_len, ptr::null(), out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_declare_graphics_pipeline_bindless(
    g: *mut VirRenderGraph, vert: *const u8, vert_len: usize, frag: *const u8, frag_len: usize,
    bindless: *const VirBindlessSet, out: *mut u32,
) -> vk::Result {
    guard_result(|| {
        let graph = graph(g)?;
        let out = out_ptr(out)?;
        let vert = read_spirv(vert, vert_len)?;
        let frag = read_spirv(frag, frag_len)?;

        let mut info = GraphicsPipelineInfo::new().with_shader(&vert).with_shader(&frag);
        if let Some(set) = unsafe { bindless.as_ref() } {
            info = info.with_bindless_set(
                set.index,
                vk::DescriptorSetLayout::from_raw(set.layout),
                vk::DescriptorSet::from_raw(set.set),
            );
        }

        *out = graph.declare_pipeline(info)?.raw();

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_declare_compute_pipeline(
    g: *mut VirRenderGraph, spirv: *const u8, spirv_len: usize, out: *mut u32,
) -> vk::Result {
    unsafe { vir_render_graph_declare_compute_pipeline_bindless(g, spirv, spirv_len, ptr::null(), out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_declare_compute_pipeline_bindless(
    g: *mut VirRenderGraph, spirv: *const u8, spirv_len: usize, bindless: *const VirBindlessSet, out: *mut u32,
) -> vk::Result {
    guard_result(|| {
        let graph = graph(g)?;
        let out = out_ptr(out)?;
        let spirv = read_spirv(spirv, spirv_len)?;

        let mut info = ComputePipelineInfo::new(&spirv);
        if let Some(set) = unsafe { bindless.as_ref() } {
            info = info.with_bindless_set(
                set.index,
                vk::DescriptorSetLayout::from_raw(set.layout),
                vk::DescriptorSet::from_raw(set.set),
            );
        }

        *out = graph.declare_compute_pipeline(info)?.raw();

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_execute(
    g: *mut VirRenderGraph, c: *mut VirContext, p: *const VirProgram, frame: *mut VirFrameAllocator,
) -> vk::Result {
    guard_result(|| {
        let graph = graph(g)?;
        let context = context(c)?;
        let program = unsafe { p.as_ref() }.ok_or_else(|| Failure::new("program is null"))?;
        let frame = unsafe { frame.as_mut() }.ok_or_else(|| Failure::new("frame allocator is null"))?;

        graph.execute(&context.ctx, &program.0, &mut AllocatorKind::Frame(&mut frame.0))?;

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_wait(g: *mut VirRenderGraph) -> vk::Result {
    guard_result(|| {
        graph(g)?.wait()?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_render_graph_execute_blocking(
    g: *mut VirRenderGraph, c: *mut VirContext, p: *const VirProgram, a: *mut VirPersistentAllocator,
) -> vk::Result {
    guard_result(|| {
        let graph = graph(g)?;
        let context = context(c)?;
        let program = unsafe { p.as_ref() }.ok_or_else(|| Failure::new("program is null"))?;
        let allocator = unsafe { a.as_mut() }.ok_or_else(|| Failure::new("persistent allocator is null"))?;

        graph.execute_blocking(
            &context.ctx,
            &program.0,
            &mut AllocatorKind::Persistent(&mut allocator.0),
        )?;

        Ok(())
    })
}
