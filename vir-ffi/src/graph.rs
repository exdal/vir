use std::io::Cursor;

use ash::vk;
use vir::{AllocatorKind, GraphicsPipelineInfo, RenderGraph};

use crate::{
    allocator::VirFrameAllocator,
    context::{VirContext, context, out_ptr},
    error::{Failure, guard_result, guard_void},
    program::VirProgram,
    types::bytes,
};

pub struct VirRenderGraph(pub RenderGraph);

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
    guard_result(|| {
        let graph = graph(g)?;
        let out = out_ptr(out)?;
        let read = |blob: &[u8]| {
            ash::util::read_spv(&mut Cursor::new(blob)).map_err(|err| Failure::new(format!("invalid SPIR-V: {err}")))
        };
        let vert = read(unsafe { bytes(vert, vert_len) })?;
        let frag = read(unsafe { bytes(frag, frag_len) })?;

        *out = graph
            .declare_pipeline(GraphicsPipelineInfo::new().with_shader(&vert).with_shader(&frag))?
            .raw();

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
