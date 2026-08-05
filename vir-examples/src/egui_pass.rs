use ash::vk;
use egui::epaint::{ClippedPrimitive, Primitive, Vertex as EguiVertex};
use vir::{
    BlendPreset,
    Buffer,
    BufferInfo,
    DynamicStateFlags,
    FrameAllocator,
    GraphicsPipelineInfo,
    MemoryLocation,
    Module,
    PipelineId,
    RasterizationState,
    Rect2D,
    RenderGraph,
    ValueId,
    allocator::Allocator,
};

/// The block `egui.slang` declares.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    screen_size: [f32; 2],
}

/// One primitive's slice of the frame's shared buffers.
struct MeshDraw {
    clip: vk::Rect2D,
    index_offset: u32,
    index_count: u32,
    vertex_offset: i32,
}

/// A frame's worth of tessellated geometry, already uploaded.
pub struct EguiFrame {
    vertices: Buffer,
    indices: Buffer,
    draws: Vec<MeshDraw>,
    screen_size_in_points: [f32; 2],
}

impl EguiFrame {
    pub fn is_empty(&self) -> bool { self.draws.is_empty() }

    pub fn draw_count(&self) -> usize { self.draws.len() }
}

pub struct EguiPass {
    pipeline: PipelineId,
}

/// Clips `rect`, which egui gives in points, to the framebuffer it is being drawn into.
fn scissor_from_clip(clip: egui::Rect, pixels_per_point: f32, target: vk::Extent2D) -> vk::Rect2D {
    let min_x = (clip.min.x * pixels_per_point).round().clamp(0.0, target.width as f32) as u32;
    let min_y = (clip.min.y * pixels_per_point).round().clamp(0.0, target.height as f32) as u32;
    let max_x = (clip.max.x * pixels_per_point).round().clamp(0.0, target.width as f32) as u32;
    let max_y = (clip.max.y * pixels_per_point).round().clamp(0.0, target.height as f32) as u32;

    vk::Rect2D {
        offset: vk::Offset2D {
            x: min_x as i32,
            y: min_y as i32,
        },
        extent: vk::Extent2D {
            width: max_x.saturating_sub(min_x),
            height: max_y.saturating_sub(min_y),
        },
    }
}

impl EguiPass {
    pub fn new(graph: &mut RenderGraph, vertex_spirv: &[u32], fragment_spirv: &[u32]) -> Result<Self, vk::Result> {
        let pipeline = graph.declare_pipeline(
            GraphicsPipelineInfo::new()
                .with_shader(vertex_spirv)
                .with_shader(fragment_spirv),
        )?;

        Ok(Self { pipeline })
    }

    /// Flattens every mesh into one vertex and one index buffer taken from this frame's
    /// allocator, so the whole UI draws out of a single binding.
    pub fn prepare(
        &self, allocator: &mut FrameAllocator, primitives: &[ClippedPrimitive], pixels_per_point: f32,
        target: vk::Extent2D,
    ) -> Result<EguiFrame, vk::Result> {
        let mut vertices: Vec<EguiVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut draws = Vec::new();

        for primitive in primitives {
            // `Callback` is for user-supplied rendering, which this example does not use
            let Primitive::Mesh(mesh) = &primitive.primitive else {
                continue;
            };
            if mesh.indices.is_empty() {
                continue;
            }

            let clip = scissor_from_clip(primitive.clip_rect, pixels_per_point, target);
            // a zero-area scissor is legal but the draw would be wasted work
            if clip.extent.width == 0 || clip.extent.height == 0 {
                continue;
            }

            draws.push(MeshDraw {
                clip,
                index_offset: indices.len() as u32,
                index_count: mesh.indices.len() as u32,
                vertex_offset: vertices.len() as i32,
            });

            vertices.extend_from_slice(&mesh.vertices);
            indices.extend_from_slice(&mesh.indices);
        }

        let screen_size_in_points = [
            target.width as f32 / pixels_per_point,
            target.height as f32 / pixels_per_point,
        ];

        if draws.is_empty() {
            return Ok(EguiFrame {
                vertices: Buffer::default(),
                indices: Buffer::default(),
                draws,
                screen_size_in_points,
            });
        }

        let mut vertex_buffer = allocator.allocate_buffer(
            &BufferInfo::vertex(size_of_val(vertices.as_slice()) as u64).with_name("egui vertices"),
        )?;
        vertex_buffer.write(0, &vertices)?;

        let mut index_buffer = allocator.allocate_buffer(
            &BufferInfo::new(
                size_of_val(indices.as_slice()) as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )
            .with_name("egui indices"),
        )?;
        index_buffer.write(0, &indices)?;

        Ok(EguiFrame {
            vertices: vertex_buffer,
            indices: index_buffer,
            draws,
            screen_size_in_points,
        })
    }

    /// Records the UI over `target`, returning the version of it the pass produced.
    pub fn record(&self, module: &mut Module, target: ValueId, frame: &EguiFrame) -> ValueId {
        if frame.is_empty() {
            return target;
        }

        let vertices = module.import_buffer(&frame.vertices);
        let indices = module.import_buffer(&frame.indices);

        let mut pass = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(self.pipeline)
            .set_dynamic_state(DynamicStateFlags::Viewport | DynamicStateFlags::Scissor)
            .set_viewport(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::PremultipliedAlphaBlend)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .push_constants(&PushConstants {
                screen_size: frame.screen_size_in_points,
            })
            .bind_vertex_buffer(0, vertices)
            .bind_index_buffer(indices, vk::IndexType::UINT32);

        for draw in &frame.draws {
            pass = pass
                .set_scissor(
                    0,
                    Rect2D::absolute(
                        draw.clip.offset.x,
                        draw.clip.offset.y,
                        draw.clip.extent.width,
                        draw.clip.extent.height,
                    ),
                )
                .draw_indexed_range(draw.index_count, 1, draw.index_offset, draw.vertex_offset, 0);
        }

        pass.end_rendering()
    }
}
