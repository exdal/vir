//! Geometry from vertex buffers instead of the shader.
//!
//! The quad is uploaded once to a persistent buffer and drawn indexed each frame. The spinning
//! triangle is rebuilt on the CPU every frame in a frame-allocator buffer, recycled once the
//! frame retires.

use ash::vk;
use vir::{
    BlendPreset,
    Buffer,
    BufferInfo,
    ClearValue,
    MemoryLocation,
    PersistentAllocator,
    PipelineId,
    RasterizationState,
    Rect2D,
    RenderGraph,
    ValueId,
    allocator::Allocator,
};
use vir_examples::{Example, Frame, Setup, graphics_pipeline};

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vertex_buffer.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vertex_buffer.frag.spv"));

/// One vertex as `vs_main` reads it: `float2 position` then `float3 color`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

/// The block `vertex_buffer.slang` declares. Only the fragment stage reads it.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PushConstants {
    highlight: [f32; 3],
    tint: f32,
}

/// Byte offset of `tint` in the block.
const TINT_OFFSET: u32 = 12;

const BACKGROUND: ClearValue = ClearValue::rgba_f32(0.02, 0.02, 0.05, 1.0);

/// The quad on the left, uploaded once.
const QUAD: [Vertex; 4] = [
    Vertex {
        position: [-0.9, -0.5],
        color: [1.0, 1.0, 0.0],
    },
    Vertex {
        position: [-0.1, -0.5],
        color: [0.0, 1.0, 1.0],
    },
    Vertex {
        position: [-0.1, 0.5],
        color: [1.0, 0.0, 1.0],
    },
    Vertex {
        position: [-0.9, 0.5],
        color: [0.2, 0.2, 0.2],
    },
];

/// Two triangles over the four quad corners, drawn indexed.
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

/// A triangle on the other side, spun by `angle`, rebuilt every frame.
fn spinning_triangle(angle: f32) -> [Vertex; 3] {
    let (sin, cos) = angle.sin_cos();
    let colors = [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]];

    std::array::from_fn(|index| {
        let corner = std::f32::consts::TAU * index as f32 / 3.0;
        let (x, y) = (0.35 * corner.cos(), 0.35 * corner.sin());
        Vertex {
            position: [0.5 + x * cos - y * sin, x * sin + y * cos],
            color: colors[index],
        }
    })
}

struct VertexBuffer {
    pipeline: PipelineId,
    quad: Buffer,
    quad_indices: Buffer,
    ui: UiState,
}

struct UiState {
    spin: bool,
    tint: f32,
}

impl Default for UiState {
    fn default() -> Self { Self { spin: true, tint: 0.0 } }
}

impl Example for VertexBuffer {
    const TITLE: &'static str = "vir: vertex buffer";

    fn new(setup: &mut Setup) -> Result<Self, vk::Result> {
        let pipeline = graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?;

        let mut quad = setup
            .allocator
            .allocate_buffer(&BufferInfo::vertex(size_of_val(&QUAD) as u64).with_name("quad vertices"))?;
        quad.write(0, &QUAD)?;

        let mut quad_indices = setup.allocator.allocate_buffer(
            &BufferInfo::new(
                size_of_val(&QUAD_INDICES) as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )
            .with_name("quad indices"),
        )?;
        quad_indices.write(0, &QUAD_INDICES)?;

        Ok(Self {
            pipeline,
            quad,
            quad_indices,
            ui: UiState::default(),
        })
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([280.0, 160.0])
            .show(ctx, |ui| {
                ui.label("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.");
                ui.checkbox(&mut self.ui.spin, "spin");
                ui.add(egui::Slider::new(&mut self.ui.tint, 0.0..=1.0).text("tint"));
                ui.separator();
                if ui.button("reset").clicked() {
                    self.ui = UiState::default();
                }
            });
    }

    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result> {
        let spinning = spinning_triangle(if self.ui.spin { frame.elapsed } else { 0.0 });
        let mut spinning_buffer = frame
            .allocator
            .allocate_buffer(&BufferInfo::vertex(size_of_val(&spinning) as u64).with_name("spinning triangle"))?;
        spinning_buffer.write(0, &spinning)?;

        let target = frame.module.clear(frame.swapchain_image, BACKGROUND);

        let quad = frame.module.import_buffer(&self.quad, vir::Access::HostWrite);
        let quad_indices = frame.module.import_buffer(&self.quad_indices, vir::Access::HostWrite);
        let spinning = frame.module.import_buffer(&spinning_buffer, vir::Access::HostWrite);
        frame.module.set_name(quad, "quad vertices");
        frame.module.set_name(quad_indices, "quad indices");
        frame.module.set_name(spinning, "spinning triangle");

        // one pass, two draws: only the bound buffer and the tail of the block change
        Ok(frame
            .module
            .begin_rendering(&[target])
            .with_name("buffer geometry")
            .bind_graphics_pipeline(self.pipeline)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .push_constants(&PushConstants {
                highlight: [1.0, 1.0, 1.0],
                tint: 0.0,
            })
            .bind_vertex_buffer(0, quad)
            .bind_index_buffer(quad_indices, vk::IndexType::UINT32)
            .draw_indexed(QUAD_INDICES.len() as u32, 1)
            .push_constants_at(TINT_OFFSET, &self.ui.tint)
            .bind_vertex_buffer(0, spinning)
            .draw(3, 1)
            .end_rendering())
    }

    fn destroy(&mut self, _graph: &mut RenderGraph, allocator: &mut PersistentAllocator) {
        allocator.deallocate_buffer(self.quad);
        allocator.deallocate_buffer(self.quad_indices);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<VertexBuffer>() }

#[cfg(test)]
mod tests {
    use vir::{
        VertexAttribute,
        VertexLayout,
        resource::{pipeline::push_constant_ranges, shader},
    };
    use vir_examples::read_spirv;

    use super::*;

    /// The reflected vertex layout must match the `Vertex` struct the example uploads.
    #[test]
    fn the_reflected_vertex_layout_matches_the_uploaded_vertex() {
        let reflections = [
            shader::reflect(&read_spirv(VERT_SPV)).expect("vertex shader should reflect"),
            shader::reflect(&read_spirv(FRAG_SPV)).expect("fragment shader should reflect"),
        ];

        let layout = VertexLayout::interleaved(&reflections);
        assert_eq!(layout.stride as usize, size_of::<Vertex>());
        assert_eq!(
            layout.attributes,
            vec![
                VertexAttribute {
                    location: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 8,
                },
            ]
        );
    }

    /// The vertex stage ignores the block, so its range is fragment-only.
    #[test]
    fn a_stage_that_ignores_the_block_stays_out_of_its_range() {
        let reflect = |spirv| shader::reflect(&read_spirv(spirv)).expect("shader should reflect");

        let ranges = push_constant_ranges(&[reflect(VERT_SPV), reflect(FRAG_SPV)]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stage_flags, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(ranges[0].size as usize, size_of::<PushConstants>());

        // TINT_OFFSET names the last member, pushed alone by the spinning draw
        assert_eq!(TINT_OFFSET as usize, size_of::<PushConstants>() - size_of::<f32>());
    }
}
