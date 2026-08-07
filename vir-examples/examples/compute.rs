//! Geometry that never exists on the CPU.
//!
//! Two dispatches and a draw, chained by what each writes: the first places a triangle per
//! thread, the second expands the placements into corners, and the draw binds the result as its
//! vertex buffer. No descriptor sets; both buffers reach the dispatches as device addresses in
//! the push constant block.
//!
//! The point is the two barriers the example never writes. From the accesses each region
//! declares, the graph works out that the expand waits on the place and the draw's vertex input
//! waits on the expand.

use ash::vk;
use vir::{
    BlendPreset,
    BufferInfo,
    ClearValue,
    ComputePipelineInfo,
    MemoryLocation,
    PipelineId,
    RasterizationState,
    Rect2D,
    ValueId,
    allocator::Allocator,
};
use vir_examples::{Example, Frame, Setup, graphics_pipeline, read_spirv};

const PLACE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compute.place.spv"));
const EXPAND_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compute.expand.spv"));
const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compute.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compute.frag.spv"));

const BACKGROUND: ClearValue = ClearValue::rgba_f32(0.02, 0.02, 0.05, 1.0);

/// The `[numthreads]` both dispatches declare, used to turn a triangle count into groups.
const WORKGROUP_SIZE: u32 = 64;

/// Bytes `cs_expand` writes per vertex: `float2 position` then `float3 color`, tightly packed.
const VERTEX_SIZE: u64 = 20;

/// Bytes `cs_place` writes per triangle: `float2 center`, `float radius`, `float3 color`.
const INSTANCE_SIZE: u64 = 24;

/// Device-only buffer, still given an address so a pushed pointer can reach it.
fn device_buffer(size: u64, extra: vk::BufferUsageFlags, name: &str) -> BufferInfo {
    BufferInfo::new(
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | extra,
        MemoryLocation::GpuOnly,
    )
    .with_name(name)
}

/// The block `compute.slang` declares. Both dispatches read it; the draw reads none of it.
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    instances: vk::DeviceAddress,
    vertices: vk::DeviceAddress,
    count: u32,
    time: f32,
    scale: [f32; 2],
}

struct Compute {
    place: PipelineId,
    expand: PipelineId,
    draw: PipelineId,
    ui: UiState,
}

struct UiState {
    count: u32,
    animate: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            count: 96,
            animate: true,
        }
    }
}

/// Per-axis shrink that keeps the ring round in `target`.
fn ring_scale(target: vk::Extent2D) -> [f32; 2] {
    let aspect = target.width.max(1) as f32 / target.height.max(1) as f32;
    if aspect > 1.0 {
        [1.0 / aspect, 1.0]
    } else {
        [1.0, aspect]
    }
}

impl Example for Compute {
    const TITLE: &'static str = "vir: compute";

    fn new(setup: &mut Setup) -> Result<Self, vk::Result> {
        Ok(Self {
            place: setup
                .graph
                .declare_compute_pipeline(ComputePipelineInfo::new(&read_spirv(PLACE_SPV)))?,
            expand: setup
                .graph
                .declare_compute_pipeline(ComputePipelineInfo::new(&read_spirv(EXPAND_SPV)))?,
            draw: graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?,
            ui: UiState::default(),
        })
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([300.0, 180.0])
            .show(ctx, |ui| {
                ui.label(format!(
                    "{} triangles placed and expanded by {} groups of {WORKGROUP_SIZE}",
                    self.ui.count,
                    self.ui.count.div_ceil(WORKGROUP_SIZE),
                ));
                ui.checkbox(&mut self.ui.animate, "animate");
                ui.add(egui::Slider::new(&mut self.ui.count, 1..=1024).text("triangles"));
                ui.separator();
                if ui.button("reset").clicked() {
                    self.ui = UiState::default();
                }
            });
    }

    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result> {
        let count = self.ui.count;

        // both buffers live and die with the frame, recycled by the frame allocator once it retires
        let instances = frame.allocator.allocate_buffer(&device_buffer(
            count as u64 * INSTANCE_SIZE,
            vk::BufferUsageFlags::empty(),
            "instances",
        ))?;
        let vertices = frame.allocator.allocate_buffer(&device_buffer(
            count as u64 * 3 * VERTEX_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "computed vertices",
        ))?;

        let push = PushConstants {
            instances: instances.device_address(),
            vertices: vertices.device_address(),
            count,
            time: if self.ui.animate { frame.elapsed } else { 0.0 },
            scale: ring_scale(frame.target.extent),
        };

        let instances = frame.module.import_buffer(&instances);
        let vertices = frame.module.import_buffer(&vertices);
        frame.module.set_name(instances, "instances");
        frame.module.set_name(vertices, "computed vertices");

        let placed = frame
            .module
            .begin_compute()
            .with_name("place")
            .bind_pipeline(self.place)
            .write(instances)
            .push_constants(&push)
            .dispatch_invocations(count, 1, 1)
            .end_compute();

        // write vertices first so `expanded` is what the draw binds; read `placed`, not
        // `instances`, to order this dispatch after the place
        let expanded = frame
            .module
            .begin_compute()
            .with_name("expand")
            .bind_pipeline(self.expand)
            .write(vertices)
            .read(placed)
            .push_constants(&push)
            .dispatch(count, 1, 1)
            .end_compute();

        let target = frame.module.clear(frame.swapchain_image, BACKGROUND);

        Ok(frame
            .module
            .begin_rendering(&[target])
            .with_name("computed geometry")
            .bind_graphics_pipeline(self.draw)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .bind_vertex_buffer(0, expanded)
            .draw(count * 3, 1)
            .end_rendering())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<Compute>() }

#[cfg(test)]
mod tests {
    use vir::{Buffer, IR, Module, VertexAttribute, VertexLayout, resource::shader};

    use super::*;

    /// A region stands in for the first resource it declared, so writing vertices before reading
    /// placements makes `expanded` the buffer the draw binds.
    #[test]
    fn the_expand_region_stands_in_for_the_buffer_the_draw_binds() {
        let mut module = Module::default();
        let instances = module.import_buffer(&Buffer::default());
        let vertices = module.import_buffer(&Buffer::default());

        let placed = module.begin_compute().write(instances).dispatch(1, 1, 1).end_compute();
        let expanded = module
            .begin_compute()
            .write(vertices)
            .read(placed)
            .dispatch(1, 1, 1)
            .end_compute();

        // the expand region is the last one the compiled module opens
        let expand = module
            .compile(expanded)
            .into_iter()
            .rev()
            .find_map(|(_, ir)| match ir {
                IR::BeginCompute { resources, .. } => Some(resources),
                _ => None,
            })
            .expect("the compute regions should be present");
        assert_eq!(expand.first().map(|(id, _)| *id), Some(vertices));
    }

    /// Nothing else checks that the dispatch's writes and the draw's vertex input agree, so the
    /// reflected layout must be the tightly packed 20 bytes `cs_expand` stores.
    #[test]
    fn the_reflected_vertex_layout_matches_what_the_dispatch_writes() {
        let reflections = [
            shader::reflect(&read_spirv(VERT_SPV)).expect("vertex shader should reflect"),
            shader::reflect(&read_spirv(FRAG_SPV)).expect("fragment shader should reflect"),
        ];

        let layout = VertexLayout::interleaved(&reflections);
        assert_eq!(layout.stride as u64, VERTEX_SIZE);
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

    /// Both dispatches reach their buffers through addresses in the block, so neither declares a
    /// binding.
    #[test]
    fn the_dispatches_reach_their_buffers_without_a_descriptor_set() {
        for spirv in [PLACE_SPV, EXPAND_SPV] {
            let reflection = shader::reflect(&read_spirv(spirv)).expect("shader should reflect");

            assert_eq!(reflection.stage, vk::ShaderStageFlags::COMPUTE);
            assert_eq!(reflection.local_size, [WORKGROUP_SIZE, 1, 1]);
            assert!(reflection.bindings.is_empty());
            assert_eq!(reflection.push_constant_offset, 0);
            assert_eq!(reflection.push_constant_size as usize, size_of::<PushConstants>());
        }
    }

    /// The geometry arrives through the vertex buffer, so the draw pushes nothing.
    #[test]
    fn the_draw_pushes_nothing() {
        let reflect = |spirv| shader::reflect(&read_spirv(spirv)).expect("shader should reflect");

        let ranges = vir::resource::pipeline::push_constant_ranges(&[reflect(VERT_SPV), reflect(FRAG_SPV)]);
        assert!(ranges.is_empty());

        let ranges = vir::resource::pipeline::push_constant_ranges(&[reflect(PLACE_SPV)]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stage_flags, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(ranges[0].size as usize, size_of::<PushConstants>());
    }
}
