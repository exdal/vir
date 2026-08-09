//! Geometry that never exists on the CPU, out of a graph compiled once.
//!
//! Two dispatches and a draw, chained by what each writes: the first places a triangle per
//! thread, the second expands the placements into corners, and the draw binds the result as its
//! vertex buffer. No descriptor sets; both buffers reach the dispatches as device addresses in
//! the push constant block.
//!
//! The point is the two barriers the example never writes. From the accesses each region
//! declares, the graph works out that the expand waits on the place and the draw's vertex input
//! waits on the expand.
//!
//! All three are a [`Program`] built in `new` and re-run every frame. Neither buffer is in it:
//! both are slots the graph declares and the frame binds, so the buffer a dispatch writes can be
//! a different one, of a different length, every frame without anything being compiled again.
//! What the graph keeps is what it needs to order them, not what they are.
//!
//! Which is also why the addresses are pushed rather than written: what is bound is not known
//! until the frame binds it, so the block the host fills leaves those eight bytes to the graph.

use ash::vk;
use vir::{
    Access,
    AllocatorKind,
    BlendPreset,
    BufferInfo,
    ClearValue,
    ComputePipelineInfo,
    DomainFlag,
    ImageAttachment,
    ImageInfo,
    MemoryLocation,
    Module,
    PersistentAllocator,
    PipelineId,
    Program,
    RasterizationState,
    Rect2D,
    RenderGraph,
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

/// The top of the slider, and so the largest either transient is ever asked for.
const MAX_TRIANGLES: u32 = 1024;

/// The frame's last use of the target is a blit onto the swapchain, so that is its resting
/// layout.
const RESTING_ACCESS: Access = Access::BlitRead;

/// Bytes `cs_expand` writes per vertex: `float2 position` then `float3 color`, tightly packed.
const VERTEX_SIZE: u64 = 20;

/// Bytes `cs_place` writes per triangle: `float2 center`, `float radius`, `float3 color`.
const INSTANCE_SIZE: u64 = 24;

/// Where each address sits in the block `compute.slang` declares.
const INSTANCES_AT: u32 = 0;
const VERTICES_AT: u32 = size_of::<vk::DeviceAddress>() as u32;

/// A buffer the graph owns for the run. Only the memory it wants is named here: what it has to
/// be usable for is what the graph goes on to do with it.
#[cfg(test)]
fn transient(name: &str) -> BufferInfo {
    BufferInfo::new(0, vk::BufferUsageFlags::empty(), MemoryLocation::GpuOnly).with_name(name)
}

fn device_buffer(size: u64, extra: vk::BufferUsageFlags, name: &str) -> BufferInfo {
    BufferInfo::new(
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | extra,
        MemoryLocation::GpuOnly,
    )
    .with_name(name)
}

/// The block `compute.slang` declares. Both dispatches read it; the draw reads none of it. The
/// two addresses are left zero, since the graph patches them in once it has allocated.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PushConstants {
    instances: vk::DeviceAddress,
    vertices: vk::DeviceAddress,
    count: u32,
    time: f32,
    scale: [f32; 2],
}

/// The compiled graph and the slots the frame writes into it.
struct Pass {
    program: Program,
    triangles: ValueId,
    instances: ValueId,
    vertices: ValueId,
    vertex_count: ValueId,
    push: ValueId,
}

struct Compute {
    place: PipelineId,
    expand: PipelineId,
    draw: PipelineId,
    target: Option<ImageAttachment>,
    pass: Option<Pass>,
    ui: UiState,
}

impl Compute {
    fn compile_pass(&self, target: &ImageAttachment) -> Pass {
        let mut module = Module::default();

        let triangles = module.declare_u32_var("triangles", 0);
        let vertex_count = module.declare_u32_var("vertex count", 0);
        let push = module.declare_bytes_var("compute push block", size_of::<PushConstants>() as u32);

        let instances = module.declare_buffer_var("instances", vir::Access::HostWrite);
        let vertices = module.declare_buffer_var("computed vertices", vir::Access::HostWrite);

        let placed = module
            .begin_compute()
            .with_name("place")
            .bind_pipeline(self.place)
            .write(instances)
            .push_constants_from(push)
            .push_constant_address(INSTANCES_AT, instances)
            .push_constant_address(VERTICES_AT, vertices)
            .dispatch_invocations(triangles, 1, 1)
            .end_compute();

        // write vertices first so `expanded` is what the draw binds; read `placed`, not
        // `instances`, to order this dispatch after the place
        let expanded = module
            .begin_compute()
            .with_name("expand")
            .bind_pipeline(self.expand)
            .write(vertices)
            .read(placed)
            .push_constants_from(push)
            .push_constant_address(INSTANCES_AT, instances)
            .push_constant_address(VERTICES_AT, vertices)
            .dispatch_invocations(triangles, 1, 1)
            .end_compute();

        let attachment = module.import_attachment(target);
        module.set_name(attachment, "computed geometry");
        let attachment = module.clear(attachment, BACKGROUND);

        let drawn = module
            .begin_rendering(&[attachment])
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
            .draw(vertex_count, 1)
            .end_rendering();

        let ready = module.release(drawn, RESTING_ACCESS, DomainFlag::Graphics);
        let program = module.compile(ready);
        vir_examples::dump_ir("compute", &program);

        Pass {
            program,
            triangles,
            instances,
            vertices,
            vertex_count,
            push,
        }
    }
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
        let mut this = Self {
            place: setup
                .graph
                .declare_compute_pipeline(ComputePipelineInfo::new(&read_spirv(PLACE_SPV)))?,
            expand: setup
                .graph
                .declare_compute_pipeline(ComputePipelineInfo::new(&read_spirv(EXPAND_SPV)))?,
            draw: graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?,
            target: None,
            pass: None,
            ui: UiState::default(),
        };
        this.resize(setup)?;

        Ok(this)
    }

    fn resize(&mut self, setup: &mut Setup) -> Result<(), vk::Result> {
        if let Some(previous) = self.target.take() {
            unsafe { setup.ctx.device().device_wait_idle() }?;
            setup.allocator.deallocate_image_view(previous.image_view());
            setup.allocator.deallocate_image(*previous.image());
        }

        let info = ImageInfo::color_target(setup.target.extent, setup.target.format)
            .with_usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .with_name("computed geometry");
        let image = setup.allocator.allocate_image(&info)?;
        let view = setup.allocator.allocate_image_view(
            image.handle(),
            setup.target.format,
            vk::ImageViewType::TYPE_2D,
            image.subresource_range(),
        )?;

        let attachment = ImageAttachment::from_image(&image, vk::ImageLayout::UNDEFINED).with_image_view(view);

        // a module with only the release, enough to put a fresh image into its resting layout
        let mut module = Module::default();
        let target = module.import_attachment(&attachment);
        let ready = module.release(target, RESTING_ACCESS, DomainFlag::Graphics);
        setup.graph.execute_blocking(
            setup.ctx,
            &module.compile(ready),
            &mut AllocatorKind::Persistent(setup.allocator),
        )?;

        let attachment = attachment.with_layout(RESTING_ACCESS.into());
        self.pass = Some(self.compile_pass(&attachment));
        self.target = Some(attachment);

        Ok(())
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
                ui.add(egui::Slider::new(&mut self.ui.count, 1..=MAX_TRIANGLES).text("triangles"));
                ui.separator();
                if ui.button("reset").clicked() {
                    self.ui = UiState::default();
                }
            });
    }

    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result> {
        let target = self.target.as_ref().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let pass = self.pass.as_mut().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let count = self.ui.count.min(MAX_TRIANGLES);

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

        // the whole of this frame's work is writes into an already compiled program: how big the
        // two transients have to be, how much of them the dispatches and the draw cover, and the
        // block both dispatches are handed, whose two addresses the graph fills in
        pass.program.set(pass.triangles, count);
        pass.program.set(pass.vertex_count, count * 3);
        pass.program.set_bytes(
            pass.push,
            &PushConstants {
                count,
                time: if self.ui.animate { frame.elapsed } else { 0.0 },
                scale: ring_scale(frame.target.extent),
                ..Default::default()
            },
        );
        pass.program.set(pass.instances, instances);
        pass.program.set(pass.vertices, vertices);

        frame
            .graph
            .execute(frame.ctx, &pass.program, &mut AllocatorKind::Frame(frame.allocator))?;

        // the program left the target in its resting layout, which is the layout the frame's
        // own module imports it at
        let attachment = frame.module.import_attachment(target);
        frame.module.set_name(attachment, "computed geometry");

        Ok(frame.module.blit(attachment, frame.swapchain_image))
    }

    fn destroy(&mut self, _graph: &mut RenderGraph, allocator: &mut PersistentAllocator) {
        if let Some(target) = self.target.take() {
            allocator.deallocate_image_view(target.image_view());
            allocator.deallocate_image(*target.image());
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<Compute>() }

#[cfg(test)]
mod tests {
    use vir::{IR, Module, Program, VertexAttribute, VertexLayout, graph::ir, resource::shader};

    use super::*;

    /// The two dispatches and the draw the example records, over transients rather than the
    /// slots it binds, which is enough for everything the graph works out about their order and
    /// their usage. Hands back the corner buffer alongside, since which value the draw binds is
    /// the point of one test.
    fn compiled_chain() -> (Program, ValueId) {
        let mut module = Module::default();
        let instances = module.transient_buffer(&transient("instances").with_size(INSTANCE_SIZE));
        let vertices = module.transient_buffer(&transient("computed vertices").with_size(3 * VERTEX_SIZE));

        let placed = module
            .begin_compute()
            .write(instances)
            .push_constant_address(INSTANCES_AT, instances)
            .push_constant_address(VERTICES_AT, vertices)
            .dispatch(1, 1, 1)
            .end_compute();
        let expanded = module
            .begin_compute()
            .write(vertices)
            .read(placed)
            .dispatch(1, 1, 1)
            .end_compute();

        let target = module.transient_image(
            &ImageInfo::color_target(vk::Extent2D::default().width(4).height(4), vk::Format::R8G8B8A8_UNORM)
                .with_usage(vk::ImageUsageFlags::TRANSFER_DST),
        );
        let target = module.clear(target, BACKGROUND);
        let end = module
            .begin_rendering(&[target])
            .bind_vertex_buffer(0, expanded)
            .draw(3, 1)
            .end_rendering();

        (module.compile(end), vertices)
    }

    fn memory_barriers(program: &Program) -> Vec<(Access, Access)> {
        let access = |id: &ValueId| {
            program
                .instructions()
                .iter()
                .find_map(|(instr_id, ir)| match ir {
                    IR::Constant(ir::Constant::Access(access)) if instr_id == id => Some(*access),
                    _ => None,
                })
                .expect("a barrier operand should be an access constant")
        };

        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::MemoryBarrier { src_access, dst_access } => Some((access(src_access), access(dst_access))),
                _ => None,
            })
            .collect()
    }

    /// A region stands in for the first resource it declared, so writing vertices before reading
    /// placements makes `expanded` the buffer the draw binds.
    #[test]
    fn the_expand_region_stands_in_for_the_buffer_the_draw_binds() {
        let (compiled, vertices) = compiled_chain();

        // the expand region is the last one the compiled module opens
        let expand = compiled
            .instructions()
            .iter()
            .rev()
            .find_map(|(_, ir)| match ir {
                IR::BeginCompute { resources, .. } => Some(resources),
                _ => None,
            })
            .expect("the compute regions should be present");
        assert_eq!(expand.first().map(|(id, _)| *id), Some(vertices));
    }

    /// The two barriers the example never writes, and only those two: a buffer the graph
    /// allocated for this run has nothing before it to wait on.
    #[test]
    fn the_graph_orders_the_dispatches_and_the_draw_by_itself() {
        assert_eq!(
            memory_barriers(&compiled_chain().0),
            vec![
                // the expand reads what the place wrote
                (Access::ComputeWrite, Access::ComputeRead),
                // the draw reads what the expand wrote
                (Access::ComputeWrite, Access::AttributeRead),
            ]
        );
    }

    /// Nothing in the example names a buffer usage, so what each transient is created for has to
    /// come out of what the graph does with it.
    #[test]
    fn the_usage_of_each_transient_comes_out_of_the_graph() {
        let compiled = compiled_chain().0;
        let usage_of = |wanted: &str| {
            compiled
                .instructions()
                .iter()
                .find_map(|(_, ir)| match ir {
                    IR::ConstructBuffer { usage, name, .. } if name.as_deref() == Some(wanted) => Some(*usage),
                    _ => None,
                })
                .expect("both transients should be constructed")
        };

        // written by the place, read by the expand, and reached through a pushed address
        assert_eq!(
            usage_of("instances"),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        );
        // the same, and bound as the draw's vertex buffer on top of it
        assert_eq!(
            usage_of("computed vertices"),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::VERTEX_BUFFER
        );
    }

    /// The addresses are left to the graph, so the block the host writes must have room for them
    /// exactly where `compute.slang` reads them.
    #[test]
    fn the_pushed_addresses_sit_where_the_shader_reads_them() {
        assert_eq!(INSTANCES_AT, 0);
        assert_eq!(VERTICES_AT, size_of::<vk::DeviceAddress>() as u32);
        assert!(VERTICES_AT as usize + size_of::<vk::DeviceAddress>() <= size_of::<PushConstants>());

        let push = PushConstants::default();
        assert_eq!(push.instances, 0);
        assert_eq!(push.vertices, 0);
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
