//! A render graph compiled once and re-run every frame.
//!
//! The offscreen pass is a `Program` built in `new`: it clears and draws into a persistent
//! target, blits that through a downscaled intermediate the graph owns for the frame, and
//! releases the target into its resting layout. Nothing about it is rebuilt per frame — the
//! clear colour, the intermediate's size and the pushed block are variables the program reads
//! when it runs, and whether to clear at all is a branch on a variable rather than a Rust
//! `if` around the builder.
//!
//! The frame itself only blits that target onto the swapchain, which is what the harness's
//! per-frame module does alongside the UI.

use ash::vk;
use vir::{
    Access,
    AllocatorKind,
    BlendPreset,
    ClearValue,
    DomainFlag,
    ImageAttachment,
    ImageInfo,
    Module,
    PersistentAllocator,
    PipelineId,
    Program,
    RasterizationState,
    Rect2D,
    RenderGraph,
    ValueId,
    allocator::Allocator,
    clear,
};
use vir_examples::{Example, Frame, Setup, graphics_pipeline};

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));

/// The frame's last use of the target is a blit out of it, so that is its resting layout.
const RESTING_ACCESS: Access = Access::BlitRead;

/// The block `triangle.slang` declares.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PushConstants {
    offset: [f32; 2],
    scale: f32,
    tint: f32,
}

/// The compiled offscreen pass and the slots the frame writes into it.
struct Pass {
    program: Program,
    clear_color: ValueId,
    animate: ValueId,
    scratch_extent: ValueId,
    push: ValueId,
}

struct Offscreen {
    pipeline: PipelineId,
    target: Option<ImageAttachment>,
    pass: Option<Pass>,
    ui: UiState,
}

struct UiState {
    /// Divisor for the size of the intermediate the frame blits through.
    downscale: u32,
    animate_clear: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            downscale: 2,
            animate_clear: true,
        }
    }
}

/// Fully saturated rainbow color for `hue` in [0, 1).
fn rainbow(hue: f32) -> ClearValue {
    let h = hue.rem_euclid(1.0) * 6.0;
    let sector = h as u32;
    let f = h - sector as f32;
    let (r, g, b) = match sector {
        0 => (1.0, f, 0.0),
        1 => (1.0 - f, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, 1.0 - f, 1.0),
        4 => (f, 0.0, 1.0),
        _ => (1.0, 0.0, 1.0 - f),
    };

    ClearValue::rgba_f32(r, g, b, 1.0)
}

impl Offscreen {
    /// Records the offscreen pass once. Everything that changes between frames is a variable
    /// the compiled program reads when it runs, so this is not called again until the
    /// swapchain is remade and the target it draws into is a different image.
    fn compile_pass(target: &ImageAttachment, pipeline: PipelineId) -> Pass {
        let mut module = Module::default();

        let clear_color = module.declare_clear_var("clear color", clear::f32::BLACK);
        let animate = module.declare_bool_var("animate clear", true);
        let scratch_extent = module.declare_extent_3d_var("scratch extent", target.extent());
        let push = module.declare_bytes_var("triangle push block", size_of::<PushConstants>() as u32);

        let attachment = module.import_attachment(target);
        module.set_name(attachment, "offscreen");

        // which colour to clear to is a branch rather than a Rust `if`, so turning the
        // animation off does not mean compiling a second graph
        let attachment = module.set_condition(
            animate,
            |m| m.clear_from(attachment, clear_color),
            |m| m.clear(attachment, clear::f32::BLACK),
        );

        let attachment = module
            .begin_rendering(&[attachment])
            .with_name("offscreen triangle")
            .bind_graphics_pipeline(pipeline)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .push_constants_from(push)
            .draw(3, 1)
            .end_rendering();

        // down and back up again through an image the graph owns for the frame, which is where
        // the pixelation comes from; the intermediate is sized by a variable
        let info = ImageInfo::color_target(vk::Extent2D::default(), target.format())
            .with_usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .with_extent3d(target.extent())
            .with_name("downscaled scratch");
        let scratch = module.transient_image(&info);
        let scratch = module.blit(attachment, scratch);
        let attachment = module.blit(scratch, attachment);

        let ready = module.release(attachment, RESTING_ACCESS, DomainFlag::Graphics);
        let program = module.compile(ready);
        vir_examples::dump_ir("offscreen", &program);

        Pass {
            program,
            clear_color,
            animate,
            scratch_extent,
            push,
        }
    }
}

impl Example for Offscreen {
    const TITLE: &'static str = "vir: offscreen";

    fn new(setup: &mut Setup) -> Result<Self, vk::Result> {
        let mut this = Self {
            pipeline: graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?,
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
            .with_name("offscreen target");
        let image = setup.allocator.allocate_image(&info)?;
        let view = setup.allocator.allocate_image_view(
            image.handle(),
            setup.target.format,
            vk::ImageViewType::TYPE_2D,
            image.subresource_range(),
        )?;

        let attachment = ImageAttachment::from_image(&image, vk::ImageLayout::UNDEFINED).with_image_view(view);

        // a module with only the release, enough to put a fresh image into its resting layout
        let mut module = vir::Module::default();
        let target = module.import_attachment(&attachment);
        let ready = module.release(target, RESTING_ACCESS, DomainFlag::Graphics);
        setup.graph.execute_blocking(
            setup.ctx,
            &module.compile(ready),
            &mut AllocatorKind::Persistent(setup.allocator),
        )?;

        let attachment = attachment.with_layout(RESTING_ACCESS.into());
        self.pass = Some(Self::compile_pass(&attachment, self.pipeline));
        self.target = Some(attachment);

        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([300.0, 180.0])
            .show(ctx, |ui| {
                ui.label(
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut \
                     labore et dolore magna aliqua.",
                );
                ui.checkbox(&mut self.ui.animate_clear, "animate the clear");
                ui.add(egui::Slider::new(&mut self.ui.downscale, 1..=16).text("downscale"));
                ui.separator();
                if ui.button("reset").clicked() {
                    self.ui = UiState::default();
                }
            });
    }

    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result> {
        let offscreen = self.target.as_ref().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let pass = self.pass.as_mut().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;

        // the whole of this frame's offscreen work is four writes into an already compiled
        // program, with no graph rebuilt and no pipeline recompiled
        let extent = offscreen.extent();
        pass.program.set(pass.animate, self.ui.animate_clear);
        pass.program.set(pass.clear_color, rainbow(frame.elapsed * 0.2));
        pass.program.set(
            pass.scratch_extent,
            vk::Extent3D {
                width: (extent.width / self.ui.downscale).max(1),
                height: (extent.height / self.ui.downscale).max(1),
                depth: 1,
            },
        );
        pass.program.set_bytes(
            pass.push,
            &PushConstants {
                offset: [0.25 * frame.elapsed.sin(), 0.0],
                scale: 1.0,
                tint: 0.0,
            },
        );

        frame
            .graph
            .execute(frame.ctx, &pass.program, &mut AllocatorKind::Frame(frame.allocator))?;

        // the program left the target in its resting layout, which is the layout the frame's
        // own module imports it at
        let attachment = frame.module.import_attachment(offscreen);
        frame.module.set_name(attachment, "offscreen");

        Ok(frame.module.blit(attachment, frame.swapchain_image))
    }

    fn destroy(&mut self, _graph: &mut RenderGraph, allocator: &mut PersistentAllocator) {
        if let Some(target) = self.target.take() {
            allocator.deallocate_image_view(target.image_view());
            allocator.deallocate_image(*target.image());
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<Offscreen>() }

#[cfg(test)]
mod tests {
    use vir::resource::shader;
    use vir_examples::read_spirv;

    use super::*;

    /// Pushes the same block `triangle.slang` declares, so the struct must match reflection.
    #[test]
    fn the_reflected_push_constant_block_matches_the_pushed_struct() {
        let reflection = shader::reflect(&read_spirv(VERT_SPV)).expect("shader should reflect");
        assert_eq!(reflection.push_constant_offset, 0);
        assert_eq!(reflection.push_constant_size as usize, size_of::<PushConstants>());
    }

    /// The blit out of the target decides its resting layout, so the two must agree.
    #[test]
    fn the_resting_access_is_the_layout_the_frame_leaves_it_in() {
        assert_eq!(
            vk::ImageLayout::from(RESTING_ACCESS),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
    }
}
