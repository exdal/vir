//! Rendering into an offscreen target and letting the graph move it to the swapchain.
//!
//! Each frame is a chain: clear and draw into a persistent offscreen target, blit that into a
//! downscaled image the graph owns for the frame, then blit that onto the swapchain. It declares
//! no barriers or layouts; the graph reads those off the accesses. The offscreen target outlives
//! the frame, so it is released back into its resting layout and added as a second frame root.

use ash::vk;
use vir::{
    Access,
    AllocatorKind,
    BlendPreset,
    ClearValue,
    DomainFlag,
    ImageAttachment,
    ImageInfo,
    PersistentAllocator,
    PipelineId,
    RasterizationState,
    Rect2D,
    RenderGraph,
    ValueId,
    allocator::Allocator,
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

struct Offscreen {
    pipeline: PipelineId,
    target: Option<ImageAttachment>,
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

impl Example for Offscreen {
    const TITLE: &'static str = "vir: offscreen";

    fn new(setup: &mut Setup) -> Result<Self, vk::Result> {
        let mut this = Self {
            pipeline: graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?,
            target: None,
            ui: UiState::default(),
        };
        this.resize(setup)?;

        Ok(this)
    }

    /// The offscreen target matches the swapchain, so it is thrown away and remade with it.
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

        self.target = Some(attachment.with_layout(RESTING_ACCESS.into()));

        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([300.0, 180.0])
            .show(ctx, |ui| {
                ui.label("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.");
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
        let extent = offscreen.extent();
        let format = offscreen.format();

        let attachment = frame.module.import_attachment(offscreen);
        frame.module.set_name(attachment, "offscreen");

        let hue = if self.ui.animate_clear {
            frame.elapsed * 0.2
        } else {
            0.0
        };
        let attachment = frame.module.clear(attachment, rainbow(hue));

        let attachment = frame
            .module
            .begin_rendering(&[attachment])
            .with_name("offscreen triangle")
            .bind_graphics_pipeline(self.pipeline)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .push_constants(&PushConstants {
                offset: [0.25 * frame.elapsed.sin(), 0.0],
                scale: 1.0,
                tint: 0.0,
            })
            .draw(3, 1)
            .end_rendering();

        // the graph owns this one for as long as the two blits need it
        let scaled = vk::Extent2D {
            width: (extent.width / self.ui.downscale).max(1),
            height: (extent.height / self.ui.downscale).max(1),
        };
        let scratch = frame
            .module
            .transient_image(&ImageInfo::color_target(scaled, format).with_name("downscaled scratch"));
        let scratch = frame.module.blit(attachment, scratch);
        let presented = frame.module.blit(scratch, frame.swapchain_image);

        // present does not depend on the offscreen target's final layout, so its release is a
        // root of its own
        let resting = frame.module.release(attachment, RESTING_ACCESS, DomainFlag::Graphics);
        frame.add_root(resting);

        Ok(presented)
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
