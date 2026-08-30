//! A render graph compiled once and re-run every frame.
//!
//! The pass clears and draws into an offscreen target, blits that through a downscaled
//! intermediate, and blits the result onto the image the frame acquired. Both are images the
//! graph owns for the run, since nothing outside it needs them once the blit is done.
//!
//! Nothing about it is rebuilt per frame: the clear colour, the intermediate's size and the
//! pushed block are variables the program reads when it runs, and whether to animate the clear
//! at all is a branch on a variable rather than a Rust `if` around the builder. The
//! intermediate is sized by one of those variables, which is where the pixelation comes from,
//! so the graph allocates a different image every time the slider moves without anything being
//! compiled again.

use ash::vk;
use vir::{Access, BlendPreset, ClearValue, ImageInfo, PipelineId, RasterizationState, Rect2D, ValueId, clear};
use vir_examples::{Example, Frame, Recording, Setup, graphics_pipeline};

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));

/// The block `triangle.slang` declares.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PushConstants {
    offset: [f32; 2],
    scale: f32,
    tint: f32,
}

/// The slots the frame writes into the compiled pass.
struct Slots {
    clear_color: ValueId,
    animate: ValueId,
    scratch_extent: ValueId,
    push: ValueId,
}

struct Offscreen {
    pipeline: PipelineId,
    slots: Option<Slots>,
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
        Ok(Self {
            pipeline: graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?,
            slots: None,
            ui: UiState::default(),
        })
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

    fn record(&mut self, recording: &mut Recording) -> Result<ValueId, vk::Result> {
        let module = &mut *recording.module;
        let target = recording.target;

        let clear_color = module.declare_clear_var("clear color", clear::f32::BLACK);
        let animate = module.declare_bool_var("animate clear", true);
        let scratch_extent = module.declare_extent_3d_var("scratch extent", extent3d(target.extent, 1));
        let push = module.declare_bytes_var("triangle push block", size_of::<PushConstants>() as u32);

        let info = ImageInfo::color_target(target.extent, target.format)
            .with_usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .with_name("offscreen");
        let attachment = module.transient_image(&info);

        // which colour to clear to is a branch rather than a Rust `if`, so turning the
        // animation off does not mean compiling a second graph
        let attachment = module.set_condition(
            animate,
            |m| m.clear_from(attachment, clear_color),
            |m| m.clear(attachment, clear::f32::BLACK),
        );

        let attachment = module
            .begin_rendering([(attachment, Access::ColorRW)])
            .with_name("offscreen triangle")
            .bind_graphics_pipeline(self.pipeline)
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

        // down and back up again through a second image the graph owns for the run, which is
        // where the pixelation comes from; how far down is the variable
        let scratch = ImageInfo::color_target(target.extent, target.format)
            .with_usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .with_name("downscaled scratch");
        let scratch = module.transient_image_sized(&scratch, scratch_extent);
        let scratch = module.blit(attachment, scratch);
        let attachment = module.blit(scratch, attachment);

        self.slots = Some(Slots {
            clear_color,
            animate,
            scratch_extent,
            push,
        });

        Ok(module.blit(attachment, recording.swapchain_image))
    }

    fn update(&mut self, frame: &mut Frame) -> Result<(), vk::Result> {
        let Some(slots) = self.slots.as_ref() else {
            return Ok(());
        };

        // the whole of this frame is four writes into an already compiled program, with no
        // graph rebuilt and no pipeline recompiled
        let extent = frame.target.extent;
        frame.program.set(slots.animate, self.ui.animate_clear);
        frame.program.set(slots.clear_color, rainbow(frame.elapsed * 0.2));
        frame.program.set(
            slots.scratch_extent,
            extent3d(
                vk::Extent2D {
                    width: (extent.width / self.ui.downscale).max(1),
                    height: (extent.height / self.ui.downscale).max(1),
                },
                1,
            ),
        );
        frame.program.set_bytes(
            slots.push,
            &PushConstants {
                offset: [0.25 * frame.elapsed.sin(), 0.0],
                scale: 1.0,
                tint: 0.0,
            },
        );

        Ok(())
    }
}

fn extent3d(extent: vk::Extent2D, depth: u32) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width,
        height: extent.height,
        depth,
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

    /// The downscale divides the target's extent, and a zero-sized image cannot be created, so
    /// the divisor the panel offers has to leave at least one texel behind.
    #[test]
    fn the_smallest_downscale_still_leaves_an_image() {
        let extent = vk::Extent2D { width: 1, height: 1 };
        let scaled = extent3d(
            vk::Extent2D {
                width: (extent.width / 16).max(1),
                height: (extent.height / 16).max(1),
            },
            1,
        );

        assert_eq!(scaled.width, 1);
        assert_eq!(scaled.height, 1);
        assert_eq!(scaled.depth, 1);
    }
}
