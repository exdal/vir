//! The smallest thing `vir` draws: a triangle whose geometry lives in the shader, so the whole
//! pipeline comes out of reflection and a draw needs no buffer at all.
//!
//! It renders twice to show what a pass costs to vary: once with the viewport and scissor
//! baked into the pipeline, and once with both promoted to dynamic state and pointed at a
//! corner of the framebuffer.

use ash::vk;
use vir::{BlendPreset, ClearValue, DynamicStateFlags, PipelineId, RasterizationState, Rect2D, ValueId};
use vir_examples::{Example, Frame, Setup, graphics_pipeline};

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));

/// The block `triangle.slang` declares, laid out to match it member for member.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PushConstants {
    offset: [f32; 2],
    scale: f32,
    tint: f32,
}

/// Where `tint` sits in the block, for the draws that push nothing else.
const TINT_OFFSET: u32 = 12;

const BACKGROUND: ClearValue = ClearValue::rgba_f32(0.02, 0.02, 0.05, 1.0);

struct Triangle {
    pipeline: PipelineId,
    ui: UiState,
}

/// What the panel is driving.
struct UiState {
    slide: bool,
    scale: f32,
    tint: f32,
    corner: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            slide: true,
            scale: 1.0,
            tint: 0.0,
            corner: true,
        }
    }
}

impl Example for Triangle {
    const TITLE: &'static str = "vir: triangle";

    fn new(setup: &mut Setup) -> Result<Self, vk::Result> {
        Ok(Self {
            pipeline: graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?,
            ui: UiState::default(),
        })
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([280.0, 180.0])
            .show(ctx, |ui| {
                ui.label("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.");
                ui.checkbox(&mut self.ui.slide, "slide");
                ui.checkbox(&mut self.ui.corner, "corner triangle");
                ui.add(egui::Slider::new(&mut self.ui.scale, 0.1..=2.0).text("scale"));
                ui.add(egui::Slider::new(&mut self.ui.tint, 0.0..=1.0).text("tint"));
                ui.separator();
                if ui.button("reset").clicked() {
                    self.ui = UiState::default();
                }
            });
    }

    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result> {
        let elapsed = frame.elapsed;
        let target = frame.module.clear(frame.swapchain_image, BACKGROUND);

        let sliding = PushConstants {
            offset: [if self.ui.slide { 0.25 * elapsed.sin() } else { 0.0 }, 0.0],
            scale: self.ui.scale,
            tint: self.ui.tint,
        };

        let target = frame
            .module
            .begin_rendering(&[target])
            .with_name("sliding triangle")
            .bind_graphics_pipeline(self.pipeline)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .push_constants(&sliding)
            .draw(3, 1)
            .end_rendering();

        if !self.ui.corner {
            return Ok(target);
        }

        // the same pipeline again, but with the viewport and the scissor set per draw rather
        // than baked in, which is what keeps this from being a second pipeline
        Ok(frame
            .module
            .begin_rendering(&[target])
            .with_name("corner triangle")
            .bind_graphics_pipeline(self.pipeline)
            .set_dynamic_state(DynamicStateFlags::Viewport | DynamicStateFlags::Scissor)
            .set_viewport(0, Rect2D::relative(0.5, 0.5, 0.5, 0.5))
            .set_scissor(0, Rect2D::relative(0.5, 0.5, 0.5, 0.5))
            .broadcast_color_blend(BlendPreset::AlphaBlend)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .push_constants(&PushConstants {
                scale: self.ui.scale * (0.5 + 0.5 * elapsed.cos()),
                ..sliding
            })
            .push_constants_at(TINT_OFFSET, &self.ui.tint)
            .draw(3, 1)
            .end_rendering())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<Triangle>() }

#[cfg(test)]
mod tests {
    use vir::{
        VertexLayout,
        resource::{pipeline::push_constant_ranges, shader},
    };
    use vir_examples::read_spirv;

    use super::*;

    /// The SV_VertexID triangle must keep compiling to a pipeline with no vertex input at all.
    #[test]
    fn the_vertex_id_shader_reflects_no_attributes() {
        let reflections = [shader::reflect(&read_spirv(VERT_SPV)).expect("vertex shader should reflect")];
        assert_eq!(VertexLayout::interleaved(&reflections), VertexLayout::default());
    }

    /// What `push_constants` sends has to be the block Slang laid out, or the shader reads
    /// whatever the neighbouring member happened to be.
    #[test]
    fn the_reflected_push_constant_block_matches_the_pushed_struct() {
        for spirv in [VERT_SPV, FRAG_SPV] {
            let reflection = shader::reflect(&read_spirv(spirv)).expect("shader should reflect");
            assert_eq!(reflection.push_constant_offset, 0);
            assert_eq!(reflection.push_constant_size as usize, size_of::<PushConstants>());
        }

        // TINT_OFFSET has to name the last member, since the corner draw pushes it alone
        assert_eq!(TINT_OFFSET as usize, size_of::<PushConstants>() - size_of::<f32>());
    }

    /// Both stages read the block, so one range covers them both and spans the whole struct.
    #[test]
    fn the_push_constant_range_covers_both_stages() {
        let reflect = |spirv| shader::reflect(&read_spirv(spirv)).expect("shader should reflect");

        let ranges = push_constant_ranges(&[reflect(VERT_SPV), reflect(FRAG_SPV)]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(ranges[0].offset, 0);
        assert_eq!(ranges[0].size as usize, size_of::<PushConstants>());
    }
}
