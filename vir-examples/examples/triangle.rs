//! A triangle with its geometry in the shader, so the pipeline comes from reflection and the
//! draw binds no buffer.
//!
//! It draws twice: once with the viewport and scissor baked into the pipeline, and once with
//! both set as dynamic state and aimed at a corner of the framebuffer.

use ash::vk;
use vir::{
    BlendPreset,
    ClearValue,
    DynamicStateFlags,
    ImageAttachment,
    Module,
    PipelineId,
    RasterizationState,
    Rect2D,
    ValueId,
    allocator::Allocator,
};
use vir_examples::{Example, Frame, Setup, graphics_pipeline};

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));

/// Matches the push constant block in `triangle.slang`, member for member.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PushConstants {
    offset: [f32; 2],
    scale: f32,
    tint: f32,
}

/// Byte offset of `tint` in the block.
const TINT_OFFSET: u32 = 12;

const BACKGROUND: ClearValue = ClearValue::rgba_f32(0.02, 0.02, 0.05, 1.0);

const RESTING_ACCESS: vir::Access = vir::Access::BlitRead;

struct Pass {
    program: vir::Program,
    has_corner_triangle: ValueId,
    push: ValueId,
}

struct Triangle {
    pipeline: PipelineId,
    ui: UiState,
    target: Option<ImageAttachment>,
    pass: Option<Pass>,
}

impl Triangle {
    fn compile_pass(&self, target: &ImageAttachment) -> Pass {
        let mut module = Module::default();

        let has_corner_triangle = module.declare_bool_var("has corner", true);
        let push = module.declare_bytes_var("push constants", size_of::<PushConstants>() as u32);

        let attachment = module.import_attachment(target);
        module.set_name(attachment, "triangles");
        let attachment = module.clear(attachment, BACKGROUND);

        let sliding_tri = module
            .begin_rendering(&[attachment])
            .with_name("sliding triangle")
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

        let target = module.set_condition(
            has_corner_triangle,
            |m| {
                m.begin_rendering(&[sliding_tri])
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
                    .push_constants_from(push)
                    .push_constants_at(TINT_OFFSET, &self.ui.tint)
                    .draw(3, 1)
                    .end_rendering()
            },
            |_| sliding_tri,
        );
        let target = module.release(target, RESTING_ACCESS, vir::DomainFlag::Graphics);
        let program = module.compile(target);
        vir_examples::dump_ir("triangle", &program);

        Pass {
            program,
            has_corner_triangle,
            push,
        }
    }
}

/// State the UI drives.
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
            pass: None,
            target: None,
        })
    }

    fn resize(&mut self, setup: &mut Setup) -> Result<(), vk::Result> {
        if let Some(previous) = self.target.take() {
            unsafe { setup.ctx.device().device_wait_idle() }?;
            setup.allocator.deallocate_image_view(previous.image_view());
            setup.allocator.deallocate_image(*previous.image());
        }

        let info = vir::ImageInfo::color_target(setup.target.extent, setup.target.format)
            .with_usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .with_name("triangles");
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
        let ready = module.release(target, RESTING_ACCESS, vir::DomainFlag::Graphics);
        setup.graph.execute_blocking(
            setup.ctx,
            &module.compile(ready),
            &mut vir::AllocatorKind::Persistent(setup.allocator),
        )?;

        let attachment = attachment.with_layout(RESTING_ACCESS.into());
        self.pass = Some(self.compile_pass(&attachment));
        self.target = Some(attachment);

        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([280.0, 180.0])
            .show(ctx, |ui| {
                ui.label(
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut \
                     labore et dolore magna aliqua.",
                );
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
        let target = self.target.as_ref().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let pass = self.pass.as_mut().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;

        let elapsed = frame.elapsed;
        pass.program.set_bytes(
            pass.push,
            &PushConstants {
                offset: [if self.ui.slide { 0.25 * elapsed.sin() } else { 0.0 }, 0.0],
                scale: self.ui.scale,
                tint: self.ui.tint,
            },
        );
        pass.program.set(pass.has_corner_triangle, self.ui.corner);

        frame.graph.execute(
            frame.ctx,
            &pass.program,
            &mut vir::AllocatorKind::Frame(frame.allocator),
        )?;

                let attachment = frame.module.import_attachment(target);
        frame.module.set_name(attachment, "triangles");

        Ok(frame.module.blit(attachment, frame.swapchain_image))
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

    /// The pushed struct must match the block Slang reflected, or its members misalign.
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
