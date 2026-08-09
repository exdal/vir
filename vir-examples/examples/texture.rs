//! Uploading an image to the GPU and sampling it.
//!
//! Startup runs a one-shot module: staging buffer, one copy into the image, then a release into
//! the layout a shader read wants. After that the image is a slot in the graph's bindless table,
//! so a draw names it by pushing an index instead of binding a descriptor set.

use ash::vk;
use vir::{
    Access,
    AllocatorKind,
    BlendPreset,
    BufferInfo,
    ClearValue,
    DomainFlag,
    Image,
    ImageAttachment,
    ImageInfo,
    PersistentAllocator,
    PipelineId,
    RasterizationState,
    Rect2D,
    RenderGraph,
    SamplerInfo,
    TextureId,
    ValueId,
    allocator::Allocator,
};
use vir_examples::{Example, Frame, Setup, graphics_pipeline};

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/texture.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/texture.frag.spv"));

const PNG: &[u8] = include_bytes!("../assets/checkerboard.png");

/// sRGB pixels, so an `_SRGB` format samples back linear.
const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// Between frames the image rests in the layout a fragment sample wants.
const RESTING_ACCESS: Access = Access::FragmentSampled;

const BACKGROUND: ClearValue = ClearValue::rgba_f32(0.02, 0.02, 0.05, 1.0);

/// The block `texture.slang` declares.
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    scale: [f32; 2],
    texture_index: u32,
}

struct Texture {
    pipeline: PipelineId,
    image: Image,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
    /// The image's slot in the bindless table, pushed by the draw.
    slot: TextureId,
    extent: vk::Extent2D,
    ui: UiState,
}

struct UiState {
    zoom: f32,
    keep_aspect: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            zoom: 0.8,
            keep_aspect: true,
        }
    }
}

impl Texture {
    fn attachment(&self) -> ImageAttachment {
        ImageAttachment::from_image(&self.image, RESTING_ACCESS.into()).with_image_view(self.image_view)
    }

    /// Per-axis shrink that fits the image into `target` without stretching.
    fn quad_scale(&self, target: vk::Extent2D) -> [f32; 2] {
        if !self.ui.keep_aspect {
            return [self.ui.zoom, self.ui.zoom];
        }

        let image = self.extent.width as f32 / self.extent.height as f32;
        let screen = target.width.max(1) as f32 / target.height.max(1) as f32;
        if image > screen {
            [self.ui.zoom, self.ui.zoom * screen / image]
        } else {
            [self.ui.zoom * image / screen, self.ui.zoom]
        }
    }
}

impl Example for Texture {
    const TITLE: &'static str = "vir: texture";

    fn new(setup: &mut Setup) -> Result<Self, vk::Result> {
        let pipeline = graphics_pipeline(setup.graph, VERT_SPV, FRAG_SPV)?;

        let decoded = image::load_from_memory_with_format(PNG, image::ImageFormat::Png)
            .expect("the embedded asset is not a readable PNG")
            .to_rgba8();
        let extent = vk::Extent2D {
            width: decoded.width(),
            height: decoded.height(),
        };

        let image = setup
            .allocator
            .allocate_image(&ImageInfo::texture(extent, TEXTURE_FORMAT).with_name("vir logo"))?;
        let image_view = setup.allocator.allocate_image_view(
            image.handle(),
            TEXTURE_FORMAT,
            vk::ImageViewType::TYPE_2D,
            image.subresource_range(),
        )?;
        let sampler = setup.allocator.allocate_sampler(
            &SamplerInfo::default()
                .with_mag_filter(vk::Filter::LINEAR)
                .with_min_filter(vk::Filter::LINEAR)
                .with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        )?;
        let slot = setup.graph.register_texture(image_view, sampler);

        let pixels = decoded.into_raw();
        let mut staging = setup.allocator.allocate_buffer(
            &BufferInfo::staging(size_of_val(pixels.as_slice()) as u64).with_name("texture staging"),
        )?;
        staging.write(0, &pixels)?;

        // one module run to completion: the copy, then the release into the resting layout
        let mut module = vir::Module::default();
        let source = module.import_buffer(&staging, vir::Access::HostWrite);
        let destination = module.import_attachment(&ImageAttachment::from_image(&image, vk::ImageLayout::UNDEFINED));
        module.set_name(source, "texture staging");
        module.set_name(destination, "vir logo");
        let uploaded = module.copy_buffer_to_image(source, destination);
        let ready = module.release(uploaded, RESTING_ACCESS, DomainFlag::Graphics);
        setup.graph.execute_blocking(
            setup.ctx,
            &module.compile(ready),
            &mut AllocatorKind::Persistent(setup.allocator),
        )?;
        setup.allocator.deallocate_buffer(staging);

        Ok(Self {
            pipeline,
            image,
            image_view,
            sampler,
            slot,
            extent,
            ui: UiState::default(),
        })
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([300.0, 180.0])
            .show(ctx, |ui| {
                ui.label(format!(
                    "{}x{} uploaded once, sampled through bindless slot {}",
                    self.extent.width, self.extent.height, self.slot.0
                ));
                ui.checkbox(&mut self.ui.keep_aspect, "keep aspect");
                ui.add(egui::Slider::new(&mut self.ui.zoom, 0.05..=1.0).text("zoom"));
                ui.separator();
                if ui.button("reset").clicked() {
                    self.ui = UiState::default();
                }
            });
    }

    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result> {
        let target = frame.module.clear(frame.swapchain_image, BACKGROUND);

        let texture = frame.module.import_attachment(&self.attachment());
        frame.module.set_name(texture, "sample texture");

        Ok(frame
            .module
            .begin_rendering(&[target])
            .with_name("textured quad")
            .bind_graphics_pipeline(self.pipeline)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .set_primitive_topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .sample_image(texture)
            .push_constants(&PushConstants {
                scale: self.quad_scale(frame.target.extent),
                texture_index: self.slot.0,
            })
            .draw(4, 1)
            .end_rendering())
    }

    fn destroy(&mut self, graph: &mut RenderGraph, allocator: &mut PersistentAllocator) {
        graph.unregister_texture(self.slot);
        allocator.deallocate_image_view(self.image_view);
        allocator.deallocate_image(self.image);
        allocator.deallocate_sampler(self.sampler);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> { vir_examples::run::<Texture>() }

#[cfg(test)]
mod tests {
    use vir::{DescriptorBinding, resource::shader};
    use vir_examples::read_spirv;

    use super::*;

    /// The bindless table is a variable-count combined image sampler array; a shader that
    /// reflects as anything else silently samples nothing.
    #[test]
    fn the_fragment_shader_declares_the_bindless_texture_table() {
        let reflection = shader::reflect(&read_spirv(FRAG_SPV)).expect("shader should reflect");

        assert_eq!(
            reflection.bindings,
            vec![DescriptorBinding {
                set: 0,
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
                variable_count: true,
            }]
        );

        // and the vertex stage stays out of it, so the table is fragment-only
        let vertex = shader::reflect(&read_spirv(VERT_SPV)).expect("shader should reflect");
        assert!(vertex.bindings.is_empty());
    }

    /// The vertex stage reads scale and the fragment stage the texture index, so one range
    /// covers both stages.
    #[test]
    fn the_push_constant_range_covers_both_stages() {
        let reflect = |spirv| shader::reflect(&read_spirv(spirv)).expect("shader should reflect");

        let ranges = vir::resource::pipeline::push_constant_ranges(&[reflect(VERT_SPV), reflect(FRAG_SPV)]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(ranges[0].offset, 0);
        assert_eq!(ranges[0].size as usize, size_of::<PushConstants>());
    }

    /// The resting access maps to the layout a sample wants.
    #[test]
    fn the_resting_access_is_the_layout_a_sample_wants() {
        assert_eq!(
            vk::ImageLayout::from(RESTING_ACCESS),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }
}
