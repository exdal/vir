use std::{collections::HashMap, sync::Arc};

use ash::vk;
use egui::{
    TextureFilter,
    TextureOptions,
    TextureWrapMode,
    TexturesDelta,
    epaint::{ClippedPrimitive, ImageData, ImageDelta, Primitive, Vertex as EguiVertex},
};
use vir::{
    AllocatorKind,
    BlendPreset,
    Buffer,
    BufferImageCopy,
    BufferInfo,
    Context,
    DynamicStateFlags,
    FrameAllocator,
    GraphicsPipelineInfo,
    Image,
    ImageAttachment,
    ImageInfo,
    MemoryLocation,
    Module,
    PassCallback,
    PersistentAllocator,
    PipelineId,
    Program,
    RasterizationState,
    Rect2D,
    RenderGraph,
    SamplerInfo,
    ValueId,
    allocator::Allocator,
};

/// The block `egui.slang` declares.
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    screen_size: [f32; 2],
    texture_index: u32,
}

/// Where `texture_index` sits in the block, since a draw pushes it on its own.
const TEXTURE_INDEX_OFFSET: u32 = 8;

/// egui hands over `Color32`, which is sRGB with premultiplied alpha, so an `_SRGB` format is
/// what makes a sample come back linear the way the shader expects.
const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// A texture is uploaded into this layout and only ever sampled afterwards, so the frame never
/// has to transition one.
const TEXTURE_RESTING: vir::Access = vir::Access::FragmentSampled;

/// One of egui's textures as the GPU holds it.
struct Texture {
    image: Image,
    image_view: vk::ImageView,
    /// The texture's slot in the graph's bindless table, which is what a draw pushes.
    slot: vir::TextureId,
    extent: vk::Extent2D,
    options: TextureOptions,
    /// Whether anything has landed in it yet, which is what says its contents are worth
    /// preserving across the next patch.
    uploaded: bool,
}

impl Texture {
    fn attachment(&self, layout: vk::ImageLayout) -> ImageAttachment {
        ImageAttachment::from_image(&self.image, layout).with_image_view(self.image_view)
    }
}

/// One primitive's slice of the frame's shared buffers.
///
/// The texture is already resolved to the bindless slot the draw pushes: the body that records
/// this runs with nothing but what it was handed, and looking a texture up again there would
/// mean sharing the pass's texture table with it.
#[derive(Clone, Copy)]
struct MeshDraw {
    clip: vk::Rect2D,
    slot: u32,
    index_offset: u32,
    index_count: u32,
    vertex_offset: i32,
}

/// One texture patch staged in host memory, waiting for the copy that lands it.
struct Upload {
    texture: egui::TextureId,
    staging: Buffer,
    region: BufferImageCopy,
}

/// A frame's worth of tessellated geometry, already staged and ready to be bound into the
/// compiled pass.
pub struct EguiFrame {
    vertices: Buffer,
    indices: Buffer,
    /// Shared with the body that records them, which outlives this.
    draws: Arc<[MeshDraw]>,
    uploads: usize,
    screen_size_in_points: [f32; 2],
}

impl EguiFrame {
    pub fn draw_count(&self) -> usize { self.draws.len() }

    pub fn upload_count(&self) -> usize { self.uploads }
}

/// The slots the compiled pass reads when it runs. What the UI looks like is all of it: the
/// pass itself is compiled once, alongside whatever the example draws under it.
pub struct EguiSlots {
    has_draws: ValueId,
    vertices: ValueId,
    indices: ValueId,
    push: ValueId,
    body: ValueId,
}

pub struct EguiPass {
    pipeline: PipelineId,
    textures: HashMap<egui::TextureId, Texture>,
    samplers: HashMap<TextureOptions, vk::Sampler>,
    /// Textures egui freed last frame, which the frame it freed them in may still have
    /// painted with.
    pending_free: Vec<egui::TextureId>,
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

fn filter(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Nearest => vk::Filter::NEAREST,
        TextureFilter::Linear => vk::Filter::LINEAR,
    }
}

fn address_mode(wrap: TextureWrapMode) -> vk::SamplerAddressMode {
    match wrap {
        TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
        TextureWrapMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

impl EguiPass {
    pub fn new(graph: &mut RenderGraph, vertex_spirv: &[u32], fragment_spirv: &[u32]) -> Result<Self, vk::Result> {
        let pipeline = graph.declare_pipeline(
            GraphicsPipelineInfo::new()
                .with_shader(vertex_spirv)
                .with_shader(fragment_spirv),
        )?;

        Ok(Self {
            pipeline,
            textures: HashMap::new(),
            samplers: HashMap::new(),
            pending_free: Vec::new(),
        })
    }

    /// Hands every texture and sampler back. The device has to be idle first.
    pub fn destroy(&mut self, graph: &mut RenderGraph, allocator: &mut PersistentAllocator) {
        for (_, texture) in self.textures.drain() {
            graph.unregister_texture(texture.slot);
            allocator.deallocate_image_view(texture.image_view);
            allocator.deallocate_image(texture.image);
        }

        for (_, sampler) in self.samplers.drain() {
            allocator.deallocate_sampler(sampler);
        }
    }

    fn sampler(
        &mut self, allocator: &mut PersistentAllocator, options: TextureOptions,
    ) -> Result<vk::Sampler, vk::Result> {
        if let Some(sampler) = self.samplers.get(&options) {
            return Ok(*sampler);
        }

        let sampler = allocator.allocate_sampler(
            &SamplerInfo::default()
                .with_mag_filter(filter(options.magnification))
                .with_min_filter(filter(options.minification))
                .with_address_mode(address_mode(options.wrap_mode)),
        )?;
        self.samplers.insert(options, sampler);

        Ok(sampler)
    }

    /// Turns this frame's texture deltas into staged patches, creating whatever images they
    /// name along the way.
    ///
    /// egui frees a texture *after* the frame that painted with it, so a free is held over to
    /// the next call rather than acted on here. Whatever does get destroyed is destroyed
    /// behind a wait, since work submitted for earlier frames may still be sampling it.
    ///
    /// The deltas are taken by value because handling them is the whole contract: egui asserts
    /// on a `TexturesDelta` that is dropped with anything left in it.
    fn stage_textures(
        &mut self, ctx: &Context, graph: &mut RenderGraph, persistent: &mut PersistentAllocator,
        allocator: &mut FrameAllocator, mut textures_delta: TexturesDelta,
    ) -> Result<Vec<Upload>, vk::Result> {
        let mut uploads = Vec::new();
        let mut retired = Vec::new();

        for id in std::mem::take(&mut self.pending_free) {
            match self.textures.remove(&id) {
                Some(texture) => retired.push(texture),
                None => tracing::warn!(?id, "egui freed a texture that was never uploaded"),
            }
        }

        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                let ImageData::Color(image) = &delta.image;
                let size = vk::Extent2D {
                    width: image.size[0] as u32,
                    height: image.size[1] as u32,
                };
                if size.width == 0 || size.height == 0 {
                    continue;
                }

                if !self.prepare_target(graph, persistent, &mut retired, *id, delta, size)? {
                    continue;
                }

                let mut staging =
                    allocator.allocate_buffer(&BufferInfo::staging(size_of_val(image.pixels.as_slice()) as u64))?;
                staging.write(0, &image.pixels)?;

                let offset = match delta.pos {
                    Some([x, y]) => vk::Offset2D {
                        x: x as i32,
                        y: y as i32,
                    },
                    None => vk::Offset2D::default(),
                };

                uploads.push(Upload {
                    texture: *id,
                    staging,
                    region: BufferImageCopy::region(offset, size),
                });
            }
        }

        self.pending_free.extend(textures_delta.free.iter().copied());
        textures_delta.clear();

        if !retired.is_empty() {
            // the only thing that says an earlier frame is done sampling these
            unsafe { ctx.device().device_wait_idle() }?;
            for texture in retired {
                graph.unregister_texture(texture.slot);
                persistent.deallocate_image_view(texture.image_view);
                persistent.deallocate_image(texture.image);
            }
        }

        Ok(uploads)
    }

    /// Makes sure `id` names an image this delta can be copied into, pushing whatever image
    /// had to be replaced to get there onto `retired`.
    ///
    /// A whole delta that changes the size or the sampling of a texture replaces it; a partial
    /// one only ever patches what is already there.
    fn prepare_target(
        &mut self, graph: &mut RenderGraph, allocator: &mut PersistentAllocator, retired: &mut Vec<Texture>,
        id: egui::TextureId, delta: &ImageDelta, size: vk::Extent2D,
    ) -> Result<bool, vk::Result> {
        let existing = self.textures.get(&id);

        if delta.pos.is_some() {
            if existing.is_none() {
                tracing::warn!(
                    ?id,
                    "egui patched a texture that was never uploaded; the patch is dropped"
                );
                return Ok(false);
            }

            return Ok(true);
        }

        if existing.is_some_and(|texture| texture.extent == size && texture.options == delta.options) {
            return Ok(true);
        }

        let sampler = self.sampler(allocator, delta.options)?;
        let info = ImageInfo::texture(size, TEXTURE_FORMAT).with_name(format!("egui texture {id:?}"));
        let image = allocator.allocate_image(&info)?;
        let image_view = allocator.allocate_image_view(
            image.handle(),
            TEXTURE_FORMAT,
            vk::ImageViewType::TYPE_2D,
            image.subresource_range(),
        )?;

        let texture = Texture {
            image,
            image_view,
            slot: graph.register_texture(image_view, sampler),
            extent: size,
            options: delta.options,
            uploaded: false,
        };

        retired.extend(self.textures.insert(id, texture));

        Ok(true)
    }

    /// Records the UI over `target`, once, into the module the whole frame is compiled from.
    ///
    /// How many draws there are, what each one clips to and which texture it samples are not
    /// known until the frame runs, so the region declares what it touches and leaves its
    /// commands to a callback. Everything else is a slot: the geometry is two buffers the
    /// frame binds, and whether there is any UI at all is a branch rather than a second graph.
    pub fn record(&self, module: &mut Module, target: ValueId) -> (ValueId, EguiSlots) {
        let slots = EguiSlots {
            has_draws: module.declare_bool_var("has ui", false),
            vertices: module.declare_buffer_var("egui vertices", vir::Access::HostWrite),
            indices: module.declare_buffer_var("egui indices", vir::Access::HostWrite),
            push: module.declare_bytes_var("egui push block", size_of::<PushConstants>() as u32),
            body: module.declare_callback_var("egui draws"),
        };

        let pipeline = self.pipeline;
        let drawn = module.set_condition(
            slots.has_draws,
            |m| {
                m.begin_rendering(&[target])
                    .with_name("egui")
                    .bind_graphics_pipeline(pipeline)
                    .set_dynamic_state(DynamicStateFlags::Viewport | DynamicStateFlags::Scissor)
                    .set_viewport(0, Rect2D::framebuffer())
                    .set_scissor(0, Rect2D::framebuffer())
                    .broadcast_color_blend(BlendPreset::PremultipliedAlphaBlend)
                    .set_rasterization(RasterizationState {
                        cull_mode: vk::CullModeFlags::NONE,
                        ..Default::default()
                    })
                    .push_constants_from(slots.push)
                    .bind_vertex_buffer(0, slots.vertices)
                    .bind_index_buffer(slots.indices, vk::IndexType::UINT32)
                    .record_from(slots.body)
                    .end_rendering()
            },
            |_| target,
        );

        (drawn, slots)
    }

    /// Writes this frame's UI into the compiled program.
    pub fn bind(&self, program: &mut Program, slots: &EguiSlots, frame: &EguiFrame) {
        program.set(slots.has_draws, !frame.draws.is_empty());
        program.set_bytes(
            slots.push,
            &PushConstants {
                screen_size: frame.screen_size_in_points,
                texture_index: 0,
            },
        );

        // a slot the program never reaches is still a slot it refuses to run without, so the
        // frame with no UI in it binds the empty buffers it was handed
        program.set(slots.vertices, frame.vertices);
        program.set(slots.indices, frame.indices);

        if frame.draws.is_empty() {
            return;
        }

        let draws = frame.draws.clone();
        program.set(
            slots.body,
            PassCallback::new(move |cmd| {
                for draw in draws.iter() {
                    cmd.set_scissor(
                        0,
                        Rect2D::absolute(
                            draw.clip.offset.x,
                            draw.clip.offset.y,
                            draw.clip.extent.width,
                            draw.clip.extent.height,
                        ),
                    )
                    .push_constants_at(TEXTURE_INDEX_OFFSET, &draw.slot)
                    .draw_indexed_range(
                        draw.index_count,
                        1,
                        draw.index_offset,
                        draw.vertex_offset,
                        0,
                    );
                }
            }),
        );
    }

    /// Uploads this frame's texture patches, then flattens every mesh into one vertex and one
    /// index buffer taken from this frame's allocator, so the whole UI draws out of a single
    /// binding.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self, ctx: &Context, graph: &mut RenderGraph, persistent: &mut PersistentAllocator,
        allocator: &mut FrameAllocator, textures_delta: TexturesDelta, primitives: &[ClippedPrimitive],
        pixels_per_point: f32, target: vk::Extent2D,
    ) -> Result<EguiFrame, vk::Result> {
        let uploads = self.stage_textures(ctx, graph, persistent, allocator, textures_delta)?;
        let uploaded = uploads.len();
        self.upload(ctx, graph, allocator, uploads)?;

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

            let Some(texture) = self.textures.get(&mesh.texture_id) else {
                tracing::warn!(id = ?mesh.texture_id, "mesh names a texture that was never uploaded; it is dropped");
                continue;
            };

            draws.push(MeshDraw {
                clip,
                slot: texture.slot.0,
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
                draws: Arc::from(draws),
                uploads: uploaded,
                screen_size_in_points,
            });
        }

        let mut vertex_buffer = allocator
            .allocate_buffer(&BufferInfo::vertex(size_of_val(vertices.as_slice()) as u64).with_name("egui vertices"))?;
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
            draws: Arc::from(draws),
            uploads: uploaded,
            screen_size_in_points,
        })
    }

    /// Lands this frame's texture patches.
    ///
    /// A copy is not part of the frame's program: how many there are changes with whatever
    /// egui re-rastered, and that program is compiled once. They are rare enough that a module
    /// of their own, run to completion, costs less than keeping them in would. Every patch
    /// ends in the layout the pass samples from, which is what lets the frame name no texture
    /// at all.
    fn upload(
        &mut self, ctx: &Context, graph: &mut RenderGraph, allocator: &mut FrameAllocator, uploads: Vec<Upload>,
    ) -> Result<(), vk::Result> {
        if uploads.is_empty() {
            return Ok(());
        }

        // one value per texture: importing the same image twice would hand the graph two
        // independent histories of one resource, and it would barrier them apart
        let mut values: Vec<(egui::TextureId, ValueId)> = Vec::new();
        let mut module = Module::default();

        for upload in &uploads {
            let Some(texture) = self.textures.get(&upload.texture) else {
                continue;
            };

            let index = match values.iter().position(|(other, _)| *other == upload.texture) {
                Some(index) => index,
                None => {
                    // a patch lands over what is already there, so only a texture this upload
                    // created has nothing to preserve
                    let layout = match texture.uploaded {
                        true => TEXTURE_RESTING.into(),
                        false => vk::ImageLayout::UNDEFINED,
                    };
                    let value = module.import_attachment(&texture.attachment(layout));
                    module.set_name(value, format!("egui texture {:?}", upload.texture));
                    values.push((upload.texture, value));
                    values.len() - 1
                },
            };

            let staging = module.import_buffer(&upload.staging, vir::Access::HostWrite);
            module.set_name(staging, format!("egui staging {:?}", upload.texture));
            values[index].1 = module.copy_buffer_to_image_region(staging, values[index].1, upload.region);
        }

        let roots = values
            .iter()
            .map(|(_, value)| module.release(*value, TEXTURE_RESTING, vir::DomainFlag::Graphics))
            .collect::<Vec<_>>();
        graph.execute_blocking(ctx, &module.compile_all(&roots), &mut AllocatorKind::Frame(allocator))?;

        for (id, _) in &values {
            if let Some(texture) = self.textures.get_mut(id) {
                texture.uploaded = true;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vir::{
        DescriptorBinding,
        VertexAttribute,
        VertexLayout,
        resource::{pipeline::push_constant_ranges, shader},
    };

    use super::*;
    use crate::{EGUI_FRAG_SPV, EGUI_VERT_SPV, read_spirv};

    /// The whole backend rests on this: `epaint::Vertex` is `[f32;2], [f32;2], [u8;4]`, and
    /// reflection only derives 32-bit attribute formats, so the shader declares the packed
    /// `Color32` as a `uint`. That has to land on the same 20 bytes epaint uploads. An egui bump
    /// that changes `Vertex` should fail here rather than render garbage.
    #[test]
    fn the_reflected_vertex_layout_matches_epaint() {
        let reflections = [
            shader::reflect(&read_spirv(EGUI_VERT_SPV)).expect("vertex shader should reflect"),
            shader::reflect(&read_spirv(EGUI_FRAG_SPV)).expect("fragment shader should reflect"),
        ];

        let layout = VertexLayout::interleaved(&reflections);
        assert_eq!(layout.stride as usize, size_of::<EguiVertex>());
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
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 8,
                },
                VertexAttribute {
                    location: 2,
                    format: vk::Format::R32_UINT,
                    offset: 16,
                },
            ]
        );
    }

    /// The vertex stage reads the screen size out of the block and the fragment stage the
    /// texture index, so one range covers both and spans exactly the struct the pass pushes.
    #[test]
    fn the_push_constant_range_covers_both_stages() {
        let reflect = |spirv| shader::reflect(&read_spirv(spirv)).expect("shader should reflect");

        let ranges = push_constant_ranges(&[reflect(EGUI_VERT_SPV), reflect(EGUI_FRAG_SPV)]);
        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(ranges[0].offset, 0);
        assert_eq!(ranges[0].size as usize, size_of::<PushConstants>());

        // and the index has to be the tail of the block, since a draw pushes it alone
        assert_eq!(
            TEXTURE_INDEX_OFFSET as usize,
            size_of::<PushConstants>() - size_of::<u32>()
        );
    }

    /// The whole bindless path rests on this shape: a variable-count combined image sampler
    /// array is what a pipeline layout recognises as the slot the graph's texture table is
    /// written into, so a shader that reflects as anything else silently samples nothing.
    #[test]
    fn the_fragment_shader_declares_the_bindless_texture_table() {
        let reflection = shader::reflect(&read_spirv(EGUI_FRAG_SPV)).expect("shader should reflect");

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
        let vertex = shader::reflect(&read_spirv(EGUI_VERT_SPV)).expect("shader should reflect");
        assert!(vertex.bindings.is_empty());
    }
}
