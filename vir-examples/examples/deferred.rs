//! Deferred shading: geometry and lighting split across two passes.
//!
//! The geometry stage draws every mesh once into a G-buffer, three color targets and a depth
//! target, and shades nothing. Each material occupies its own rendering region because an
//! ordinary descriptor is pass-scoped. The lighting pass draws a single triangle over the
//! screen and shades every pixel out of what the first one wrote, so the cost of a light no
//! longer scales with the geometry behind it.
//!
//! The G-buffer keeps world position in a target of its own rather than reconstructing it from
//! depth. That is a target more than a lean renderer would carry, and it is the reason the
//! lighting shader needs no inverse matrices at all.
//!
//! It draws the embedded `DamagedHelmet.glb`, whose base color arrives as a texture rather than
//! a factor, so the geometry pass samples it into albedo the same way the lighting pass later
//! samples albedo back out. Pass a path to draw a different model instead.

use std::{error::Error, path::Path};

use ash::vk;
use glam::{
    Mat3,
    Mat4,
    Vec3,
    camera::rh::{proj::vulkan::perspective, view::look_at_mat4},
};
use vir::{
    Access,
    AllocatorKind,
    BlendPreset,
    BufferInfo,
    ClearValue,
    DepthState,
    DomainFlag,
    Image,
    ImageAttachment,
    ImageInfo,
    MemoryLocation,
    PersistentAllocator,
    PipelineId,
    RasterizationState,
    Rect2D,
    RenderGraph,
    SamplerInfo,
    ValueId,
    allocator::Allocator,
};
use vir_examples::{Example, Frame, Recording, Setup, graphics_pipeline};

const GEOMETRY_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deferred_geometry.vert.spv"));
const GEOMETRY_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deferred_geometry.frag.spv"));
const LIGHTING_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deferred_lighting.vert.spv"));
const LIGHTING_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deferred_lighting.frag.spv"));

/// Albedo is the one G-buffer channel that holds a color rather than a quantity, so it is
/// stored encoded: eight linear bits would band in the darks the lighting pass leans on.
const ALBEDO_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
/// glTF authors base color in sRGB, and an `_SRGB` image is what samples it back linear.
const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
/// Normals and positions both need range and sign that eight bits cannot hold.
const NORMAL_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const POSITION_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// A material texture is written once at startup and only ever sampled after that.
const TEXTURE_RESTING: Access = Access::FragmentSampled;

/// glTF winds a front face counter-clockwise seen from outside, and the projection is already
/// the Y-down one Vulkan measures that winding in, so nothing has to be turned around.
const FRONT_FACE: vk::FrontFace = vk::FrontFace::COUNTER_CLOCKWISE;

/// Cleared to zero alpha, which is the coverage the lighting pass reads the background out of.
const CLEAR_UNCOVERED: ClearValue = ClearValue::rgba_f32(0.0, 0.0, 0.0, 0.0);
/// A forward depth test starts from the far plane.
const CLEAR_DEPTH: ClearValue = ClearValue::depth(1.0);

/// One vertex as `vs_main` reads it: `float3 position`, `float3 normal`, then `float2 uv`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

/// The block `deferred_geometry.slang` declares. `view_proj` goes over row by row, which is the
/// matrix layout the build compiles Slang with.
#[repr(C)]
#[derive(Clone, Copy)]
struct GeometryPush {
    view_proj: [[f32; 4]; 4],
    base_color: [f32; 3],
}

/// The block `deferred_lighting.slang` declares. Each `float3` is followed by a scalar that
/// fills the padding a `float3` would otherwise carry.
#[repr(C)]
#[derive(Clone, Copy)]
struct LightingPush {
    camera_position: [f32; 3],
    ambient: f32,
    light_direction: [f32; 3],
    light_intensity: f32,
    light_color: [f32; 3],
    specular_strength: f32,
    mode: u32,
}

/// What the lighting pass writes, which is either the shading or one of the targets it shaded
/// from. The debug modes are what make a G-buffer worth looking at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Shaded = 0,
    Albedo = 1,
    Normal = 2,
    Position = 3,
}

impl ViewMode {
    const ALL: [(Self, &'static str); 4] = [
        (Self::Shaded, "shaded"),
        (Self::Albedo, "albedo"),
        (Self::Normal, "normal"),
        (Self::Position, "position"),
    ];
}

/// Geometry as it comes off a loader, before it reaches the GPU.
struct MeshData {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    base_color: [f32; 3],
    /// Indexes [`SceneData::images`].
    base_color_image: Option<usize>,
}

/// One decoded glTF image, already widened to the four channels an upload wants.
struct ImageData {
    pixels: Vec<u8>,
    extent: vk::Extent2D,
}

/// A whole glTF file, flattened into what the example draws.
struct SceneData {
    meshes: Vec<MeshData>,
    images: Vec<ImageData>,
}

/// One mesh's buffers, uploaded once and drawn every frame.
struct Mesh {
    vertices: vir::Buffer,
    indices: vir::Buffer,
    index_count: u32,
    base_color: [f32; 3],
    /// The scalar descriptor this mesh binds before drawing.
    base_color_texture: usize,
}

/// A material texture, uploaded once and sampled by every draw that names it.
struct Texture {
    image: Image,
    view: vk::ImageView,
}

impl Texture {
    fn attachment(&self) -> ImageAttachment {
        ImageAttachment::from_image(&self.image, TEXTURE_RESTING.into()).with_image_view(self.view)
    }
}

/// A G-buffer target, which is a color attachment the geometry pass writes and the lighting
/// pass samples.
struct GTarget {
    image: Image,
    view: vk::ImageView,
}

impl GTarget {
    /// Every run clears it before anything reads it, so what the last one left in it is not
    /// worth a transition.
    fn attachment(&self) -> ImageAttachment {
        ImageAttachment::from_image(&self.image, vk::ImageLayout::UNDEFINED).with_image_view(self.view)
    }
}

/// Everything the two passes render through, all of it sized to the swapchain.
struct GBuffer {
    albedo: GTarget,
    normal: GTarget,
    position: GTarget,
    depth: Image,
    depth_view: vk::ImageView,
}

impl GBuffer {
    fn depth_attachment(&self) -> ImageAttachment {
        ImageAttachment::from_image(&self.depth, vk::ImageLayout::UNDEFINED).with_image_view(self.depth_view)
    }

    fn targets(&self) -> [&GTarget; 3] { [&self.albedo, &self.normal, &self.position] }
}

/// Where the camera sits and how far out it has to be to see everything.
#[derive(Clone, Copy)]
struct Bounds {
    center: Vec3,
    radius: f32,
}

impl Bounds {
    fn of(meshes: &[MeshData]) -> Self {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for vertex in meshes.iter().flat_map(|mesh| &mesh.vertices) {
            let position = Vec3::from(vertex.position);
            min = min.min(position);
            max = max.max(position);
        }

        // an empty scene still has to give the camera somewhere to look
        if !min.is_finite() || !max.is_finite() {
            return Self {
                center: Vec3::ZERO,
                radius: 1.0,
            };
        }

        Self {
            center: (min + max) * 0.5,
            radius: ((max - min).length() * 0.5).max(0.001),
        }
    }
}

struct UiState {
    yaw: f32,
    pitch: f32,
    /// Multiplies the scene radius, so the framing holds whatever the model's scale is.
    distance: f32,
    orbit: bool,
    light_yaw: f32,
    light_pitch: f32,
    ambient: f32,
    intensity: f32,
    specular: f32,
    mode: ViewMode,
    cull_backfaces: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.45,
            distance: 2.6,
            orbit: true,
            light_yaw: 1.1,
            light_pitch: 0.9,
            ambient: 0.08,
            intensity: 1.0,
            specular: 0.35,
            mode: ViewMode::Shaded,
            cull_backfaces: false,
        }
    }
}

/// The slots the frame writes into the compiled passes.
struct Slots {
    /// One block per mesh, since what the camera decided and what the material decided share
    /// it.
    geometry: Vec<ValueId>,
    lighting: ValueId,
    cull_backfaces: ValueId,
}

struct Deferred {
    geometry_pipeline: PipelineId,
    lighting_pipeline: PipelineId,
    meshes: Vec<Mesh>,
    textures: Vec<Texture>,
    /// Samples the G-buffer, which is read one texel to one pixel.
    sampler: vk::Sampler,
    /// Samples material textures, which are read at whatever rate the model's UVs ask for.
    texture_sampler: vk::Sampler,
    gbuffer: Option<GBuffer>,
    slots: Option<Slots>,
    bounds: Bounds,
    /// What the scene was built from, which the panel names.
    source: String,
    triangles: usize,
    ui: UiState,
}

impl Deferred {
    /// Where the camera orbits to, in world space.
    fn camera_position(&self, elapsed: f32) -> Vec3 {
        let yaw = self.ui.yaw + if self.ui.orbit { elapsed * 0.3 } else { 0.0 };
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.ui.pitch.sin_cos();

        let direction = Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw);
        self.bounds.center + direction * self.bounds.radius * self.ui.distance
    }

    /// Clip space for `extent`. The near and far planes come off the scene's own size, so the
    /// depth buffer keeps its precision whatever scale the model was authored at.
    fn view_projection(&self, eye: Vec3, extent: vk::Extent2D) -> Mat4 {
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        let radius = self.bounds.radius;

        let view = look_at_mat4(eye, self.bounds.center, Vec3::Y);
        // the Vulkan variant is already Y-down with depth over [0, 1], so nothing is flipped
        // back afterwards
        let projection = perspective(60f32.to_radians(), aspect, radius * 0.01, radius * 40.0);

        projection * view
    }

    fn light_direction(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.ui.light_yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.ui.light_pitch.sin_cos();

        // the direction light travels in, so the panel's angles point at where it comes from
        -Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw).normalize_or(Vec3::NEG_Y)
    }

    /// Throws away the G-buffer and everything it holds.
    fn release_gbuffer(&mut self, _graph: &mut RenderGraph, allocator: &mut PersistentAllocator) {
        let Some(gbuffer) = self.gbuffer.take() else {
            return;
        };

        for target in gbuffer.targets() {
            allocator.deallocate_image_view(target.view);
            allocator.deallocate_image(target.image);
        }

        allocator.deallocate_image_view(gbuffer.depth_view);
        allocator.deallocate_image(gbuffer.depth);
    }

    /// One color target the lighting pass samples through an ordinary descriptor.
    fn create_target(recording: &mut Recording, format: vk::Format, name: &str) -> Result<GTarget, vk::Result> {
        let info = ImageInfo::color_target(recording.target.extent, format)
            // sampled by the lighting pass, cleared by a transfer at the top of the frame
            .with_usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .with_name(name);

        let image = recording.allocator.allocate_image(&info)?;
        let view = recording.allocator.allocate_image_view(
            image.handle(),
            format,
            vk::ImageViewType::TYPE_2D,
            image.subresource_range(),
        )?;
        Ok(GTarget { image, view })
    }

    /// The whole G-buffer matches the swapchain, so it is thrown away and remade with it.
    fn create_gbuffer(&mut self, recording: &mut Recording) -> Result<GBuffer, vk::Result> {
        self.release_gbuffer(recording.graph, recording.allocator);

        let albedo = Self::create_target(recording, ALBEDO_FORMAT, "g-buffer albedo")?;
        let normal = Self::create_target(recording, NORMAL_FORMAT, "g-buffer normal")?;
        let position = Self::create_target(recording, POSITION_FORMAT, "g-buffer position")?;

        let depth_info = ImageInfo::depth_target(recording.target.extent, DEPTH_FORMAT)
            .with_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .with_name("g-buffer depth");
        let depth = recording.allocator.allocate_image(&depth_info)?;
        let depth_view = recording.allocator.allocate_image_view(
            depth.handle(),
            DEPTH_FORMAT,
            vk::ImageViewType::TYPE_2D,
            depth.subresource_range(),
        )?;

        Ok(GBuffer {
            albedo,
            normal,
            position,
            depth,
            depth_view,
        })
    }

    /// Uploads every material image in one go: a staging buffer and a copy each, then a single
    /// release apiece into the layout a sample wants.
    fn upload_textures(setup: &mut Setup, images: &[ImageData]) -> Result<Vec<Texture>, vk::Result> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut textures = Vec::with_capacity(images.len());
        let mut staging = Vec::with_capacity(images.len());
        let mut module = vir::Module::default();
        let mut roots = Vec::new();

        for (index, data) in images.iter().enumerate() {
            let image = setup
                .allocator
                .allocate_image(&ImageInfo::texture(data.extent, TEXTURE_FORMAT).with_name("material texture"))?;
            let view = setup.allocator.allocate_image_view(
                image.handle(),
                TEXTURE_FORMAT,
                vk::ImageViewType::TYPE_2D,
                image.subresource_range(),
            )?;
            let mut buffer = setup.allocator.allocate_buffer(
                &BufferInfo::staging(size_of_val(data.pixels.as_slice()) as u64).with_name("texture staging"),
            )?;
            buffer.write(0, &data.pixels)?;

            let source = module.import_buffer(&buffer, vir::Access::HostWrite);
            let destination =
                module.import_attachment(&ImageAttachment::from_image(&image, vk::ImageLayout::UNDEFINED));
            module.set_name(destination, "material texture");

            let copied = module.copy_buffer_to_image(source, destination);
            roots.push(module.release(copied, TEXTURE_RESTING, DomainFlag::Graphics));

            tracing::debug!(
                index,
                width = data.extent.width,
                height = data.extent.height,
                "uploading"
            );
            textures.push(Texture { image, view });
            staging.push(buffer);
        }

        let program = module.compile_all(&*setup.graph, &roots)?;
        setup
            .graph
            .execute_blocking(setup.ctx, &program, &mut AllocatorKind::Persistent(setup.allocator))?;

        for buffer in staging {
            setup.allocator.deallocate_buffer(buffer);
        }

        Ok(textures)
    }

    fn upload(setup: &mut Setup, data: &MeshData, textures: &[Texture]) -> Result<Mesh, vk::Result> {
        let mut vertices = setup.allocator.allocate_buffer(
            &BufferInfo::vertex(size_of_val(data.vertices.as_slice()) as u64).with_name("mesh vertices"),
        )?;
        vertices.write(0, &data.vertices)?;

        let mut indices = setup.allocator.allocate_buffer(
            &BufferInfo::new(
                size_of_val(data.indices.as_slice()) as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )
            .with_name("mesh indices"),
        )?;
        indices.write(0, &data.indices)?;

        Ok(Mesh {
            vertices,
            indices,
            index_count: data.indices.len() as u32,
            base_color: data.base_color,
            base_color_texture: data
                .base_color_image
                .filter(|index| *index < textures.len().saturating_sub(1))
                .unwrap_or(textures.len().saturating_sub(1)),
        })
    }
}

impl Example for Deferred {
    const TITLE: &'static str = "vir: deferred";

    fn new(setup: &mut Setup) -> Result<Self, vk::Result> {
        // whatever the scene came from, the rest of the example only sees meshes
        let (source, loaded) = match std::env::args().nth(1) {
            Some(path) => {
                let meshes = scene::load_gltf(Path::new(&path));
                (path, meshes)
            },
            None => ("DamagedHelmet.glb".to_owned(), scene::helmet()),
        };

        let mut scene = loaded.map_err(|err| {
            tracing::error!(%source, %err, "could not read the model");
            vk::Result::ERROR_INITIALIZATION_FAILED
        })?;

        let triangles = scene.meshes.iter().map(|mesh| mesh.indices.len() / 3).sum();
        let bounds = Bounds::of(&scene.meshes);
        tracing::info!(
            %source,
            meshes = scene.meshes.len(),
            textures = scene.images.len(),
            triangles,
            radius = bounds.radius,
            "scene loaded"
        );

        let sampler = setup.allocator.allocate_sampler(
            &SamplerInfo::default()
                // the G-buffer is sampled one texel to one pixel, so filtering it would only
                // smear what the geometry pass wrote
                .with_mag_filter(vk::Filter::NEAREST)
                .with_min_filter(vk::Filter::NEAREST)
                .with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        )?;
        let texture_sampler = setup.allocator.allocate_sampler(
            &SamplerInfo::default()
                .with_mag_filter(vk::Filter::LINEAR)
                .with_min_filter(vk::Filter::LINEAR)
                // glTF wraps by default, and a model's UVs are free to run outside [0, 1]
                .with_address_mode(vk::SamplerAddressMode::REPEAT),
        )?;

        // A scalar combined-image descriptor must always be populated. Materials without an
        // image bind this white texel, leaving their base-color factor unchanged.
        scene.images.push(ImageData {
            pixels: vec![255, 255, 255, 255],
            extent: vk::Extent2D { width: 1, height: 1 },
        });
        let textures = Self::upload_textures(setup, &scene.images)?;

        Ok(Self {
            geometry_pipeline: graphics_pipeline(setup.graph, GEOMETRY_VERT_SPV, GEOMETRY_FRAG_SPV)?,
            lighting_pipeline: graphics_pipeline(setup.graph, LIGHTING_VERT_SPV, LIGHTING_FRAG_SPV)?,
            meshes: scene
                .meshes
                .iter()
                .map(|data| Self::upload(setup, data, &textures))
                .collect::<Result<_, _>>()?,
            textures,
            sampler,
            texture_sampler,
            gbuffer: None,
            slots: None,
            bounds,
            source,
            triangles,
            ui: UiState::default(),
        })
    }

    fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(Self::TITLE)
            .default_pos([24.0, 24.0])
            .default_size([320.0, 420.0])
            .show(ctx, |ui| {
                ui.label(format!(
                    "{} - {} meshes, {} triangles",
                    self.source,
                    self.meshes.len(),
                    self.triangles
                ));

                ui.separator();
                ui.label("g-buffer");
                ui.horizontal(|ui| {
                    for (mode, name) in ViewMode::ALL {
                        ui.selectable_value(&mut self.ui.mode, mode, name);
                    }
                });

                ui.separator();
                ui.label("camera");
                ui.checkbox(&mut self.ui.orbit, "orbit");
                ui.add(egui::Slider::new(&mut self.ui.yaw, -3.2..=3.2).text("yaw"));
                ui.add(egui::Slider::new(&mut self.ui.pitch, -1.5..=1.5).text("pitch"));
                ui.add(egui::Slider::new(&mut self.ui.distance, 0.5..=8.0).text("distance"));

                ui.separator();
                ui.label("light");
                ui.add(egui::Slider::new(&mut self.ui.light_yaw, -3.2..=3.2).text("yaw"));
                ui.add(egui::Slider::new(&mut self.ui.light_pitch, -1.5..=1.5).text("pitch"));
                ui.add(egui::Slider::new(&mut self.ui.intensity, 0.0..=3.0).text("intensity"));
                ui.add(egui::Slider::new(&mut self.ui.ambient, 0.0..=1.0).text("ambient"));
                ui.add(egui::Slider::new(&mut self.ui.specular, 0.0..=2.0).text("specular"));

                ui.separator();
                ui.checkbox(&mut self.ui.cull_backfaces, "cull backfaces");
                if ui.button("reset").clicked() {
                    self.ui = UiState::default();
                }
            });
    }

    fn record(&mut self, recording: &mut Recording) -> Result<ValueId, vk::Result> {
        let gbuffer = self.create_gbuffer(recording)?;
        let module = &mut *recording.module;

        // a block is either bytes or a variable, never both, so what changes every frame and
        // what was decided when the model loaded go into one block per mesh
        let geometry_push = (0..self.meshes.len())
            .map(|index| module.declare_bytes_var(&format!("mesh {index} block"), size_of::<GeometryPush>() as u32))
            .collect::<Vec<_>>();
        let lighting_push = module.declare_bytes_var("lighting block", size_of::<LightingPush>() as u32);
        let cull_backfaces = module.declare_bool_var("cull backfaces", false);

        // every buffer the region draws from has to be imported before the pass borrows the
        // module, so the geometry is gathered up front
        let meshes = self
            .meshes
            .iter()
            .zip(&geometry_push)
            .map(|(mesh, push)| {
                (
                    module.import_buffer(&mesh.vertices, vir::Access::HostWrite),
                    module.import_buffer(&mesh.indices, vir::Access::HostWrite),
                    mesh.index_count,
                    mesh.base_color_texture,
                    *push,
                )
            })
            .collect::<Vec<_>>();

        // Material textures are imported as graph images so each mesh can bind its own scalar
        // descriptor before drawing.
        let material_textures = self
            .textures
            .iter()
            .map(|texture| module.import_attachment(&texture.attachment()))
            .collect::<Vec<_>>();

        let albedo = module.import_attachment(&gbuffer.albedo.attachment());
        let normal = module.import_attachment(&gbuffer.normal.attachment());
        let position = module.import_attachment(&gbuffer.position.attachment());
        let depth = module.import_attachment(&gbuffer.depth_attachment());
        module.set_name(albedo, "g-buffer albedo");
        module.set_name(normal, "g-buffer normal");
        module.set_name(position, "g-buffer position");
        module.set_name(depth, "g-buffer depth");

        let albedo = module.clear(albedo, CLEAR_UNCOVERED);
        let normal = module.clear(normal, CLEAR_UNCOVERED);
        let position = module.clear(position, CLEAR_UNCOVERED);
        let depth = module.clear(depth, CLEAR_DEPTH);

        // which way the geometry pass culls is a branch, since a cull mode is baked into a
        // pipeline and both of them are worth having compiled
        let geometry = |m: &mut vir::Module, cull_mode| {
            let mut filled = albedo;
            for (vertices, indices, index_count, texture, push) in &meshes {
                filled = m
                    .begin_rendering([
                        (filled, Access::ColorRW),
                        (normal, Access::ColorRW),
                        (position, Access::ColorRW),
                        (depth, Access::DepthStencilRW),
                    ])
                    .with_name("g-buffer mesh")
                    .bind_graphics_pipeline(self.geometry_pipeline)
                    .set_viewport(0, Rect2D::framebuffer())
                    .set_scissor(0, Rect2D::framebuffer())
                    .broadcast_color_blend(BlendPreset::Off)
                    .set_depth(DepthState::less())
                    .set_rasterization(RasterizationState {
                        cull_mode,
                        front_face: FRONT_FACE,
                        ..Default::default()
                    })
                    .bind_texture(0, 0, material_textures[*texture], self.texture_sampler)
                    .push_constants_from(*push)
                    .bind_vertex_buffer(0, *vertices)
                    .bind_index_buffer(*indices, vk::IndexType::UINT32)
                    .draw_indexed(*index_count, 1)
                    .end_rendering();
            }
            filled
        };
        let filled = module.set_condition(
            cull_backfaces,
            |m| geometry(m, vk::CullModeFlags::BACK),
            |m| geometry(m, vk::CullModeFlags::NONE),
        );

        // `filled` carries the albedo target out of the region; the other two are the same
        // region's writes, so naming them by the value that went in is what orders the sample
        let shaded = module
            .begin_rendering([(recording.swapchain_image, Access::ColorRW)])
            .with_name("deferred lighting")
            .bind_graphics_pipeline(self.lighting_pipeline)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .bind_texture(0, 0, filled, self.sampler)
            .bind_texture(0, 1, normal, self.sampler)
            .bind_texture(0, 2, position, self.sampler)
            .push_constants_from(lighting_push)
            .draw(3, 1)
            .end_rendering();

        self.slots = Some(Slots {
            geometry: geometry_push,
            lighting: lighting_push,
            cull_backfaces,
        });
        self.gbuffer = Some(gbuffer);

        Ok(shaded)
    }

    fn update(&mut self, frame: &mut Frame) -> Result<(), vk::Result> {
        let (Some(slots), Some(_gbuffer)) = (self.slots.as_ref(), self.gbuffer.as_ref()) else {
            return Ok(());
        };

        let eye = self.camera_position(frame.elapsed);
        let view_proj = row_major(self.view_projection(eye, frame.target.extent));

        for (mesh, push) in self.meshes.iter().zip(&slots.geometry) {
            frame.program.set_bytes(
                *push,
                &GeometryPush {
                    view_proj,
                    base_color: mesh.base_color,
                },
            );
        }

        let light_direction = self.light_direction();
        frame.program.set_bytes(
            slots.lighting,
            &LightingPush {
                camera_position: eye.to_array(),
                ambient: self.ui.ambient,
                light_direction: light_direction.to_array(),
                light_intensity: self.ui.intensity,
                light_color: [1.0, 0.97, 0.92],
                specular_strength: self.ui.specular,
                mode: self.ui.mode as u32,
            },
        );
        frame.program.set(slots.cull_backfaces, self.ui.cull_backfaces);

        Ok(())
    }

    fn destroy(&mut self, graph: &mut RenderGraph, allocator: &mut PersistentAllocator) {
        self.release_gbuffer(graph, allocator);
        allocator.deallocate_sampler(self.sampler);
        allocator.deallocate_sampler(self.texture_sampler);

        for mesh in self.meshes.drain(..) {
            allocator.deallocate_buffer(mesh.vertices);
            allocator.deallocate_buffer(mesh.indices);
        }

        for texture in self.textures.drain(..) {
            allocator.deallocate_image_view(texture.view);
            allocator.deallocate_image(texture.image);
        }
    }
}

/// Slang reads a matrix row by row, and glam stores one column by column.
fn row_major(matrix: Mat4) -> [[f32; 4]; 4] { matrix.transpose().to_cols_array_2d() }

/// Where the geometry comes from.
mod scene {
    use super::*;

    /// The model the example draws unless it is handed another one. A `.glb` carries its
    /// buffers inside it, so embedding the one file is the whole asset.
    const HELMET_GLB: &[u8] = include_bytes!("../assets/DamagedHelmet.glb");

    pub fn helmet() -> Result<SceneData, Box<dyn Error>> { flatten(gltf::import_slice(HELMET_GLB)?) }

    pub fn load_gltf(path: &Path) -> Result<SceneData, Box<dyn Error>> { flatten(gltf::import(path)?) }

    /// Flattens a glTF scene into world-space meshes.
    ///
    /// Node transforms are baked into the vertices rather than kept as per-draw matrices, which
    /// is what lets the geometry pass push nothing but the camera.
    fn flatten(
        (document, buffers, images): (gltf::Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>),
    ) -> Result<SceneData, Box<dyn Error>> {
        let scene = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .ok_or("the file has no scene to draw")?;

        let mut meshes = Vec::new();
        for node in scene.nodes() {
            visit(&node, Mat4::IDENTITY, &buffers, &mut meshes);
        }

        if meshes.is_empty() {
            return Err("the scene has no triangles in it".into());
        }

        // a glTF file carries images no material points at, and each one would cost its full
        // size in device memory, so only what a base color actually names is converted
        let mut kept = vec![None; images.len()];
        let mut converted = Vec::new();
        for mesh in &mut meshes {
            let Some(source) = mesh.base_color_image else {
                continue;
            };

            let slot = kept
                .get_mut(source)
                .ok_or("a material names an image that is not there")?;
            if slot.is_none() {
                *slot = Some(converted.len());
                converted.push(to_rgba(&images[source])?);
            }

            mesh.base_color_image = *slot;
        }

        Ok(SceneData {
            meshes,
            images: converted,
        })
    }

    /// Widens a decoded glTF image to the four 8-bit channels an upload wants.
    fn to_rgba(image: &gltf::image::Data) -> Result<ImageData, Box<dyn Error>> {
        use gltf::image::Format;

        // an 8-bit source is the only one a base color realistically arrives in, and widening
        // it is a channel count away rather than a conversion
        let channels = match image.format {
            Format::R8 => 1,
            Format::R8G8 => 2,
            Format::R8G8B8 => 3,
            Format::R8G8B8A8 => 4,
            format => return Err(format!("unsupported base color image format {format:?}").into()),
        };

        let pixels = image
            .pixels
            .chunks_exact(channels)
            .flat_map(|texel| {
                [
                    texel[0],
                    // a single channel reads as grey rather than as red
                    if channels >= 3 { texel[1] } else { texel[0] },
                    if channels >= 3 { texel[2] } else { texel[0] },
                    if channels == 4 { texel[3] } else { u8::MAX },
                ]
            })
            .collect();

        Ok(ImageData {
            pixels,
            extent: vk::Extent2D {
                width: image.width,
                height: image.height,
            },
        })
    }

    fn visit(node: &gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], out: &mut Vec<MeshData>) {
        let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());

        if let Some(mesh) = node.mesh() {
            out.extend(
                mesh.primitives()
                    .filter(|primitive| primitive.mode() == gltf::mesh::Mode::Triangles)
                    .filter_map(|primitive| read_primitive(&primitive, transform, buffers)),
            );
        }

        for child in node.children() {
            visit(&child, transform, buffers, out);
        }
    }

    fn read_primitive(
        primitive: &gltf::Primitive, transform: Mat4, buffers: &[gltf::buffer::Data],
    ) -> Option<MeshData> {
        let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));

        let positions = reader.read_positions()?.collect::<Vec<_>>();
        let normals = reader.read_normals().map(|read| read.collect::<Vec<_>>());
        let uvs = reader
            .read_tex_coords(0)
            .map(|read| read.into_f32().collect::<Vec<_>>());
        let indices = match reader.read_indices() {
            Some(read) => read.into_u32().collect::<Vec<_>>(),
            None => (0..positions.len() as u32).collect(),
        };

        // a normal survives a non-uniform scale only through the inverse transpose
        let normal_matrix = Mat3::from_mat4(transform).inverse().transpose();

        let mut vertices = positions
            .iter()
            .enumerate()
            .map(|(index, position)| Vertex {
                position: transform.transform_point3(Vec3::from(*position)).to_array(),
                normal: match normals.as_ref().and_then(|normals| normals.get(index)) {
                    Some(normal) => (normal_matrix * Vec3::from(*normal)).normalize_or(Vec3::Y).to_array(),
                    None => [0.0; 3],
                },
                uv: uvs.as_ref().and_then(|uvs| uvs.get(index)).copied().unwrap_or([0.0; 2]),
            })
            .collect::<Vec<_>>();

        // a primitive is allowed to ship without normals, and the lighting pass needs them
        if normals.is_none() {
            generate_normals(&mut vertices, &indices);
        }

        let material = primitive.material().pbr_metallic_roughness();
        let base_color = material.base_color_factor();

        Some(MeshData {
            vertices,
            indices,
            base_color: [base_color[0], base_color[1], base_color[2]],
            // the texture names an image, and the image is what actually gets uploaded
            base_color_image: material
                .base_color_texture()
                .map(|info| info.texture().source().index()),
        })
    }

    /// Area-weighted face normals accumulated onto the vertices that share them.
    fn generate_normals(vertices: &mut [Vertex], indices: &[u32]) {
        let mut accumulated = vec![Vec3::ZERO; vertices.len()];

        for triangle in indices.as_chunks::<3>().0 {
            let [a, b, c] = [triangle[0] as usize, triangle[1] as usize, triangle[2] as usize];
            let Some(([pa, pb], pc)) = vertices
                .get(a)
                .zip(vertices.get(b))
                .map(|(pa, pb)| [Vec3::from(pa.position), Vec3::from(pb.position)])
                .zip(vertices.get(c).map(|pc| Vec3::from(pc.position)))
            else {
                continue;
            };

            // the cross product's length is twice the triangle's area, which is the weight
            let face = (pb - pa).cross(pc - pa);
            for index in [a, b, c] {
                accumulated[index] += face;
            }
        }

        for (vertex, normal) in vertices.iter_mut().zip(accumulated) {
            vertex.normal = normal.normalize_or(Vec3::Y).to_array();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> { vir_examples::run::<Deferred>() }

#[cfg(test)]
mod tests {
    use vir::{
        DescriptorBinding,
        VertexAttribute,
        VertexLayout,
        resource::{pipeline::push_constant_ranges, shader},
    };
    use vir_examples::read_spirv;

    use super::*;

    fn reflect(spirv: &[u8]) -> shader::Reflection {
        shader::reflect(&read_spirv(spirv)).expect("shader should reflect")
    }

    /// The reflected vertex layout must match the `Vertex` struct every mesh uploads.
    #[test]
    fn the_reflected_vertex_layout_matches_the_uploaded_vertex() {
        let layout = VertexLayout::interleaved(&[reflect(GEOMETRY_VERT_SPV), reflect(GEOMETRY_FRAG_SPV)]);

        assert_eq!(layout.stride as usize, size_of::<Vertex>());
        assert_eq!(
            layout.attributes,
            vec![
                VertexAttribute {
                    location: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 12,
                },
                VertexAttribute {
                    location: 2,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 24,
                },
            ]
        );
    }

    /// Both stages read the geometry block, so one range covers them and it has to be the size
    /// of the struct the draw pushes.
    #[test]
    fn the_geometry_block_covers_both_stages_at_the_pushed_size() {
        let ranges = push_constant_ranges(&[reflect(GEOMETRY_VERT_SPV), reflect(GEOMETRY_FRAG_SPV)]);

        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(ranges[0].offset, 0);
        assert_eq!(ranges[0].size as usize, size_of::<GeometryPush>());
    }

    /// The lighting stage builds its geometry from the vertex id alone, so only the fragment
    /// stage reads the block the pass pushes.
    #[test]
    fn the_lighting_block_is_fragment_only_at_the_pushed_size() {
        let ranges = push_constant_ranges(&[reflect(LIGHTING_VERT_SPV), reflect(LIGHTING_FRAG_SPV)]);

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stage_flags, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(ranges[0].size as usize, size_of::<LightingPush>());
    }

    #[test]
    fn fragment_stages_declare_their_scalar_textures() {
        let texture = DescriptorBinding {
            set: 0,
            binding: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            count: 1,
            variable_count: false,
            stages: vk::ShaderStageFlags::FRAGMENT,
        };

        assert_eq!(reflect(GEOMETRY_FRAG_SPV).bindings, vec![texture]);
        assert_eq!(
            reflect(LIGHTING_FRAG_SPV).bindings,
            vec![
                texture,
                DescriptorBinding { binding: 1, ..texture },
                DescriptorBinding { binding: 2, ..texture },
            ]
        );

        // the vertex stages stay out of them, so every descriptor is fragment-only
        assert!(reflect(GEOMETRY_VERT_SPV).bindings.is_empty());
        assert!(reflect(LIGHTING_VERT_SPV).bindings.is_empty());
    }

    /// The base color image a material names has to survive the loader's dedup and land on a
    /// texture that actually gets uploaded.
    #[test]
    fn the_model_keeps_the_base_color_image_its_material_names() {
        let scene = scene::helmet().expect("the embedded model should load");

        assert!(!scene.images.is_empty(), "the helmet is a textured model");
        for mesh in &scene.meshes {
            let Some(image) = mesh.base_color_image else {
                continue;
            };
            assert!(image < scene.images.len(), "the image index should be uploadable");
        }

        // every kept image is referenced, which is what stops unused ones costing memory
        for (index, image) in scene.images.iter().enumerate() {
            assert!(
                scene.meshes.iter().any(|mesh| mesh.base_color_image == Some(index)),
                "image {index} was kept without a material naming it"
            );
            assert_eq!(
                image.pixels.len(),
                image.extent.width as usize * image.extent.height as usize * 4,
                "the upload expects four tightly packed channels"
            );
        }
    }

    /// A material texture is uploaded once and read from then on, so the layout it is left in
    /// has to be the one a sample wants.
    #[test]
    fn the_texture_resting_access_is_the_layout_a_sample_wants() {
        assert_eq!(
            vk::ImageLayout::from(TEXTURE_RESTING),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    /// Every index has to name a vertex that exists, or a draw reads past the buffer.
    #[test]
    fn the_model_is_made_of_whole_triangles() {
        let scene = scene::helmet().expect("the embedded model should load");
        let meshes = &scene.meshes;
        assert!(!meshes.is_empty());

        for mesh in meshes {
            assert_eq!(mesh.indices.len() % 3, 0, "indices should be whole triangles");
            assert!(mesh.indices.iter().all(|index| (*index as usize) < mesh.vertices.len()));
            assert!(
                mesh.vertices
                    .iter()
                    .all(|vertex| Vec3::from(vertex.normal).is_normalized()),
                "the lighting pass needs unit normals"
            );
        }
    }

    /// Culling only keeps the right half of the model if the winding the loader produces is the
    /// one [`FRONT_FACE`] names. A triangle wound counter-clockwise seen from outside has a
    /// cross product pointing the same way as the normals of the vertices it is built from.
    #[test]
    fn the_model_winds_the_way_the_front_face_is_declared() {
        assert_eq!(
            FRONT_FACE,
            vk::FrontFace::COUNTER_CLOCKWISE,
            "the check below is written for counter-clockwise front faces"
        );

        let scene = scene::helmet().expect("the embedded model should load");
        let meshes = &scene.meshes;
        let (mut agree, mut total) = (0usize, 0usize);

        for mesh in meshes {
            for triangle in mesh.indices.as_chunks::<3>().0 {
                let [a, b, c] = [
                    mesh.vertices[triangle[0] as usize],
                    mesh.vertices[triangle[1] as usize],
                    mesh.vertices[triangle[2] as usize],
                ];
                let [pa, pb, pc] = [a, b, c].map(|vertex| Vec3::from(vertex.position));

                let face = (pb - pa).cross(pc - pa);
                // a degenerate triangle has no winding to read
                if face.length_squared() <= f32::EPSILON {
                    continue;
                }

                let shaded = [a, b, c].iter().map(|vertex| Vec3::from(vertex.normal)).sum::<Vec3>();

                total += 1;
                agree += usize::from(face.dot(shaded) > 0.0);
            }
        }

        assert!(total > 0, "the model should have triangles to check");
        // authored models carry the odd inverted triangle, so this is a verdict on the whole
        // mesh rather than on any one of them
        let ratio = agree as f32 / total as f32;
        assert!(ratio > 0.95, "only {agree} of {total} triangles wind outwards");
    }

    /// The camera frames whatever it is given, so the bounds have to cover every vertex.
    #[test]
    fn the_bounds_cover_the_scene_they_were_measured_from() {
        let scene = scene::helmet().expect("the embedded model should load");
        let meshes = &scene.meshes;
        let bounds = Bounds::of(meshes);

        for vertex in meshes.iter().flat_map(|mesh| &mesh.vertices) {
            let distance = (Vec3::from(vertex.position) - bounds.center).length();
            assert!(
                distance <= bounds.radius + 1e-3,
                "{distance} is outside {}",
                bounds.radius
            );
        }
    }

    /// An empty scene still has to leave the camera somewhere finite to look from.
    #[test]
    fn empty_bounds_stay_finite() {
        let bounds = Bounds::of(&[]);
        assert!(bounds.center.is_finite() && bounds.radius > 0.0);
    }

    /// Slang reads the pushed matrix row by row, so what goes over has to be the transpose of
    /// what glam holds.
    #[test]
    fn the_pushed_matrix_is_row_major() {
        let matrix = Mat4::from_cols_array(&std::array::from_fn(|index| index as f32));
        let pushed = row_major(matrix);

        for (row, values) in pushed.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                assert_eq!(*value, matrix.col(column)[row]);
            }
        }
    }
}
