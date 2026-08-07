//! The scaffolding every example sits on.
//!
//! An example is a type that implements [`Example`] and a `main` that hands it to [`run`]. The
//! harness owns everything that is not the point of any one example: the window, the instance
//! and device, the swapchain and its allocators, the render graph, and the egui overlay that
//! every example draws its controls into. What an example writes is its pipelines, its
//! resources, and the passes it records into the frame's module.

pub mod device_builder;
pub mod egui_pass;
mod window;

use std::{error::Error, io::Cursor, result::Result, time::Instant};

use ash::{Entry, khr, vk};
pub use egui;
pub use vir;
use vir::{
    AllocatorKind,
    Context,
    FrameAllocator,
    GraphicsPipelineInfo,
    Image,
    ImageAttachment,
    Module,
    PersistentAllocator,
    PipelineId,
    RenderGraph,
    SuperFrameAllocator,
    SwapChain,
    ValueId,
};
pub use winit;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

pub use crate::window::Window;
use crate::{
    device_builder::{DeviceBuilder, InstanceBuilder, PhysicalDeviceSelector, SwapChainBuilder},
    egui_pass::EguiPass,
    window::{create_surface, surface_extension},
};

const EGUI_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/egui.vert.spv"));
const EGUI_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/egui.frag.spv"));

/// Turns a `.spv` blob into the words a pipeline is declared from.
pub fn read_spirv(bytes: &[u8]) -> Vec<u32> {
    ash::util::read_spv(&mut Cursor::new(bytes)).expect("shader is not valid SPIR-V")
}

/// Declares a pipeline from one vertex and one fragment blob. Everything else about it, down to
/// the vertex layout, comes out of reflection.
pub fn graphics_pipeline(graph: &mut RenderGraph, vertex: &[u8], fragment: &[u8]) -> Result<PipelineId, vk::Result> {
    graph.declare_pipeline(
        GraphicsPipelineInfo::new()
            .with_shader(&read_spirv(vertex))
            .with_shader(&read_spirv(fragment)),
    )
}

/// What the swapchain currently looks like, which is what any target an example renders into
/// has to match.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

/// What an example gets to build itself out of, both at startup and after every resize.
pub struct Setup<'a> {
    pub ctx: &'a Context,
    pub graph: &'a mut RenderGraph,
    pub allocator: &'a mut PersistentAllocator,
    pub target: Target,
}

/// One frame in flight. The example records into `module` and hands back the value the harness
/// should present, after it has drawn the UI over it.
pub struct Frame<'a> {
    pub ctx: &'a Context,
    pub graph: &'a mut RenderGraph,
    pub module: &'a mut Module,
    /// Whatever this frame allocates out of is recycled once the frame retires, so it is where
    /// geometry rebuilt every frame belongs.
    pub allocator: &'a mut FrameAllocator,
    /// The acquired swapchain image, already imported into `module`.
    pub swapchain_image: ValueId,
    pub target: Target,
    /// Seconds since the example started, for anything that animates.
    pub elapsed: f32,
    roots: Vec<ValueId>,
}

impl Frame<'_> {
    /// Marks `value` as something the frame has to produce even though presenting does not
    /// depend on it, which is how a resource handed back to the example outlives the frame.
    pub fn add_root(&mut self, value: ValueId) { self.roots.push(value); }
}

pub trait Example: Sized {
    /// The window title, which is also how the example names itself in its own UI.
    const TITLE: &'static str;

    fn new(setup: &mut Setup) -> Result<Self, vk::Result>;

    /// Called after every swapchain creation, including the first, for targets that have to
    /// track the window size.
    fn resize(&mut self, _setup: &mut Setup) -> Result<(), vk::Result> { Ok(()) }

    /// The panel the harness draws over whatever the example rendered.
    fn ui(&mut self, _ctx: &egui::Context) {}

    /// Records this frame and returns the image to present.
    fn render(&mut self, frame: &mut Frame) -> Result<ValueId, vk::Result>;

    /// Hands back anything taken from the persistent allocator. The device is already idle.
    fn destroy(&mut self, _graph: &mut RenderGraph, _allocator: &mut PersistentAllocator) {}
}

/// Runs `E` until the window closes.
pub fn run<E: Example>() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut wrapper = Host::<E> { app: None };
    event_loop.run_app(&mut wrapper)?;

    Ok(())
}

/// Everything that is the same whatever the example is.
struct Renderer {
    /// The loader everything else was created through, which has to outlive all of it.
    _ash_entry: ash::Entry,
    window: Window,
    surface: vk::SurfaceKHR,
    ctx: Context,
    graph: RenderGraph,
    persistent_allocator: PersistentAllocator,
    swapchain: Option<SwapChain>,
    super_frame_allocator: Option<SuperFrameAllocator>,
    target: Target,
}

impl Renderer {
    fn new(window: Window) -> Result<Self, Box<dyn Error>> {
        let raw_window_handle = window.raw_window_handle();
        let raw_display_handle = window.raw_display_handle();
        let ash_entry = unsafe { Entry::load()? };

        let instance = InstanceBuilder::default()
            .require_api_version(1, 3, 0)
            .require_extension(khr::surface::NAME.to_owned())
            .require_extension(khr::get_surface_capabilities2::NAME.to_owned())
            .require_extension(surface_extension(Some(raw_window_handle))?.to_owned())
            .require_extension(khr::get_physical_device_properties2::NAME.to_owned())
            .require_surface_extensions()
            .set_app_name(c"Example".to_owned())
            .set_app_version(0, 0, 0)
            .set_engine_name(c"vir".to_owned())
            .set_engine_version(0, 0, 0)
            .build(&ash_entry)?;

        let physical_device = PhysicalDeviceSelector::default()
            .set_min_api_version(1, 3, 0)
            .add_required_extension(khr::swapchain::NAME.to_owned())
            .set_preferred_device_type(vk::PhysicalDeviceType::DISCRETE_GPU)
            .allow_any_device_type(true)
            .require_separate_compute_queue(true)
            .select(&instance)?;

        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true)
            .shader_demote_to_helper_invocation(true);
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .timeline_semaphore(true)
            .buffer_device_address(true)
            .scalar_block_layout(true);
        let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default()
            .variable_pointers(true)
            .variable_pointers_storage_buffer(true)
            .shader_draw_parameters(true);
        let vk10_features = vk::PhysicalDeviceFeatures::default().fill_mode_non_solid(true);
        let features = vk::PhysicalDeviceFeatures2::default()
            .features(vk10_features)
            .push_next(&mut vk11_features)
            .push_next(&mut vk12_features)
            .push_next(&mut vk13_features);

        let device = DeviceBuilder::default()
            .set_features(features)
            .build(&instance, &physical_device)?;

        let mut ctx = Context::new(device, physical_device.handle, instance, &ash_entry)?;
        let graphics_queue_index = physical_device
            .get_queue_index(vk::QueueFlags::GRAPHICS)
            .expect("No graphics queue");
        ctx.create_command_queue(graphics_queue_index, vir::DomainFlag::Graphics);

        let persistent_allocator = ctx.create_persistent_allocator();
        let surface = create_surface(&ash_entry, ctx.instance(), raw_window_handle, raw_display_handle)?;
        let graph = RenderGraph::new(&ctx);

        Ok(Self {
            _ash_entry: ash_entry,
            window,
            surface,
            ctx,
            graph,
            persistent_allocator,
            swapchain: None,
            super_frame_allocator: None,
            target: Target {
                extent: vk::Extent2D::default(),
                format: vk::Format::UNDEFINED,
            },
        })
    }

    fn recreate_swapchain(&mut self) -> Result<(), vk::Result> {
        let extent = self.window.extent();
        let physical_device = self.ctx.physical_device();
        let swapchain_loader = self.ctx.swapchain_loader();
        let surface_loader = self.ctx.surface_loader();

        let old_swapchain = self.swapchain.as_ref().map_or(vk::SwapchainKHR::null(), |s| s.handle);
        let (swapchain_handle, format, extent) = SwapChainBuilder::new(*physical_device)
            .set_desired_extent(extent.width, extent.height)
            .set_old_swapchain(old_swapchain)
            .build(surface_loader, &self.surface, swapchain_loader)?;

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain_handle) }?;
        let attachments = swapchain_images
            .into_iter()
            .map(|image_handle| {
                let extent = vk::Extent3D::default()
                    .width(extent.width)
                    .height(extent.height)
                    .depth(1);
                let image = Image::imported(image_handle, format, extent, vk::SampleCountFlags::TYPE_1);
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1);

                ImageAttachment::new(
                    image,
                    format,
                    extent,
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageLayout::UNDEFINED,
                )
                .with_subresource_range(subresource_range)
            })
            .collect::<Vec<_>>();

        let image_count = attachments.len();
        self.swapchain = Some(SwapChain::new(
            &mut self.persistent_allocator,
            swapchain_handle,
            self.surface,
            attachments,
        )?);
        self.super_frame_allocator = Some(self.ctx.create_super_frame_allocator(image_count));
        self.target = Target { extent, format };

        Ok(())
    }

    fn setup(&mut self) -> Setup<'_> {
        Setup {
            ctx: &self.ctx,
            graph: &mut self.graph,
            allocator: &mut self.persistent_allocator,
            target: self.target,
        }
    }
}

struct App<E: Example> {
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_pass: EguiPass,
    example: E,
    start_time: Instant,
    dumped_ir: bool,
}

impl<E: Example> App<E> {
    fn new(window: Window) -> Result<Self, Box<dyn Error>> {
        let mut renderer = Renderer::new(window)?;

        let egui_pass = EguiPass::new(
            &mut renderer.graph,
            &read_spirv(EGUI_VERT_SPV),
            &read_spirv(EGUI_FRAG_SPV),
        )?;
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &renderer.window.handle,
            Some(renderer.window.handle.scale_factor() as f32),
            None,
            None,
        );

        // the example is built against a live swapchain, so `new` can lean on the same
        // size-dependent setup `resize` does
        renderer.recreate_swapchain()?;
        let example = E::new(&mut renderer.setup())?;

        Ok(Self {
            renderer,
            egui_ctx,
            egui_state,
            egui_pass,
            example,
            start_time: Instant::now(),
            dumped_ir: false,
        })
    }

    fn recreate_swapchain(&mut self) -> Result<(), vk::Result> {
        self.renderer.recreate_swapchain()?;
        self.example.resize(&mut self.renderer.setup())
    }

    fn render(&mut self) -> Result<(), vk::Result> {
        let renderer = &mut self.renderer;
        let Some(ref swapchain) = renderer.swapchain else {
            return Err(vk::Result::ERROR_SURFACE_LOST_KHR);
        };
        let target = renderer.target;

        let raw_input = self.egui_state.take_egui_input(&renderer.window.handle);
        let example = &mut self.example;
        let egui_output = self.egui_ctx.run_ui(raw_input, |root| example.ui(root.ctx()));
        self.egui_state
            .handle_platform_output(&renderer.window.handle, egui_output.platform_output);

        let pixels_per_point = egui_output.pixels_per_point;
        let primitives = self.egui_ctx.tessellate(egui_output.shapes, pixels_per_point);

        let next_frame = renderer
            .super_frame_allocator
            .as_mut()
            .expect("swapchain must be created before rendering")
            .get_next_frame()?;

        let egui_frame = self.egui_pass.prepare(
            &renderer.ctx,
            &mut renderer.graph,
            &mut renderer.persistent_allocator,
            next_frame,
            egui_output.textures_delta,
            &primitives,
            pixels_per_point,
            target.extent,
        )?;
        tracing::trace!(
            primitives = primitives.len(),
            draws = egui_frame.draw_count(),
            uploads = egui_frame.upload_count(),
            pixels_per_point,
            "egui frame"
        );

        let mut module = Module::default();
        let swapchain_image = module.acquire_next_image(swapchain);

        let mut frame = Frame {
            ctx: &renderer.ctx,
            graph: &mut renderer.graph,
            module: &mut module,
            allocator: &mut *next_frame,
            swapchain_image,
            target,
            elapsed: self.start_time.elapsed().as_secs_f32(),
            roots: Vec::new(),
        };
        let drawn = self.example.render(&mut frame)?;
        let mut roots = std::mem::take(&mut frame.roots);
        drop(frame);

        let drawn = self.egui_pass.record(&mut module, drawn, &egui_frame);
        roots.insert(0, module.present(drawn));
        let executable = module.compile_all(&roots);

        // the first frame egui produces is the font atlas upload with nothing drawn yet, which
        // is not the one worth looking at
        if !self.dumped_ir && egui_frame.draw_count() > 0 {
            if std::env::var_os("VIR_DUMP_IR").is_some() {
                RenderGraph::dump(&executable);
            }
            self.dumped_ir = true;
        }

        renderer
            .graph
            .execute(&renderer.ctx, &executable, &mut AllocatorKind::Frame(next_frame))
    }
}

impl<E: Example> Drop for App<E> {
    fn drop(&mut self) {
        if let Err(err) = unsafe { self.renderer.ctx.device().device_wait_idle() } {
            tracing::error!(?err, "failed to wait for the device before tearing the app down");
            return;
        }

        self.example
            .destroy(&mut self.renderer.graph, &mut self.renderer.persistent_allocator);
        self.egui_pass
            .destroy(&mut self.renderer.graph, &mut self.renderer.persistent_allocator);
    }
}

/// The winit side, which only exists because the app cannot be built before `resumed`.
struct Host<E: Example> {
    app: Option<App<E>>,
}

impl<E: Example> ApplicationHandler for Host<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Window::new(
            event_loop,
            E::TITLE,
            vk::Extent2D {
                width: 1020,
                height: 780,
            },
        )
        .expect("Cannot create new window");

        self.app = Some(App::new(window).expect("Cannot initialize app"));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else {
            return;
        };

        let _ = app.egui_state.on_window_event(&app.renderer.window.handle, &event);

        match event {
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                app.recreate_swapchain()
                    .expect("Failed to recreate swapchain on resize");
            },
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                app.renderer.window.handle.request_redraw();
                match app.render() {
                    Ok(()) => {},
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain().expect("Failed to recreate swapchain");
                    },
                    Err(e) => panic!("Render error: {e}"),
                }
            },
            _ => {},
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) { self.app = None; }
}
