//! The scaffolding every example sits on.
//!
//! An example is a type that implements [`Example`] and a `main` that hands it to [`run`]. The
//! harness owns everything that is not the point of any one example: the window, the instance
//! and device, the swapchain and its allocators, the render graph, and the egui overlay that
//! every example draws its controls into.
//!
//! A frame is one program, compiled by [`compile_example`] and re-run until the swapchain
//! changes. The example records its passes into it, the egui pass records the UI over them, and
//! the present closes it, so the graph is handed the whole frame at once and works out every
//! barrier in it. What an example does per frame is write into the slots it declared while
//! recording, and nothing it does there reaches the graph at all.

pub mod device_builder;
pub mod egui_pass;
mod window;

use std::{
    collections::HashSet,
    error::Error,
    io::Cursor,
    result::Result,
    sync::{LazyLock, Mutex},
    time::Instant,
};

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
    Program,
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
    egui_pass::{EguiPass, EguiSlots},
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

/// Prints `program` under `VIR_DUMP_IR`, once per `name`. The harness dumps the frame it
/// compiled; a module an example builds and runs itself is dumped by asking for it here.
pub fn dump_ir(name: &str, program: &vir::Program) {
    static DUMPED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Mutex::default);

    if std::env::var_os("VIR_DUMP_IR").is_none() {
        return;
    }

    let mut dumped = DUMPED.lock().expect("the dumped set should not be poisoned");
    if dumped.insert(name.to_owned()) {
        println!("; program \"{name}\"");
        RenderGraph::dump(program);
    }
}

/// What the swapchain currently looks like, which is what any target an example renders into
/// has to match.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

/// What an example is built out of at startup, before there is a frame to record.
pub struct Setup<'a> {
    pub ctx: &'a Context,
    pub graph: &'a mut RenderGraph,
    pub allocator: &'a mut PersistentAllocator,
    pub target: Target,
}

/// The module the whole frame is compiled from.
///
/// An example records its passes here once and reaches them afterwards through the variables
/// it declared alongside them, so nothing about a frame is rebuilt until the swapchain is.
/// Anything sized to the window belongs here too, since this is the one place that runs again
/// when the window changes size.
pub struct Recording<'a> {
    pub ctx: &'a Context,
    pub graph: &'a mut RenderGraph,
    pub module: &'a mut Module,
    /// The device is idle here, so whatever the last program was built around can be handed
    /// back before the replacement is taken.
    pub allocator: &'a mut PersistentAllocator,
    /// The image this run acquires, which is what the example draws into or blits onto.
    pub swapchain_image: ValueId,
    pub target: Target,
}

/// One frame in flight. What an example does here is write into the program it recorded, which
/// is why none of this reaches the graph.
pub struct Frame<'a> {
    pub ctx: &'a Context,
    pub program: &'a mut Program,
    /// Whatever this frame allocates out of is recycled once the frame retires, so it is where
    /// geometry rebuilt every frame belongs.
    pub allocator: &'a mut FrameAllocator,
    pub target: Target,
    /// Seconds since the example started, for anything that animates.
    pub elapsed: f32,
}

pub trait Example: Sized {
    /// The window title, which is also how the example names itself in its own UI.
    const TITLE: &'static str;

    fn new(setup: &mut Setup) -> Result<Self, vk::Result>;

    /// Records the example's passes and hands back the image to present. Called once per
    /// swapchain rather than once per frame, so what it returns is a graph, not a picture.
    fn record(&mut self, recording: &mut Recording) -> Result<ValueId, vk::Result>;

    /// Writes this frame into the slots [`Example::record`] declared.
    fn update(&mut self, _frame: &mut Frame) -> Result<(), vk::Result> { Ok(()) }

    /// The panel the harness draws over whatever the example rendered.
    fn ui(&mut self, _ctx: &egui::Context) {}

    /// Hands back anything taken from the persistent allocator. The device is already idle.
    fn destroy(&mut self, _graph: &mut RenderGraph, _allocator: &mut PersistentAllocator) {}
}

/// The one program a frame is: the example's passes, the UI over them, and the present.
pub struct CompiledFrame {
    pub program: Program,
    egui: EguiSlots,
}

/// Compiles that program.
///
/// The example records first and hands back what it drew, the egui pass goes over it, and the
/// present closes it. Keeping the UI in the same module as the passes under it is what leaves
/// the graph a single frame to order: the layout the UI wants its target in is one the barrier
/// pass works out, rather than something the example has to leave it in for a second run.
pub fn compile_example<E: Example>(
    example: &mut E, egui_pass: &EguiPass, ctx: &Context, graph: &mut RenderGraph, allocator: &mut PersistentAllocator,
    swapchain: &SwapChain, target: Target,
) -> Result<CompiledFrame, vk::Result> {
    let mut module = Module::default();
    let swapchain_image = module.acquire_next_image(swapchain);

    let mut recording = Recording {
        ctx,
        graph,
        module: &mut module,
        allocator,
        swapchain_image,
        target,
    };
    let drawn = example.record(&mut recording)?;

    let (drawn, egui) = egui_pass.record(&mut module, drawn);
    let present = module.present(drawn);

    let program = module.compile(present);
    dump_ir("frame", &program);

    Ok(CompiledFrame { program, egui })
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
    frame: Option<CompiledFrame>,
    start_time: Instant,
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

        // the example is built against a live swapchain, since everything it records is sized
        // to one
        renderer.recreate_swapchain()?;
        let example = E::new(&mut renderer.setup())?;

        let mut app = Self {
            renderer,
            egui_ctx,
            egui_state,
            egui_pass,
            example,
            frame: None,
            start_time: Instant::now(),
        };
        app.compile()?;

        Ok(app)
    }

    /// Rebuilds the frame's program, which is the only thing a resize costs: the example gets
    /// to hand back whatever it sized to the old swapchain and record against the new one.
    fn compile(&mut self) -> Result<(), vk::Result> {
        let renderer = &mut self.renderer;
        let Some(ref swapchain) = renderer.swapchain else {
            return Err(vk::Result::ERROR_SURFACE_LOST_KHR);
        };

        // the program being replaced may still be in flight, and it is what holds every
        // resource the example is about to hand back
        unsafe { renderer.ctx.device().device_wait_idle() }?;
        self.frame = None;

        self.frame = Some(compile_example(
            &mut self.example,
            &self.egui_pass,
            &renderer.ctx,
            &mut renderer.graph,
            &mut renderer.persistent_allocator,
            swapchain,
            renderer.target,
        )?);

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<(), vk::Result> {
        self.renderer.recreate_swapchain()?;
        self.compile()
    }

    fn render(&mut self) -> Result<(), vk::Result> {
        let renderer = &mut self.renderer;
        let Some(frame) = self.frame.as_mut() else {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
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

        // the whole of a frame is writes into an already compiled program
        self.example.update(&mut Frame {
            ctx: &renderer.ctx,
            program: &mut frame.program,
            allocator: &mut *next_frame,
            target,
            elapsed: self.start_time.elapsed().as_secs_f32(),
        })?;
        self.egui_pass.bind(&mut frame.program, &frame.egui, &egui_frame);

        renderer
            .graph
            .execute(&renderer.ctx, &frame.program, &mut AllocatorKind::Frame(next_frame))
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
