mod device_builder;

use std::{error::Error, ffi::CStr, io::Cursor, result::Result, time::Instant};

use ash::{Entry, khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use vir::{
    AllocatorKind,
    BlendPreset,
    Buffer,
    BufferInfo,
    ClearValue,
    Context,
    DynamicStateFlags,
    GraphicsPipelineInfo,
    Image,
    ImageAttachment,
    PersistentAllocator,
    PipelineId,
    RasterizationState,
    Rect2D,
    RenderGraph,
    SuperFrameAllocator,
    SwapChain,
    allocator::Allocator,
};
pub use winit;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    error::OsError,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window as WinitWindow, WindowId},
};

use crate::device_builder::{DeviceBuilder, InstanceBuilder, PhysicalDeviceSelector, SwapChainBuilder};

fn get_surface_extension(handle: Option<RawWindowHandle>) -> Result<&'static CStr, vk::Result> {
    Ok(match handle {
        Some(handle) => match handle {
            RawWindowHandle::Win32(_) => khr::win32_surface::NAME,
            RawWindowHandle::Wayland(_) => khr::wayland_surface::NAME,
            RawWindowHandle::Xlib(_) => khr::xlib_surface::NAME,
            RawWindowHandle::Xcb(_) => khr::xcb_surface::NAME,
            _ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
        },
        None => {
            if cfg!(target_os = "windows") {
                khr::win32_surface::NAME
            } else if cfg!(target_os = "linux") {
                khr::xlib_surface::NAME
            } else {
                return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
            }
        },
    })
}

fn create_surface(
    entry: &ash::Entry, instance: &ash::Instance, window: RawWindowHandle, display: RawDisplayHandle,
) -> Result<vk::SurfaceKHR, vk::Result> {
    unsafe {
        match (window, display) {
            (RawWindowHandle::Win32(handle), _) => {
                let surface_fn = khr::win32_surface::Instance::new(entry, instance);
                surface_fn.create_win32_surface(
                    &vk::Win32SurfaceCreateInfoKHR::default()
                        .hinstance(handle.hinstance.map_or(0, |x| x.get()))
                        .hwnd(handle.hwnd.get()),
                    None,
                )
            },
            (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
                let surface_fn = khr::wayland_surface::Instance::new(entry, instance);
                surface_fn.create_wayland_surface(
                    &vk::WaylandSurfaceCreateInfoKHR::default()
                        .display(display.display.as_ptr())
                        .surface(window.surface.as_ptr()),
                    None,
                )
            },
            (RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display)) => {
                let surface_fn = khr::xlib_surface::Instance::new(entry, instance);
                surface_fn.create_xlib_surface(
                    &vk::XlibSurfaceCreateInfoKHR::default()
                        .dpy(display.display.map_or(std::ptr::null_mut(), |x| x.as_ptr()))
                        .window(window.window),
                    None,
                )
            },
            (RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => {
                let surface_fn = khr::xcb_surface::Instance::new(entry, instance);
                surface_fn.create_xcb_surface(
                    &vk::XcbSurfaceCreateInfoKHR::default()
                        .connection(display.connection.map_or(std::ptr::null_mut(), |x| x.as_ptr()))
                        .window(window.window.get()),
                    None,
                )
            },
            _ => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
        }
    }
}

const TRIANGLE_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
const TRIANGLE_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));
const TRIANGLE_BUFFER_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle_buffer.vert.spv"));

/// One vertex as `vs_buffer` reads it: `float2 position` then `float3 color`, which is exactly
/// what reflection packs into binding 0.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

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

const STATIC_TRIANGLE: [Vertex; 3] = [
    Vertex {
        position: [-0.9, -0.1],
        color: [1.0, 1.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.9],
        color: [0.0, 1.0, 1.0],
    },
    Vertex {
        position: [-0.1, -0.1],
        color: [1.0, 0.0, 1.0],
    },
];

/// The same triangle on the other side, spun by `angle`, rebuilt every frame.
fn spinning_triangle(angle: f32) -> [Vertex; 3] {
    let (sin, cos) = angle.sin_cos();
    let colors = [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]];

    std::array::from_fn(|index| {
        let corner = std::f32::consts::TAU * index as f32 / 3.0;
        let (x, y) = (0.3 * corner.cos(), 0.3 * corner.sin());
        Vertex {
            position: [0.5 + x * cos - y * sin, 0.5 + x * sin + y * cos],
            color: colors[index],
        }
    })
}

struct App {
    ash_entry: ash::Entry,
    window: Window,
    surface: vk::SurfaceKHR,
    swapchain: Option<SwapChain>,
    // everything below borrows the device and the memory allocator from ctx, so it has to drop first
    graph: RenderGraph,
    super_frame_allocator: Option<SuperFrameAllocator>,
    persistent_allocator: PersistentAllocator,
    ctx: Context,
    triangle_pipeline: PipelineId,
    vertex_pipeline: PipelineId,
    static_vertices: Buffer,
    dumped_ir: bool,
    start_time: Instant,
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

fn read_spirv(bytes: &[u8]) -> Vec<u32> {
    ash::util::read_spv(&mut Cursor::new(bytes)).expect("shader is not valid SPIR-V")
}

#[derive(Default)]
struct AppWrapper {
    app: Option<App>,
}

impl App {
    fn new(
        window: Window, surface: vk::SurfaceKHR, ash_entry: ash::Entry, ctx: Context,
        mut persistent_allocator: PersistentAllocator,
    ) -> Result<Self, vk::Result> {
        let mut graph = RenderGraph::new(&ctx);

        let triangle_pipeline = graph.declare_pipeline(
            GraphicsPipelineInfo::new()
                .with_shader(&read_spirv(TRIANGLE_VERT_SPV))
                .with_shader(&read_spirv(TRIANGLE_FRAG_SPV)),
        )?;

        let vertex_pipeline = graph.declare_pipeline(
            GraphicsPipelineInfo::new()
                .with_shader(&read_spirv(TRIANGLE_BUFFER_VERT_SPV))
                .with_shader(&read_spirv(TRIANGLE_FRAG_SPV)),
        )?;

        let mut static_vertices = persistent_allocator
            .allocate_buffer(&BufferInfo::vertex(size_of_val(&STATIC_TRIANGLE) as u64).with_name("static triangle"))?;
        static_vertices.write(0, &STATIC_TRIANGLE)?;

        Ok(Self {
            window,
            ash_entry,
            surface,
            swapchain: None,
            graph,
            super_frame_allocator: None,
            persistent_allocator,
            ctx,
            triangle_pipeline,
            vertex_pipeline,
            static_vertices,
            dumped_ir: false,
            start_time: Instant::now(),
        })
    }

    fn init(window: Window) -> Result<Self, Box<dyn Error>> {
        let raw_window_handle = window.raw_window_handle();
        let raw_display_handle = window.raw_display_handle();
        let ash_entry = unsafe { Entry::load()? };

        let instance = InstanceBuilder::default()
            .require_api_version(1, 3, 0)
            .require_extension(khr::surface::NAME.to_owned())
            .require_extension(khr::get_surface_capabilities2::NAME.to_owned())
            .require_extension(get_surface_extension(Some(raw_window_handle))?.to_owned())
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

        Ok(Self::new(window, surface, ash_entry, ctx, persistent_allocator)?)
    }

    fn create_swapchain(&mut self, extent: vk::Extent2D) -> Result<SwapChain, vk::Result> {
        let physical_device = self.ctx.physical_device();
        let swapchain_loader = self.ctx.swapchain_loader();
        let surface_loader = self.ctx.surface_loader();

        let old_swapchain = self.swapchain.as_ref().map_or(vk::SwapchainKHR::null(), |s| s.handle);
        let (swapchain_handle, swapchain_format, swapchain_extent) = SwapChainBuilder::new(*physical_device)
            .set_desired_extent(extent.width, extent.height)
            .set_old_swapchain(old_swapchain)
            .build(surface_loader, &self.surface, swapchain_loader)?;

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain_handle) }?;
        let attachments = swapchain_images
            .into_iter()
            .map(|image_handle| {
                let image = Image::new(image_handle, None);
                let extent = vk::Extent3D::default()
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .depth(1);
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1);

                ImageAttachment::new(
                    image,
                    swapchain_format,
                    extent,
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageLayout::UNDEFINED,
                )
                .with_subresource_range(subresource_range)
            })
            .collect::<Vec<_>>();

        let image_count = attachments.len();
        let swapchain = SwapChain::new(
            &mut self.persistent_allocator,
            swapchain_handle,
            self.surface,
            attachments,
        )?;
        self.super_frame_allocator = Some(self.ctx.create_super_frame_allocator(image_count));

        Ok(swapchain)
    }

    fn recreate_swapchain(&mut self) -> Result<(), vk::Result> {
        let size = self.window.handle.inner_size();
        let extent = vk::Extent2D {
            width: size.width,
            height: size.height,
        };
        self.swapchain = Some(self.create_swapchain(extent)?);
        Ok(())
    }

    fn run(&mut self) -> Result<(), vk::Result> {
        let Some(ref swapchain) = self.swapchain else {
            return Err(vk::Result::ERROR_SURFACE_LOST_KHR);
        };

        let next_frame = self
            .super_frame_allocator
            .as_mut()
            .expect("swapchain must be created before rendering")
            .get_next_frame()?;

        let elapsed = self.start_time.elapsed().as_secs_f32();
        let spinning = spinning_triangle(elapsed);
        let mut spinning_vertices = next_frame
            .allocate_buffer(&BufferInfo::vertex(size_of_val(&spinning) as u64).with_name("spinning triangle"))?;
        spinning_vertices.write(0, &spinning)?;

        let mut frame_allocator = AllocatorKind::Frame(next_frame);

        let mut module = vir::Module::default();
        let attachment = module.acquire_next_image(swapchain);
        let attachment = module.clear(attachment, rainbow(elapsed * 0.2));

        let static_vertices = module.import_buffer(&self.static_vertices);
        let spinning_vertices = module.import_buffer(&spinning_vertices);

        let sliding = PushConstants {
            offset: [0.25 * elapsed.sin(), 0.0],
            scale: 1.0,
            tint: 0.0,
        };

        let attachment = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(self.triangle_pipeline)
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

        let attachment = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(self.triangle_pipeline)
            .set_dynamic_state(DynamicStateFlags::Viewport | DynamicStateFlags::Scissor)
            .set_viewport(0, Rect2D::relative(0.5, 0.5, 0.5, 0.5))
            .set_scissor(0, Rect2D::relative(0.5, 0.5, 0.5, 0.5))
            .broadcast_color_blend(BlendPreset::AlphaBlend)
            .push_constants(&PushConstants {
                scale: 0.5 + 0.5 * elapsed.cos(),
                ..sliding
            })
            .draw(3, 1)
            .end_rendering();

        let attachment = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(self.vertex_pipeline)
            .set_viewport(0, Rect2D::framebuffer())
            .set_scissor(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            })
            .push_constants_at(TINT_OFFSET, &0.0f32)
            .bind_vertex_buffer(0, static_vertices)
            .draw(3, 1)
            .push_constants_at(TINT_OFFSET, &(0.5 + 0.5 * elapsed.sin()))
            .bind_vertex_buffer(0, spinning_vertices)
            .draw(3, 1)
            .end_rendering();

        let attachment = module.present(attachment);
        let executable = module.compile(attachment);

        if !self.dumped_ir {
            RenderGraph::dump(executable.as_slice());
            self.dumped_ir = true;
        }

        self.graph
            .execute(&self.ctx, executable.as_slice(), &mut frame_allocator)
    }
}

impl ApplicationHandler for AppWrapper {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Window::new(
            event_loop,
            "Lorr",
            vk::Extent2D {
                width: 1020,
                height: 780,
            },
        )
        .expect("Cannot create new window");

        let mut app = App::init(window).expect("Cannot initialize app");

        app.recreate_swapchain().expect("Failed to create swapchain");

        self.app = Some(app);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else {
            return;
        };

        match event {
            WindowEvent::Resized(_) => {
                app.recreate_swapchain()
                    .expect("Failed to recreate swapchain on resize");
            },
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                app.window.handle.request_redraw();
                match app.run() {
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

    fn exiting(&mut self, _: &ActiveEventLoop) {
        // drops the graph, which waits for the device and tears down its pipelines
        self.app = None;
    }
}

struct Window {
    handle: WinitWindow,
}

impl Window {
    fn new(event_loop: &ActiveEventLoop, title: &str, size: vk::Extent2D) -> Result<Self, OsError> {
        let window_attribs = WinitWindow::default_attributes()
            .with_title(title)
            .with_inner_size(LogicalSize::new(size.width, size.height));

        Ok(Self {
            handle: event_loop.create_window(window_attribs)?,
        })
    }

    fn raw_window_handle(&self) -> RawWindowHandle { self.handle.window_handle().unwrap().as_raw() }

    fn raw_display_handle(&self) -> RawDisplayHandle { self.handle.display_handle().unwrap().as_raw() }
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app_wrapper = AppWrapper::default();
    let _ = event_loop.run_app(&mut app_wrapper);

    Ok(())
}

#[cfg(test)]
mod tests {
    use vir::{
        VertexAttribute,
        VertexLayout,
        resource::{pipeline::push_constant_ranges, shader},
    };

    use super::*;

    /// The whole chain: Slang emits POSITION/COLOR as locations, reflection reads them back, and
    /// the layout that comes out has to match the `Vertex` struct the example uploads.
    #[test]
    fn the_reflected_vertex_layout_matches_the_uploaded_vertex() {
        let reflections = [
            shader::reflect(&read_spirv(TRIANGLE_BUFFER_VERT_SPV)).expect("vertex shader should reflect"),
            shader::reflect(&read_spirv(TRIANGLE_FRAG_SPV)).expect("fragment shader should reflect"),
        ];

        let layout = VertexLayout::interleaved(&reflections);
        assert_eq!(layout.stride as usize, size_of::<Vertex>());
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
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 8,
                },
            ]
        );
    }

    /// The SV_VertexID triangle must keep compiling to a pipeline with no vertex input at all.
    #[test]
    fn the_vertex_id_shader_reflects_no_attributes() {
        let reflections = [shader::reflect(&read_spirv(TRIANGLE_VERT_SPV)).expect("vertex shader should reflect")];
        assert_eq!(VertexLayout::interleaved(&reflections), VertexLayout::default());
    }

    /// What `push_constants` sends has to be the block Slang laid out, or the shader reads
    /// whatever the neighbouring member happened to be.
    #[test]
    fn the_reflected_push_constant_block_matches_the_pushed_struct() {
        for spirv in [TRIANGLE_VERT_SPV, TRIANGLE_FRAG_SPV] {
            let reflection = shader::reflect(&read_spirv(spirv)).expect("shader should reflect");
            assert_eq!(reflection.push_constant_offset, 0);
            assert_eq!(reflection.push_constant_size as usize, size_of::<PushConstants>());
        }

        // TINT_OFFSET has to name the last member, since the geometry draws push it alone
        assert_eq!(TINT_OFFSET as usize, size_of::<PushConstants>() - size_of::<f32>());
    }

    /// `vs_buffer` ignores the block, so its pipeline gets a fragment-only range over it while
    /// the SV_VertexID pipeline gets one range both stages share.
    #[test]
    fn a_stage_that_ignores_the_block_stays_out_of_its_range() {
        let reflect = |spirv| shader::reflect(&read_spirv(spirv)).expect("shader should reflect");

        let shared = push_constant_ranges(&[reflect(TRIANGLE_VERT_SPV), reflect(TRIANGLE_FRAG_SPV)]);
        assert_eq!(shared.len(), 1);
        assert_eq!(
            shared[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(shared[0].size as usize, size_of::<PushConstants>());

        let fragment_only = push_constant_ranges(&[reflect(TRIANGLE_BUFFER_VERT_SPV), reflect(TRIANGLE_FRAG_SPV)]);
        assert_eq!(fragment_only.len(), 1);
        assert_eq!(fragment_only[0].stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }
}
