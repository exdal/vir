mod device_builder;

use std::{error::Error, ffi::CStr, result::Result};

use ash::{Entry, khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use vir::{
    AllocatorKind,
    Context,
    Image,
    ImageAttachment,
    PersistentAllocator,
    RenderGraph,
    SuperFrameAllocator,
    SwapChain,
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

#[derive(Default)]
struct App {
    ash_entry: Option<ash::Entry>,
    window: Option<Window>,
    swapchain: SwapChain,
    ctx: Option<Context>,
    super_frame_allocator: Option<SuperFrameAllocator>,
    persistent_allocator: Option<PersistentAllocator>,
}

impl App {
    fn init(&mut self, window: Window) -> Result<(), Box<dyn Error>> {
        let raw_window_handle = window.raw_window_handle();
        let raw_display_handle = window.raw_display_handle();
        let window_inner_size = window.handle.inner_size();
        let window_extent = vk::Extent2D {
            width: window_inner_size.width,
            height: window_inner_size.height,
        };
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
            .variable_pointers_storage_buffer(true);
        let vk10_features = vk::PhysicalDeviceFeatures::default();
        let features = vk::PhysicalDeviceFeatures2::default()
            .features(vk10_features)
            .push_next(&mut vk11_features)
            .push_next(&mut vk12_features)
            .push_next(&mut vk13_features);

        let device = DeviceBuilder::default()
            .set_features(features)
            .build(&instance, &physical_device)?;

        let mut ctx = Context::new(device.into(), physical_device.handle, instance.into(), &ash_entry);
        let graphics_queue_index = physical_device
            .get_queue_index(vk::QueueFlags::GRAPHICS)
            .expect("No graphics queue");
        ctx.create_command_queue(graphics_queue_index, vir::DomainFlag::Graphics);
        self.window = Some(window);
        self.ash_entry = Some(ash_entry);
        self.persistent_allocator = Some(ctx.create_persistent_allocator());
        self.super_frame_allocator = Some(ctx.create_super_frame_allocator(3));
        self.ctx = Some(ctx);
        self.swapchain = self.create_swapchain(window_extent, raw_window_handle, raw_display_handle)?;

        Ok(())
    }

    fn create_swapchain(
        &mut self, extent: vk::Extent2D, raw_window_handle: RawWindowHandle, raw_display_handle: RawDisplayHandle,
    ) -> Result<SwapChain, vk::Result> {
        let runtime = self.ctx.as_ref().unwrap();

        let ash_entry = self.ash_entry.as_ref().unwrap();
        let instance = runtime.instance();
        let physical_device = runtime.physical_device();
        let swapchain_loader = runtime.swapchain_loader();
        let surface_loader = runtime.surface_loader();

        let surface = create_surface(ash_entry, instance, raw_window_handle, raw_display_handle)?;
        let (swapchain_handle, swapchain_format, swapchain_extent) = SwapChainBuilder::new(*physical_device)
            .set_desired_extent(extent.width, extent.height)
            .set_old_swapchain(self.swapchain.handle)
            .build(surface_loader, &surface, swapchain_loader)?;

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain_handle) }?;
        let attachments = swapchain_images
            .into_iter()
            .map(|image_handle| {
                let image = Image::new(image_handle, None);
                let extent = vk::Extent3D::default()
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height);

                ImageAttachment::new(
                    image,
                    swapchain_format,
                    extent,
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageLayout::UNDEFINED,
                )
            })
            .collect::<Vec<_>>();

        SwapChain::new(
            self.persistent_allocator.as_mut().unwrap(),
            swapchain_handle,
            surface,
            attachments,
        )
    }

    fn run(&mut self) {
        let ctx = self.ctx.as_ref().unwrap();
        let super_frame_allocator = self.super_frame_allocator.as_mut().unwrap();
        let mut frame_allocator = AllocatorKind::Frame(super_frame_allocator.get_next_frame());

        let mut module = vir::Module::default();
        let attachment = module.acquire_next_image(&self.swapchain);
        let attachment = module.clear(attachment, vir::clear::f32::WHITE);
        let attachment = module.present(attachment);

        let graph = vir::RenderGraph::new(ctx, module.compile(attachment));
        graph.dump();
        graph.submit(&mut frame_allocator);

        panic!();
    }
}

impl ApplicationHandler for App {
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
        self.init(window).expect("Cannot create new app");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let window = self.window.as_mut().unwrap();
        match event {
            WindowEvent::Resized(physical_size) => {
                let raw_window_handle = window.raw_window_handle();
                let raw_display_handle = window.raw_display_handle();
                self.swapchain = self
                    .create_swapchain(
                        vk::Extent2D {
                            width: physical_size.width,
                            height: physical_size.height,
                        },
                        raw_window_handle,
                        raw_display_handle,
                    )
                    .expect("Failed to create swaphcain");
            },
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                window.handle.request_redraw();
                self.run()
            },
            _ => (),
        };
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

        let handle = event_loop.create_window(window_attribs)?;

        Ok(Self { handle })
    }

    fn raw_window_handle(&self) -> RawWindowHandle { self.handle.window_handle().unwrap().as_raw() }

    fn raw_display_handle(&self) -> RawDisplayHandle { self.handle.display_handle().unwrap().as_raw() }
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt() //
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);

    Ok(())
}
