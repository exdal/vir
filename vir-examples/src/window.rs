use std::{ffi::CStr, result::Result};

use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{dpi::LogicalSize, error::OsError, event_loop::ActiveEventLoop, window::Window as WinitWindow};

pub struct Window {
    pub handle: WinitWindow,
}

impl Window {
    pub fn new(event_loop: &ActiveEventLoop, title: &str, size: vk::Extent2D) -> Result<Self, OsError> {
        let attributes = WinitWindow::default_attributes()
            .with_title(title)
            .with_inner_size(LogicalSize::new(size.width, size.height));

        Ok(Self {
            handle: event_loop.create_window(attributes)?,
        })
    }

    pub fn raw_window_handle(&self) -> RawWindowHandle { self.handle.window_handle().unwrap().as_raw() }

    pub fn raw_display_handle(&self) -> RawDisplayHandle { self.handle.display_handle().unwrap().as_raw() }

    pub fn extent(&self) -> vk::Extent2D {
        let size = self.handle.inner_size();
        vk::Extent2D {
            width: size.width,
            height: size.height,
        }
    }
}

pub fn surface_extension(handle: Option<RawWindowHandle>) -> Result<&'static CStr, vk::Result> {
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

pub fn create_surface(
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
