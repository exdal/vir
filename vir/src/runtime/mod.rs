use std::sync::Arc;

use ash::{khr, vk};

use crate::{
    allocator::{frame::SuperFrameAllocator, persistent::PersistentAllocator},
    resource::swapchain::SwapChain,
};

pub struct Runtime {
    device: Arc<ash::Device>,
    physical_device: vk::PhysicalDevice,
    instance: Arc<ash::Instance>,
    swapchain_loader: khr::swapchain::Device,
    surface_loader: khr::surface::Instance,
}

impl Runtime {
    pub fn new(
        device: Arc<ash::Device>, physical_device: vk::PhysicalDevice, instance: Arc<ash::Instance>, entry: &ash::Entry,
    ) -> Self {
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let surface_loader = khr::surface::Instance::new(entry, &instance);
        Self {
            device,
            physical_device,
            instance,
            swapchain_loader,
            surface_loader,
        }
    }

    pub fn device(&self) -> &ash::Device { &self.device }

    pub fn physical_device(&self) -> &vk::PhysicalDevice { &self.physical_device }

    pub fn instance(&self) -> &ash::Instance { &self.instance }

    pub fn swapchain_loader(&self) -> &khr::swapchain::Device { &self.swapchain_loader }

    pub fn surface_loader(&self) -> &khr::surface::Instance { &self.surface_loader }

    pub fn acquire_next_frame(&self, swapchain: &SwapChain, semaphore: vk::Semaphore) -> Result<u32, vk::Result> {
        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(swapchain.handle, u64::MAX, semaphore, vk::Fence::null())
        }?;

        Ok(image_index)
    }

    pub fn create_persistent_allocator(&self) -> PersistentAllocator { PersistentAllocator::new(self.device.clone()) }

    pub fn create_super_frame_allocator(&self, frames_in_flight: usize) -> SuperFrameAllocator {
        SuperFrameAllocator::new(self.device.clone(), frames_in_flight)
    }
}
