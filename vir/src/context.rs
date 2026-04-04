use std::{ptr::NonNull, sync::Arc};

use ash::{khr, vk};

pub mod access;
pub mod command_buffer;
pub mod command_queue;

pub use access::Access;
pub use command_buffer::CommandBuffer;
pub use command_queue::{CommandQueue, DomainFlag};

use crate::{PersistentAllocator, SuperFrameAllocator};

pub struct Context {
    device: Arc<ash::Device>,
    physical_device: vk::PhysicalDevice,
    instance: Arc<ash::Instance>,
    command_queues: Vec<CommandQueue>,
    swapchain_loader: khr::swapchain::Device,
    surface_loader: khr::surface::Instance,
}

impl Context {
    pub fn new(
        device: ash::Device, physical_device: vk::PhysicalDevice, instance: ash::Instance, entry: &ash::Entry,
    ) -> Self {
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let surface_loader = khr::surface::Instance::new(entry, &instance);
        Self {
            device: Arc::new(device),
            physical_device,
            instance: Arc::new(instance),
            command_queues: Vec::new(),
            swapchain_loader,
            surface_loader,
        }
    }

    pub fn device(&self) -> &ash::Device { &self.device }

    pub fn physical_device(&self) -> &vk::PhysicalDevice { &self.physical_device }

    pub fn instance(&self) -> &ash::Instance { &self.instance }

    pub fn swapchain_loader(&self) -> &khr::swapchain::Device { &self.swapchain_loader }

    pub fn surface_loader(&self) -> &khr::surface::Instance { &self.surface_loader }

    pub fn acquire_next_image(&self, swapchain: vk::SwapchainKHR, semaphore: vk::Semaphore) -> Result<u32, vk::Result> {
        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(swapchain, u64::MAX, semaphore, vk::Fence::null())
        }?;

        Ok(image_index)
    }

    pub fn create_command_queue(&mut self, queue_family_index: u32, domain_flags: DomainFlag) {
        let queue_handle = unsafe { self.device.get_device_queue(queue_family_index, 0) };

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let timeline_semaphore = unsafe { self.device.create_semaphore(&semaphore_create_info, None) }
            .expect("Failed to create timeline semaphore for queue!");

        let queue = CommandQueue::new(queue_handle, queue_family_index, domain_flags, timeline_semaphore);
        self.command_queues.push(queue);
    }

    pub fn create_persistent_allocator(&self) -> PersistentAllocator {
        PersistentAllocator::new(NonNull::from(self.device.as_ref()))
    }

    pub fn create_super_frame_allocator(&self) -> SuperFrameAllocator {
        SuperFrameAllocator::new(NonNull::from(self.device.as_ref()), 3)
    }
}
