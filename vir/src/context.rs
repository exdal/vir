use std::{cell::RefCell, ptr::NonNull, rc::Rc, sync::Arc};

use ash::{ext, khr, vk};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator as GpuAllocator, AllocatorCreateDesc},
};

pub mod access;
pub mod command_buffer;
pub mod command_queue;

pub use access::Access;
pub use command_buffer::CommandBuffer;
pub use command_queue::{CommandQueue, DomainFlag};

use crate::{PersistentAllocator, SuperFrameAllocator, allocator::MemoryAllocator};

const MIB: u64 = 1024 * 1024;
const DEFAULT_DEVICE_MEMBLOCK_SIZE: u64 = 256 * MIB;
const HOST_MEMBLOCK_SIZE: u64 = 64 * MIB;

// gpu-allocator sizes blocks of every non-host-visible type with this, so the smallest heap holding one bounds it
fn device_memblock_size(properties: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    let smallest_heap = properties.memory_types[..properties.memory_type_count as usize]
        .iter()
        .filter(|memory_type| {
            !memory_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        })
        .map(|memory_type| properties.memory_heaps[memory_type.heap_index as usize].size)
        .min();

    match smallest_heap {
        Some(size) if size <= 1024 * MIB => size / 8,
        _ => DEFAULT_DEVICE_MEMBLOCK_SIZE,
    }
}

pub struct Context {
    device: Arc<ash::Device>,
    physical_device: vk::PhysicalDevice,
    instance: Arc<ash::Instance>,
    memory: MemoryAllocator,
    command_queues: Vec<CommandQueue>,
    swapchain_loader: khr::swapchain::Device,
    surface_loader: khr::surface::Instance,
    debug_utils: Option<ext::debug_utils::Device>,
    push_descriptor: Option<PushDescriptor>,
}

#[derive(Clone)]
pub(crate) struct PushDescriptor {
    pub(crate) loader: khr::push_descriptor::Device,
    pub(crate) max_descriptors: u32,
}

impl Context {
    pub fn new(
        device: ash::Device, physical_device: vk::PhysicalDevice, instance: ash::Instance, entry: &ash::Entry,
    ) -> Result<Self, vk::Result> {
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let surface_loader = khr::surface::Instance::new(entry, &instance);

        // ash fills missing pointers with panicking stubs, so extensions only load when the user enabled them
        let debug_utils = [c"vkCmdBeginDebugUtilsLabelEXT", c"vkCmdEndDebugUtilsLabelEXT"]
            .iter()
            .all(|name| unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }.is_some())
            .then(|| ext::debug_utils::Device::new(&instance, &device));

        let push_descriptor =
            unsafe { instance.get_device_proc_addr(device.handle(), c"vkCmdPushDescriptorSetKHR".as_ptr()) }
                .is_some()
                .then(|| {
                    let mut limits = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
                    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut limits);
                    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
                    PushDescriptor {
                        loader: khr::push_descriptor::Device::new(&instance, &device),
                        max_descriptors: limits.max_push_descriptors,
                    }
                });

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let device_memblock = device_memblock_size(&memory_properties);

        // one memory allocator per device; every allocator handed out below shares it
        let memory = GpuAllocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::new(device_memblock, HOST_MEMBLOCK_SIZE)
                .with_max_device_memblock_size(device_memblock),
        })
        .map_err(crate::allocator::map_allocation_error)?;

        Ok(Self {
            device: Arc::new(device),
            physical_device,
            instance: Arc::new(instance),
            memory: Rc::new(RefCell::new(memory)),
            command_queues: Vec::new(),
            swapchain_loader,
            surface_loader,
            debug_utils,
            push_descriptor,
        })
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn physical_device(&self) -> &vk::PhysicalDevice {
        &self.physical_device
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn swapchain_loader(&self) -> &khr::swapchain::Device {
        &self.swapchain_loader
    }

    pub fn surface_loader(&self) -> &khr::surface::Instance {
        &self.surface_loader
    }

    pub fn debug_utils(&self) -> Option<&ext::debug_utils::Device> {
        self.debug_utils.as_ref()
    }

    pub(crate) fn push_descriptor(&self) -> Option<&PushDescriptor> {
        self.push_descriptor.as_ref()
    }

    pub fn command_queue_by_domain(&self, domain: DomainFlag) -> Option<&CommandQueue> {
        self.command_queues.iter().find(|queue| queue.domain_flags() == domain)
    }

    pub fn acquire_next_image(&self, swapchain: vk::SwapchainKHR, semaphore: vk::Semaphore) -> Result<u32, vk::Result> {
        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(swapchain, u64::MAX, semaphore, vk::Fence::null())
        }?;

        Ok(image_index)
    }

    pub fn present(&self, queue: &CommandQueue, present_info: &vk::PresentInfoKHR) -> Result<bool, vk::Result> {
        unsafe { self.swapchain_loader.queue_present(queue.inner(), present_info) }
    }

    pub fn create_command_queue(&mut self, queue_family_index: u32, domain_flags: DomainFlag) {
        let queue_handle = unsafe { self.device.get_device_queue(queue_family_index, 0) };

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let timeline_semaphore = unsafe { self.device.create_semaphore(&semaphore_create_info, None) }
            .expect("Failed to create timeline semaphore for queue!");

        let queue = CommandQueue::new(
            queue_handle,
            NonNull::from(self.device.as_ref()),
            queue_family_index,
            domain_flags,
            timeline_semaphore,
        );
        self.command_queues.push(queue);
    }

    fn memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    pub fn create_persistent_allocator(&self) -> PersistentAllocator {
        let limits = unsafe { self.instance.get_physical_device_properties(self.physical_device) }.limits;
        let queue_families = self.command_queues.iter().map(CommandQueue::family_index).collect();
        PersistentAllocator::new(
            NonNull::from(self.device.as_ref()),
            self.memory.clone(),
            limits,
            self.memory_properties(),
            queue_families,
        )
    }

    pub fn create_super_frame_allocator(&self, frame_count: usize) -> SuperFrameAllocator {
        let limits = unsafe { self.instance.get_physical_device_properties(self.physical_device) }.limits;
        let queue_families = self.command_queues.iter().map(CommandQueue::family_index).collect();
        SuperFrameAllocator::new(
            NonNull::from(self.device.as_ref()),
            self.memory.clone(),
            limits,
            self.memory_properties(),
            queue_families,
            frame_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use vk::MemoryPropertyFlags as Flags;

    const GIB: u64 = 1024 * MIB;

    const DL: Flags = Flags::DEVICE_LOCAL;
    const HV_HC: Flags = Flags::from_raw(Flags::HOST_VISIBLE.as_raw() | Flags::HOST_COHERENT.as_raw());
    const DL_HV_HC: Flags = Flags::from_raw(DL.as_raw() | HV_HC.as_raw());

    fn properties(heaps: &[u64], types: &[(Flags, u32)]) -> vk::PhysicalDeviceMemoryProperties {
        let mut properties = vk::PhysicalDeviceMemoryProperties::default();
        properties.memory_heap_count = heaps.len() as u32;
        for (slot, &size) in properties.memory_heaps.iter_mut().zip(heaps) {
            slot.size = size;
        }

        properties.memory_type_count = types.len() as u32;
        for (slot, &(property_flags, heap_index)) in properties.memory_types.iter_mut().zip(types) {
            slot.property_flags = property_flags;
            slot.heap_index = heap_index;
        }

        properties
    }

    #[test]
    fn small_device_heap_gets_small_blocks() {
        let apu = properties(&[256 * MIB, 8 * GIB], &[(DL, 0), (DL_HV_HC, 0), (HV_HC, 1)]);
        assert_eq!(device_memblock_size(&apu), 32 * MIB);
    }

    #[test]
    fn device_local_system_heap_does_not_hide_the_small_heap() {
        let apu = properties(&[256 * MIB, 8 * GIB], &[(DL, 0), (DL, 1), (DL_HV_HC, 1)]);
        assert_eq!(device_memblock_size(&apu), 32 * MIB);
    }

    #[test]
    fn large_device_heap_keeps_default_blocks() {
        let discrete = properties(&[8 * GIB, 16 * GIB, 256 * MIB], &[(DL, 0), (HV_HC, 1), (DL_HV_HC, 2)]);
        assert_eq!(device_memblock_size(&discrete), DEFAULT_DEVICE_MEMBLOCK_SIZE);
    }
}
