use std::sync::Arc;

use ash::vk;

use super::Allocator;

pub struct PersistentAllocator {
    device: Arc<ash::Device>,
}

impl PersistentAllocator {
    pub fn new(device: Arc<ash::Device>) -> Self { PersistentAllocator { device } }
}

impl Allocator for PersistentAllocator {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let create_info = vk::SemaphoreCreateInfo::default();

        unsafe { self.device.create_semaphore(&create_info, None) }
    }

    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        unsafe { self.device.create_semaphore(&create_info, None) }
    }

    fn deallocate_semaphore(&self, semaphore: vk::Semaphore) {
        unsafe { self.device.destroy_semaphore(semaphore, None) };
    }
}
