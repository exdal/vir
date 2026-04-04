use std::ptr::NonNull;

use ash::vk::{self, Handle};

use super::Allocator;
use crate::CommandBuffer;

pub struct PersistentAllocator {
    device: NonNull<ash::Device>,
    cmd_pool: vk::CommandPool,
}

impl PersistentAllocator {
    pub fn new(device: NonNull<ash::Device>) -> Self {
        PersistentAllocator {
            device,
            cmd_pool: vk::CommandPool::null(),
        }
    }

    pub fn allocate_command_pool(&self, queue_family: u32) -> Result<vk::CommandPool, vk::Result> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);

        unsafe { self.device.as_ref().create_command_pool(&create_info, None) }
    }

    pub fn reset_command_pool(&self, cmd_pool: vk::CommandPool, release_resources: bool) -> Result<(), vk::Result> {
        let flags = if release_resources {
            vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandPoolResetFlags::empty()
        };

        unsafe { self.device.as_ref().reset_command_pool(cmd_pool, flags) }
    }

    fn ensure_cmd_pool(&mut self, queue_family: u32) -> Result<(), vk::Result> {
        if !self.cmd_pool.is_null() {
            return Ok(());
        }

        self.cmd_pool = self.allocate_command_pool(queue_family)?;

        Ok(())
    }
}

impl Allocator for PersistentAllocator {
    fn allocate_binary_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let create_info = vk::SemaphoreCreateInfo::default();

        unsafe { self.device.as_ref().create_semaphore(&create_info, None) }
    }

    fn allocate_timeline_semaphore(&mut self) -> Result<vk::Semaphore, vk::Result> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        unsafe { self.device.as_ref().create_semaphore(&create_info, None) }
    }

    fn deallocate_semaphore(&self, semaphore: vk::Semaphore) {
        unsafe { self.device.as_ref().destroy_semaphore(semaphore, None) };
    }

    fn allocate_command_buffer(&mut self, queue_family: u32) -> Result<CommandBuffer, vk::Result> {
        self.ensure_cmd_pool(queue_family)?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        let cmd_buffer = unsafe { self.device.as_ref().allocate_command_buffers(&alloc_info) }?[0];

        Ok(CommandBuffer::new(self.device, cmd_buffer))
    }
}

impl std::fmt::Debug for PersistentAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentAllocator").finish_non_exhaustive()
    }
}
