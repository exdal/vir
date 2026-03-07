use ash::vk::{self};

use crate::{ImageAttachment, PersistentAllocator, allocator::Allocator};

#[derive(Default)]
pub struct SwapChain {
    pub handle: vk::SwapchainKHR,
    pub surface: vk::SurfaceKHR,
    pub semaphores: Vec<vk::Semaphore>,
    pub attachments: Vec<ImageAttachment>,
}

impl SwapChain {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        allocator: &mut PersistentAllocator, handle: vk::SwapchainKHR, surface: vk::SurfaceKHR,
        attachments: Vec<ImageAttachment>,
    ) -> Result<Self, vk::Result> {
        let semaphores = (0..attachments.len())
            .map(|_| allocator.allocate_binary_semaphore())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            handle,
            surface,
            semaphores,
            attachments,
        })
    }
}
