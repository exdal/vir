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

        let attachments = attachments
            .into_iter()
            .map(|attachment| {
                let view_type = if attachment.layer_count() > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                };
                let view = allocator.allocate_image_view(
                    attachment.image().into(),
                    attachment.format(),
                    view_type,
                    attachment.subresource_range(),
                )?;
                Ok(attachment.with_image_view(view))
            })
            .collect::<Result<Vec<_>, vk::Result>>()?;

        Ok(Self {
            handle,
            surface,
            semaphores,
            attachments,
        })
    }
}
