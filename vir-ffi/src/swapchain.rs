use ash::{vk, vk::Handle};
use vir::{Image, ImageAttachment, SwapChain};

use crate::{
    allocator::VirPersistentAllocator,
    context::{VirContext, context, out_ptr},
    error::{Failure, guard_result, guard_void},
};

#[repr(C)]
pub struct VirSwapChainInfo {
    pub swapchain: u64,
    pub images: *const u64,
    pub image_count: u32,
    pub format: i32,
    pub width: u32,
    pub height: u32,
}

pub struct VirSwapChain(pub SwapChain);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_swapchain_create(
    a: *mut VirPersistentAllocator, info: *const VirSwapChainInfo, out: *mut *mut VirSwapChain,
) -> vk::Result {
    guard_result(|| {
        let allocator = unsafe { a.as_mut() }.ok_or_else(|| Failure::new("persistent allocator is null"))?;
        let info = unsafe { info.as_ref() }.ok_or_else(|| Failure::new("swapchain info is null"))?;
        let out = out_ptr(out)?;
        if info.swapchain == 0 || info.images.is_null() || info.image_count == 0 {
            return Err(Failure::new("a swapchain needs a handle and at least one image"));
        }

        let images = unsafe { std::slice::from_raw_parts(info.images, info.image_count as usize) };
        let format = vk::Format::from_raw(info.format);
        let extent = vk::Extent3D::default().width(info.width).height(info.height).depth(1);
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1);
        let attachments = images
            .iter()
            .map(|&image| {
                ImageAttachment::new(
                    Image::imported(vk::Image::from_raw(image), format, extent, vk::SampleCountFlags::TYPE_1),
                    format,
                    extent,
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageLayout::UNDEFINED,
                )
                .with_subresource_range(subresource_range)
            })
            .collect();

        let swapchain = SwapChain::new(
            &mut allocator.0,
            vk::SwapchainKHR::from_raw(info.swapchain),
            vk::SurfaceKHR::null(),
            attachments,
        )?;
        *out = Box::into_raw(Box::new(VirSwapChain(swapchain)));

        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vir_swapchain_destroy(c: *mut VirContext, s: *mut VirSwapChain) {
    guard_void(|| {
        let device = context(c)?.ctx.device();
        if s.is_null() {
            return Ok(());
        }

        let swapchain = unsafe { Box::from_raw(s) };
        for attachment in &swapchain.0.attachments {
            unsafe { device.destroy_image_view(attachment.image_view(), None) };
        }

        for &semaphore in &swapchain.0.semaphores {
            unsafe { device.destroy_semaphore(semaphore, None) };
        }

        Ok(())
    });
}
