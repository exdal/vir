use std::rc::Rc;

use ash::vk;

use crate::Access;

#[derive(Clone)]
pub struct CommandBuffer {
    device: Rc<ash::Device>,
    handle: vk::CommandBuffer,
}

impl CommandBuffer {
    pub fn new(device: Rc<ash::Device>, handle: vk::CommandBuffer) -> Self { Self { device, handle } }

    pub fn begin(&self) -> Result<(), vk::Result> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.begin_command_buffer(self.handle, &begin_info) }
    }

    pub fn end(&self) -> Result<(), vk::Result> { unsafe { self.device.end_command_buffer(self.handle) } }

    pub fn memory_barrier(&self, src_access: Access, dst_access: Access) {
        let memory_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_access.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_access.into())
            .dst_access_mask(dst_access.into())];
        let dependency_info = vk::DependencyInfo::default().memory_barriers(&memory_barrier);
        unsafe { self.device.cmd_pipeline_barrier2(self.handle, &dependency_info) }
    }

    pub fn clear(
        &self, image: vk::Image, image_layout: vk::ImageLayout, clear_color_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device
                .cmd_clear_color_image(self.handle, image, image_layout, &clear_color_value, ranges);
        }
    }
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBuffer").finish_non_exhaustive()
    }
}
