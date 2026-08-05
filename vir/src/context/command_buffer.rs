use std::ptr::NonNull;

use ash::vk;

use crate::{Access, ClearValue};

#[derive(Clone)]
pub struct CommandBuffer {
    device: NonNull<ash::Device>,
    handle: vk::CommandBuffer,
}

impl From<&CommandBuffer> for vk::CommandBuffer {
    fn from(cmd_buf: &CommandBuffer) -> Self { cmd_buf.handle }
}

impl CommandBuffer {
    pub fn new(device: NonNull<ash::Device>, handle: vk::CommandBuffer) -> Self { Self { device, handle } }

    pub fn reset(&self, release: bool) -> Result<(), vk::Result> {
        let flags = if release {
            vk::CommandBufferResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandBufferResetFlags::empty()
        };

        unsafe { self.device.as_ref().reset_command_buffer(self.handle, flags) }
    }

    pub fn begin(&self) -> Result<(), vk::Result> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.as_ref().begin_command_buffer(self.handle, &begin_info) }
    }

    pub fn end(&self) -> Result<(), vk::Result> { unsafe { self.device.as_ref().end_command_buffer(self.handle) } }

    pub fn memory_barrier(&self, src_access: Access, dst_access: Access) {
        let memory_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_access.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_access.into())
            .dst_access_mask(dst_access.into())];
        let dependency_info = vk::DependencyInfo::default().memory_barriers(&memory_barrier);
        unsafe {
            self.device
                .as_ref()
                .cmd_pipeline_barrier2(self.handle, &dependency_info)
        }
    }

    pub fn image_barrier(
        &self, image: vk::Image, src_access: Access, dst_access: Access, old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout, subresource_range: vk::ImageSubresourceRange,
    ) {
        let image_barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_access.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_access.into())
            .dst_access_mask(dst_access.into())
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(u32::MAX)
            .dst_queue_family_index(u32::MAX)
            .subresource_range(subresource_range)
            .image(image)];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&image_barrier);
        unsafe {
            self.device
                .as_ref()
                .cmd_pipeline_barrier2(self.handle, &dependency_info)
        }
    }

    pub fn begin_rendering(&self, render_area: vk::Rect2D, color_attachments: &[vk::RenderingAttachmentInfo]) {
        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(color_attachments);

        unsafe { self.device.as_ref().cmd_begin_rendering(self.handle, &rendering_info) }
    }

    pub fn set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.device
                .as_ref()
                .cmd_set_viewport(self.handle, first_viewport, viewports)
        }
    }

    pub fn set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device
                .as_ref()
                .cmd_set_scissor(self.handle, first_scissor, scissors)
        }
    }

    pub fn end_rendering(&self) { unsafe { self.device.as_ref().cmd_end_rendering(self.handle) } }

    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .as_ref()
                .cmd_bind_pipeline(self.handle, bind_point, pipeline)
        }
    }

    pub fn push_constants(
        &self, layout: vk::PipelineLayout, stage_flags: vk::ShaderStageFlags, offset: u32, data: &[u8],
    ) {
        unsafe {
            self.device
                .as_ref()
                .cmd_push_constants(self.handle, layout, stage_flags, offset, data)
        }
    }

    pub fn bind_vertex_buffers(&self, first_binding: u32, buffers: &[vk::Buffer], offsets: &[vk::DeviceSize]) {
        unsafe {
            self.device
                .as_ref()
                .cmd_bind_vertex_buffers(self.handle, first_binding, buffers, offsets)
        }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe {
            self.device
                .as_ref()
                .cmd_draw(self.handle, vertex_count, instance_count, first_vertex, first_instance)
        }
    }

    pub fn clear_color(
        &self, image: vk::Image, image_layout: vk::ImageLayout, clear_color_value: &ClearValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device.as_ref().cmd_clear_color_image(
                self.handle,
                image,
                image_layout,
                &clear_color_value.0.color,
                ranges,
            );
        }
    }
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBuffer").finish_non_exhaustive()
    }
}
