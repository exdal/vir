use std::collections::HashMap;

use ash::vk::{self, Handle};

use crate::{
    Access,
    AllocatorKind,
    CommandBuffer,
    Context,
    DomainFlag,
    IR,
    ImageAttachment,
    Value,
    ValueId,
    core::ScopedStack,
    graph::{ir, value::FromValue},
};

struct SemaphoreSubmitInfo {
    semaphore: vk::Semaphore,
    value: u64,
    access: Access,
}

impl SemaphoreSubmitInfo {
    fn default() -> Self {
        Self {
            semaphore: vk::Semaphore::null(),
            value: 0,
            access: Access::None,
        }
    }
}

impl From<&SemaphoreSubmitInfo> for vk::SemaphoreSubmitInfo<'_> {
    fn from(info: &SemaphoreSubmitInfo) -> Self {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(info.semaphore)
            .value(info.value)
            .stage_mask(info.access.into())
    }
}

struct Batch {
    cmd_buf: CommandBuffer,
}

impl Batch {
    fn new(cmd_buf: CommandBuffer) -> Result<Self, vk::Result> {
        cmd_buf.begin()?;
        Ok(Self { cmd_buf })
    }

    fn end(self) -> Result<CommandBuffer, vk::Result> {
        self.cmd_buf.end()?;
        Ok(self.cmd_buf)
    }

    fn cmd_buf(&self) -> &CommandBuffer { &self.cmd_buf }
}

struct Submit {
    domain: DomainFlag,
    wait_semas: Vec<SemaphoreSubmitInfo>,
    cmd_buffers: Vec<CommandBuffer>,
    signal_semas: Vec<SemaphoreSubmitInfo>,
}

impl Default for Submit {
    fn default() -> Self {
        Self {
            domain: DomainFlag::Graphics,
            wait_semas: Vec::new(),
            cmd_buffers: Vec::new(),
            signal_semas: Vec::new(),
        }
    }
}

impl Submit {
    fn seal_batch(&mut self, batch: Batch) -> Result<(), vk::Result> {
        self.cmd_buffers.push(batch.end()?);
        Ok(())
    }
}

struct PresentInfo {
    image_index: u32,
    swapchain: vk::SwapchainKHR,
    semaphore: vk::Semaphore,
}

impl PresentInfo {
    fn new(image_index: u32, swapchain: vk::SwapchainKHR, semaphore: vk::Semaphore) -> Self {
        Self {
            image_index,
            swapchain,
            semaphore,
        }
    }
}

pub struct RenderGraph<'a> {
    ctx: &'a Context,
    nodes: &'a [(ValueId, IR)],
    values: Vec<Value>,
    current_batch: Option<Batch>,
    current_submit: Submit,
    submits: Vec<Submit>,
    presents: Vec<PresentInfo>,
    resource_to_swapchain: HashMap<ValueId, PresentInfo>,
}

impl<'a> RenderGraph<'a> {
    pub fn new(ctx: &'a Context, nodes: &'a [(ValueId, IR)]) -> Self {
        Self {
            ctx,
            nodes,
            values: Vec::new(),
            current_batch: None,
            current_submit: Submit::default(),
            submits: Vec::new(),
            presents: Vec::new(),
            resource_to_swapchain: HashMap::new(),
        }
    }

    fn set_value(&mut self, value_id: &ValueId, value: Value) {
        let index = value_id.0 as usize;
        self.values.resize(index + 1, Value::None);
        self.values[index] = value;
    }

    fn get<T: FromValue>(&self, id: &ValueId) -> T {
        match self.get_value(id) {
            Value::Reference(v) => self.get::<T>(v),
            value => T::from_value(value),
        }
    }

    fn get_value(&self, value_id: &ValueId) -> &Value { self.values.get(value_id.0 as usize).unwrap() }

    fn resolve_id(&self, value_id: &ValueId) -> ValueId {
        match self.get_value(value_id) {
            Value::Reference(inner) => self.resolve_id(inner),
            _ => *value_id,
        }
    }

    fn ensure_batch(&mut self, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        if self.current_batch.is_some() {
            return Ok(());
        }

        let queue = self
            .ctx
            .command_queue_by_domain(self.current_submit.domain)
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        let cmd_buf = allocator.allocate_command_buffer(queue.family_index())?;
        self.current_batch = Some(Batch::new(cmd_buf)?);
        Ok(())
    }

    fn batch(&self) -> Result<&CommandBuffer, vk::Result> {
        self.current_batch
            .as_ref()
            .map(|b| b.cmd_buf())
            .ok_or(vk::Result::ERROR_UNKNOWN)
    }

    fn flush_submit(&mut self, signal_sema: Option<SemaphoreSubmitInfo>) -> Result<(), vk::Result> {
        if let Some(signal) = signal_sema {
            self.current_submit.signal_semas.push(signal);
        }
        if let Some(batch) = self.current_batch.take() {
            self.current_submit.seal_batch(batch)?;
        }
        let submit = std::mem::take(&mut self.current_submit);
        self.submits.push(submit);
        Ok(())
    }

    pub fn submit(&mut self, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        for (value_id, node) in self.nodes {
            self.execute(value_id, node, allocator)?;
        }

        for submit in self.submits.drain(..) {
            let stack = ScopedStack::new();
            let wait_semas = stack.alloc_slice::<vk::SemaphoreSubmitInfo>(submit.wait_semas.len());
            let cmd_buf_infos = stack.alloc_slice::<vk::CommandBufferSubmitInfo>(submit.cmd_buffers.len());
            let signal_semas = stack.alloc_slice::<vk::SemaphoreSubmitInfo>(submit.signal_semas.len());

            for (dst, src) in wait_semas.iter_mut().zip(&submit.wait_semas) {
                *dst = src.into();
            }
            for (dst, src) in cmd_buf_infos.iter_mut().zip(&submit.cmd_buffers) {
                *dst = vk::CommandBufferSubmitInfo::default().command_buffer(src.into());
            }
            for (dst, src) in signal_semas.iter_mut().zip(&submit.signal_semas) {
                *dst = src.into();
            }

            let queue = self
                .ctx
                .command_queue_by_domain(submit.domain)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;

            let submit_info = vk::SubmitInfo2KHR::default()
                .wait_semaphore_infos(wait_semas)
                .command_buffer_infos(cmd_buf_infos)
                .signal_semaphore_infos(signal_semas);
            queue.submit(&[submit_info])?;
        }

        for present in self.presents.drain(..) {
            let queue = self
                .ctx
                .command_queue_by_domain(DomainFlag::Graphics)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(&present.semaphore))
                .swapchains(std::slice::from_ref(&present.swapchain))
                .image_indices(std::slice::from_ref(&present.image_index));
            self.ctx.present(queue, &present_info)?;
        }

        Ok(())
    }

    pub fn dump(&self) {
        self.nodes.iter().for_each(|(id, node)| {
            println!("%{} = {}", id.0, node);
        });
    }

    fn execute(&mut self, value_id: &ValueId, ir: &IR, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        self.ensure_batch(allocator)?;

        match ir {
            IR::Constant(c) => match c {
                ir::Constant::I32(_) => todo!(),
                ir::Constant::U32(v) => self.set_value(value_id, Value::U32(*v)),
                ir::Constant::Extent2D(_) => todo!(),
                ir::Constant::Extent3D(v) => self.set_value(value_id, Value::Extent3D(*v)),
            },
            IR::Array(v) => self.set_value(value_id, Value::Slice(v.clone())),
            IR::ConstructBuffer { .. } => todo!(),
            IR::ConstructImage {
                image,
                image_view,
                extent,
                format,
                samples,
                base_level,
                level_count,
                base_layer,
                layer_count,
                usage,
            } => {
                let extent = self.get::<vk::Extent3D>(extent);
                let subresource_range = vk::ImageSubresourceRange {
                    base_mip_level: self.get::<u32>(base_level),
                    level_count: self.get::<u32>(level_count),
                    base_array_layer: self.get::<u32>(base_layer),
                    layer_count: self.get::<u32>(layer_count),
                    ..Default::default()
                };
                // let image_view = if image_view.is_null() {
                //     allocator.allocate_image_view(image.handle, *format, view_type, subresource_range)?
                // } else {
                //     *image_view
                // };

                let attachment =
                    ImageAttachment::new(image.clone(), *format, extent, *samples, vk::ImageLayout::UNDEFINED)
                        // .with_image_view(image_view)
                        .with_subresource_range(subresource_range)
                        .with_usage(*usage);
                self.set_value(value_id, Value::ImageAttachment(attachment));
            },
            IR::AcquireNextImage {
                swapchain,
                attachments,
                present_semaphores,
            } => {
                let acquire_semaphore = allocator.allocate_binary_semaphore()?;
                let image_index = self.ctx.acquire_next_image(*swapchain, acquire_semaphore)?;
                allocator.deallocate_semaphore(acquire_semaphore);

                let attachments = self.get::<Vec<ValueId>>(attachments);
                let attachment_value_id = attachments[image_index as usize];
                self.resource_to_swapchain.insert(
                    attachment_value_id,
                    PresentInfo::new(image_index, *swapchain, present_semaphores[image_index as usize]),
                );
                self.set_value(value_id, Value::Reference(attachment_value_id));
            },
            IR::Acquire { .. } => todo!(),
            IR::Release {
                resource,
                access,
                dst_domain,
            } => {
                let resolved = self.resolve_id(resource);
                if dst_domain.contains(DomainFlag::Present) {
                    let present = self
                        .resource_to_swapchain
                        .remove(&resolved)
                        .ok_or(vk::Result::ERROR_UNKNOWN)?;

                    self.flush_submit(Some(SemaphoreSubmitInfo {
                        semaphore: present.semaphore,
                        value: 0,
                        access: *access,
                    }))?;

                    self.presents.push(present);
                } else {
                    self.flush_submit(None)?;
                }
            },
            IR::CallOpaque { .. } => todo!(),
            IR::Clear { attachment, color } => {
                self.set_value(value_id, Value::Reference(*attachment));
                let attachment = self.get::<ImageAttachment>(attachment);
                let subresource_range = attachment.subresource_range();
                self.batch()?.clear_color(
                    attachment.image().into(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    color,
                    &[subresource_range],
                );
            },
            IR::MemoryBarrier {
                src_access_flags,
                dst_access_flags,
            } => {
                self.batch()?.memory_barrier(*src_access_flags, *dst_access_flags);
            },
            IR::ImageBarrier {
                src_access_flags,
                dst_access_flags,
                old_layout,
                new_layout,
                value,
            } => {
                let attachment = self.get::<ImageAttachment>(value);
                let subresource_range = attachment.subresource_range();
                self.batch()?.image_barrier(
                    attachment.image().into(),
                    *src_access_flags,
                    *dst_access_flags,
                    *old_layout,
                    *new_layout,
                    subresource_range,
                );
            },
        }

        Ok(())
    }
}
