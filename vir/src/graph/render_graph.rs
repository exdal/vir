use ash::vk;

use crate::{
    Access, AllocatorKind, CommandBuffer, Context, DomainFlag, IR, ImageAttachment, Value, ValueId, graph::{ir, value::FromValue}
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

#[derive(Default)]
struct SubmitInfo {
    wait_semas: Vec<SemaphoreSubmitInfo>,
    cmd_buffers: Vec<CommandBuffer>,
    signal_semas: Vec<SemaphoreSubmitInfo>,
}

struct OngoingSubmit {
    info: SubmitInfo,
    cmd_buf: Option<CommandBuffer>,
}

impl OngoingSubmit {
    fn new() -> Self {
        Self {
            info: SubmitInfo::default(),
            cmd_buf: None,
        }
    }
}

pub struct RenderGraph<'a> {
    ctx: &'a Context,
    nodes: &'a [(ValueId, IR)],
    values: Vec<Value>,
    submits: Vec<SubmitInfo>,
    ongoing_submit: OngoingSubmit,
}

impl<'a> RenderGraph<'a> {
    pub fn new(ctx: &'a Context, nodes: &'a [(ValueId, IR)]) -> Self {
        return Self {
            ctx,
            nodes,
            values: Vec::new(),
            submits: Vec::new(),
            ongoing_submit: OngoingSubmit::new(),
        };
    }

    fn set_value(&mut self, value_id: &ValueId, value: Value) {
        let index = value_id.0 as usize;
        self.values.resize(index + 1, Value::None);
        self.values[index] = value;
    }

    fn get<T: FromValue>(&self, id: &ValueId) -> T { T::from_value(self.get_value(id)) }

    fn get_value(&self, value_id: &ValueId) -> &Value { self.values.get(value_id.0 as usize).unwrap() }

    fn current_submit(&mut self) -> &SubmitInfo {
        if self.submits.is_empty() {
            self.submits.push(SubmitInfo::default());
        }

        self.submits.last().unwrap()
    }

    fn cmd_buf(&mut self, domain: DomainFlag) -> &CommandBuffer {
        if self.ongoing_submit.cmd_buf.is_none() {

        }

        todo!()
    }

    pub fn submit(&mut self, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        let nodes = std::mem::take(&mut self.nodes);
        for (value_id, node) in nodes {
            self.execute(value_id, node, allocator)?;
        }
        self.nodes = nodes;

        Ok(())
    }

    pub fn dump(&self) {
        self.nodes.iter().for_each(|(id, node)| {
            println!("%{} = {}", id.0, node);
        });
    }

    fn execute(&mut self, value_id: &ValueId, ir: &IR, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        match ir {
            IR::Constant(c) => match c {
                ir::Constant::I32(_) => todo!(),
                ir::Constant::U32(v) => {
                    self.set_value(value_id, Value::U32(*v));
                },
                ir::Constant::Extent2D(_) => todo!(),
                ir::Constant::Extent3D(v) => {
                    self.set_value(value_id, Value::Extent3D(*v));
                },
            },
            IR::Array(v) => self.set_value(value_id, Value::Slice(v.clone())),
            IR::ConstructBuffer { buffer, size } => todo!(),
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
            } => {
                let extent = self.get::<vk::Extent3D>(extent);
                let attachment =
                    ImageAttachment::new(image.clone(), *format, extent, *samples, vk::ImageLayout::UNDEFINED)
                        .with_image_view(*image_view)
                        .with_subresource(vk::ImageSubresourceRange {
                            base_mip_level: self.get::<u32>(base_level),
                            level_count: self.get::<u32>(level_count),
                            base_array_layer: self.get::<u32>(base_layer),
                            layer_count: self.get::<u32>(layer_count),
                            ..Default::default()
                        });
                self.set_value(value_id, Value::ImageAttachment(attachment));
            },
            IR::AcquireNextImage { swapchain, attachments } => {
                let acquire_semaphore = allocator.allocate_binary_semaphore()?;
                let image_index = self.ctx.acquire_next_image(*swapchain, acquire_semaphore)?;
                allocator.deallocate_semaphore(acquire_semaphore);

                let attachments = self.get::<Vec<ValueId>>(attachments);
                let attachment_value_id = &attachments[image_index as usize];
                let attachment = self.get::<ImageAttachment>(attachment_value_id);
                self.set_value(value_id, Value::ImageAttachment(attachment));
            },
            IR::Acquire { resource, access } => todo!(),
            IR::Release {
                resource,
                access,
                dst_domain,
            } => todo!(),
            IR::CallOpaque {
                args,
                returns,
                callback,
                domain,
            } => todo!(),
            IR::Clear { attachment, color } => todo!(),
            IR::MemoryBarrier {
                src_access_flags,
                dst_access_flags,
            } => todo!(),
            IR::ImageBarrier {
                src_access_flags,
                dst_access_flags,
                old_layout,
                new_layout,
                value,
            } => todo!(),
        }

        Ok(())
    }
}
