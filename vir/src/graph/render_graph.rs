use std::collections::HashMap;

use ash::vk;

use crate::{Access, AllocatorKind, Context, DomainFlag, IR, SwapChain, Value, ValueId};

pub struct RenderGraph<'a> {
    ctx: &'a Context,
    nodes: Vec<IR>,
    values: HashMap<ValueId, Value>,
}

impl<'a> RenderGraph<'a> {
    pub fn new(ctx: &'a Context, nodes: Vec<IR>) -> Self {
        return Self {
            ctx,
            nodes,
            values: HashMap::new(),
        };
    }

    fn node(&self, id: ValueId) -> &IR { return &self.nodes[id.0 as usize]; }

    pub fn submit(&self, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        for node in &self.nodes {
            self.execute(node, allocator)?;
        }

        Ok(())
    }

    pub fn dump(&self) {
        self.nodes.iter().enumerate().for_each(|(id, node)| {
            println!("%{} = {}", id, node);
        });
    }

    fn execute(&self, ir: &IR, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        match ir {
            IR::Constant(constant) => todo!(),
            IR::Array(value_ids) => todo!(),
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
            } => todo!(),
            IR::AcquireNextImage { swapchain, attachments } => {
                let acquire_semaphore = allocator.allocate_binary_semaphore()?;
                let image_index = self.ctx.acquire_next_image(*swapchain, acquire_semaphore)?;
                allocator.deallocate_semaphore(acquire_semaphore);

                let attachments = self.node(*attachments);
                match &attachments {
                    IR::Array(values) => {},
                    _ => panic!(),
                };
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
        }

        Ok(())
    }
}
