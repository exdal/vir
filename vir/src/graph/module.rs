use std::collections::HashMap;

use ash::vk;

use crate::{Access, DomainFlag, IR, ImageAttachment, SwapChain, ValueId, graph::ir};

pub struct Module {
    constants: HashMap<ir::Constant, ValueId>,
    nodes: Vec<IR>,
}

impl Module {
    pub fn default() -> Self {
        Self {
            constants: HashMap::new(),
            nodes: Vec::new(),
        }
    }

    pub fn compile(&self, id: ValueId) -> Vec<IR> { self.topo_sort(id) }

    fn get(&self, id: ValueId) -> &IR { return &self.nodes[id.0 as usize]; }

    fn topo_sort(&self, value_id: ValueId) -> Vec<IR> {
        let mut nodes = Vec::new();
        let mut stack = vec![value_id];
        let mut visited = std::collections::HashSet::new();
        let mut processed = std::collections::HashSet::new();

        while let Some(id) = stack.pop() {
            if processed.contains(&id) {
                continue;
            }

            let ir = self.get(id);
            if visited.insert(id) {
                stack.push(id);

                match &ir {
                    IR::Constant(_) => {},
                    IR::Array(value_ids) => {
                        value_ids.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::ConstructBuffer { size, .. } => {
                        stack.push(*size);
                    },
                    IR::ConstructImage {
                        extent,
                        base_level,
                        level_count,
                        base_layer,
                        layer_count,
                        ..
                    } => {
                        stack.push(*layer_count);
                        stack.push(*base_layer);
                        stack.push(*level_count);
                        stack.push(*base_level);
                        stack.push(*extent);
                    },
                    IR::AcquireNextImage { attachments, .. } => {
                        stack.push(*attachments);
                    },
                    IR::Acquire { resource, .. } => {
                        stack.push(*resource);
                    },
                    IR::Release { resource, .. } => {
                        stack.push(*resource);
                    },
                    IR::CallOpaque { args, returns, .. } => {
                        stack.extend(returns);
                        stack.extend(args);
                    },
                    IR::Clear { attachment, .. } => {
                        stack.push(*attachment);
                    },
                }
            } else {
                processed.insert(id);
                nodes.push(ir.clone());
            }
        }

        nodes
    }

    fn emit(&mut self, ir: IR) -> ValueId {
        let id = ValueId(self.nodes.len() as u32);
        self.nodes.push(ir);
        id
    }

    fn lower_constant(&mut self, constant: ir::Constant) -> ValueId {
        if let Some(&id) = self.constants.get(&constant) {
            return id;
        }

        let id = self.emit(IR::Constant(constant));
        self.constants.insert(constant, id);
        id
    }

    fn lower_u32(&mut self, v: u32) -> ValueId { self.lower_constant(ir::Constant::U32(v)) }

    fn lower_i32(&mut self, v: i32) -> ValueId { self.lower_constant(ir::Constant::I32(v)) }

    fn lower_array(&mut self, v: Vec<ValueId>) -> ValueId { self.emit(IR::Array(v)) }

    fn lower_image_attachment(&mut self, attachment: &ImageAttachment) -> ValueId {
        let extent = self.lower_constant(ir::Constant::Extent3D(attachment.extent()));
        let base_level = self.lower_u32(attachment.base_level());
        let level_count = self.lower_u32(attachment.level_count());
        let base_layer = self.lower_u32(attachment.base_layer());
        let layer_count = self.lower_u32(attachment.layer_count());

        self.emit(IR::ConstructImage {
            image: attachment.image().handle,
            image_view: attachment.image_view(),
            extent,
            format: attachment.format(),
            samples: attachment.samples(),
            base_level,
            level_count,
            base_layer,
            layer_count,
        })
    }

    pub fn acquire_next_image(&mut self, swapchain: &SwapChain) -> ValueId {
        let attach_values = swapchain
            .attachments
            .iter()
            .map(|attach| self.lower_image_attachment(attach))
            .collect::<Vec<_>>();
        let attachments = self.lower_array(attach_values);
        self.emit(IR::AcquireNextImage {
            swapchain: swapchain.handle,
            attachments,
        })
    }

    pub fn release(&mut self, value: ValueId, access: Access, dst_domain: DomainFlag) -> ValueId {
        self.emit(IR::Release {
            resource: value,
            access,
            dst_domain,
        })
    }

    pub fn present(&mut self, attachment: ValueId) -> ValueId {
        self.release(attachment, Access::None, DomainFlag::Present)
    }

    pub fn clear(&mut self, attachment: ValueId, color: vk::ClearValue) -> ValueId {
        self.emit(IR::Clear { attachment, color })
    }
}
