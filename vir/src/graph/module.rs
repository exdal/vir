use std::collections::HashMap;

use ash::vk;

use crate::{Access, DomainFlag, IR, ImageAttachment, SwapChain, ValueId, graph::ir};

pub struct Value {
    pub ir: IR,
    deps: Vec<ValueId>,
}

impl Value {
    pub fn new(ir: IR) -> Self {
        Self {
            ir,
            deps: Vec::default(),
        }
    }
}

pub struct Module {
    constants: HashMap<ir::Constant, ValueId>,
    nodes: Vec<Value>,
}

impl Module {
    pub fn default() -> Self {
        Self {
            constants: HashMap::new(),
            nodes: Vec::new(),
        }
    }

    pub fn get(&self, id: ValueId) -> &Value { return &self.nodes[id.0 as usize]; }

    pub fn topo_sort(&self, value_id: ValueId) -> Vec<ValueId> {
        let mut values = Vec::new();
        let mut stack = vec![value_id];
        let mut visited = std::collections::HashSet::new();
        let mut processed = std::collections::HashSet::new();

        while let Some(id) = stack.pop() {
            if processed.contains(&id) {
                continue;
            }

            if visited.insert(id) {
                stack.push(id);

                let value = self.get(id);
                match &value.ir {
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
                    IR::AcquireSwapChain { attachments, .. } => {
                        stack.push(*attachments);
                    },
                    IR::AcquireNextImage { swapchain } => {
                        stack.push(*swapchain);
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

                for &dep in &value.deps {
                    stack.push(dep);
                }
            } else {
                processed.insert(id);
                values.push(id);
            }
        }

        values
    }

    pub fn add_dep(&mut self, src: ValueId, dep: ValueId) { self.nodes[src.0 as usize].deps.push(dep); }

    pub fn emit(&mut self, ir: IR) -> ValueId {
        let id = ValueId(self.nodes.len() as u32);
        self.nodes.push(Value::new(ir));
        id
    }

    pub fn lower_constant(&mut self, constant: ir::Constant) -> ValueId {
        if let Some(&id) = self.constants.get(&constant) {
            return id;
        }

        let id = self.emit(IR::Constant(constant));
        self.constants.insert(constant, id);
        id
    }

    pub fn lower_u32(&mut self, v: u32) -> ValueId { self.lower_constant(ir::Constant::U32(v)) }

    pub fn lower_i32(&mut self, v: i32) -> ValueId { self.lower_constant(ir::Constant::I32(v)) }

    pub fn lower_array(&mut self, v: Vec<ValueId>) -> ValueId { self.emit(IR::Array(v)) }

    pub fn lower_image_attachment(&mut self, attachment: &ImageAttachment) -> ValueId {
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

    pub fn lower_acquire_swapchain(&mut self, swapchain: &SwapChain) -> ValueId {
        let attach_values = swapchain
            .attachments
            .iter()
            .map(|attach| self.lower_image_attachment(attach))
            .collect::<Vec<_>>();
        let attachments = self.lower_array(attach_values);
        self.emit(IR::AcquireSwapChain {
            swapchain: swapchain.handle,
            attachments,
        })
    }

    pub fn lower_acquire_next_image(&mut self, swapchain: ValueId) -> ValueId {
        self.emit(IR::AcquireNextImage { swapchain })
    }

    pub fn lower_release(&mut self, value: ValueId, access: Access, dst_domain: DomainFlag) -> ValueId {
        self.emit(IR::Release {
            resource: value,
            access,
            dst_domain,
        })
    }

    pub fn lower_clear(&mut self, attachment: ValueId, color: vk::ClearValue) -> ValueId {
        self.emit(IR::Clear { attachment, color })
    }
}
