use std::collections::HashMap;

use ash::vk;

use crate::{Access, DomainFlag, IR, ImageAttachment, SwapChain, ValueId, graph::ir};

#[derive(Clone, Copy)]
struct ResourceState {
    layout: vk::ImageLayout,
    last_access: Access,
}

impl ResourceState {
    fn undefined() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            last_access: Access::None,
        }
    }
}

fn layout_for_domain(domain: DomainFlag) -> vk::ImageLayout {
    if domain.contains(DomainFlag::Present) {
        return vk::ImageLayout::PRESENT_SRC_KHR;
    }
    vk::ImageLayout::GENERAL
}

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

    fn get(&self, id: ValueId) -> &IR { return &self.nodes[id.0 as usize]; }

    pub fn compile(&self, id: ValueId) -> Vec<(ValueId, IR)> {
        let linearized = self.topo_sort(id);
        self.sync(linearized)
    }

    fn topo_sort(&self, value_id: ValueId) -> Vec<(ValueId, IR)> {
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
                    IR::DeclareBuffer { size, .. } => {
                        stack.push(*size);
                    },
                    IR::ConstructBuffer { buffer } => {
                        stack.push(*buffer);
                    },
                    IR::DeclareImage {
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
                    IR::ConstructImage { image } => {
                        stack.push(*image);
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
                    IR::MemoryBarrier { .. } => {},
                    IR::ImageBarrier { value, .. } => {
                        stack.push(*value);
                    },
                }
            } else {
                processed.insert(id);
                nodes.push((id, ir.clone()));
            }
        }

        nodes
    }

    fn sync(&self, nodes: Vec<(ValueId, IR)>) -> Vec<(ValueId, IR)> {
        let mut result: Vec<(ValueId, IR)> = Vec::with_capacity(nodes.len());
        let mut states = HashMap::<ValueId, ResourceState>::new();
        let mut next_id = nodes.iter().map(|(id, _)| id.0).max().unwrap_or(0) + 1;

        let mut new_id = || {
            let id = ValueId(next_id);
            next_id += 1;
            id
        };
        let mut image_barrier =
            |state: ResourceState, access: &Access, new_layout: vk::ImageLayout, resource: &ValueId| {
                (
                    new_id(),
                    IR::ImageBarrier {
                        src_access_flags: state.last_access,
                        dst_access_flags: *access,
                        old_layout: state.layout,
                        new_layout,
                        value: *resource,
                    },
                )
            };

        for (value_id, ir) in nodes {
            match &ir {
                IR::AcquireNextImage { .. } | IR::ConstructImage { .. } => {
                    states.insert(value_id, ResourceState::undefined());
                },

                IR::Acquire { resource, access } => {
                    let state = states.get(resource).copied().unwrap_or(ResourceState::undefined());
                    let new_layout = (*access).into();
                    let new_state = ResourceState {
                        layout: new_layout,
                        last_access: *access,
                    };
                    result.push(image_barrier(state, access, new_layout, resource));
                    states.insert(*resource, new_state);
                    states.insert(value_id, new_state);
                },

                IR::Clear { attachment, .. } => {
                    let state = states.get(attachment).copied().unwrap_or(ResourceState::undefined());
                    let new_state = ResourceState {
                        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        last_access: Access::Clear,
                    };
                    result.push(image_barrier(
                        state,
                        &Access::Clear,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        attachment,
                    ));
                    states.insert(*attachment, new_state);
                    states.insert(value_id, new_state);
                },

                IR::Release {
                    resource,
                    access,
                    dst_domain,
                } => {
                    let state = states.get(resource).copied().unwrap_or(ResourceState::undefined());
                    let new_layout = layout_for_domain(*dst_domain);
                    result.push(image_barrier(state, access, new_layout, resource));
                    states.insert(
                        *resource,
                        ResourceState {
                            layout: new_layout,
                            last_access: *access,
                        },
                    );
                },

                _ => {},
            }

            result.push((value_id, ir));
        }

        result
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

        self.emit(IR::DeclareImage {
            image: attachment.image().clone(),
            image_view: attachment.image_view(),
            extent,
            format: attachment.format(),
            samples: attachment.samples(),
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: attachment.usage(),
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
            present_semaphores: swapchain.semaphores.clone(),
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
