use std::collections::HashMap;

use ash::vk;

use crate::{Access, ClearValue, DomainFlag, IR, ImageAttachment, SwapChain, ValueId, graph::ir};

#[derive(Clone, Copy)]
struct ResourceState {
    layout: vk::ImageLayout,
    last_access: ValueId,
}

fn layout_for_domain(domain: DomainFlag) -> vk::ImageLayout {
    if domain.contains(DomainFlag::Present) {
        return vk::ImageLayout::PRESENT_SRC_KHR;
    }
    vk::ImageLayout::GENERAL
}

#[derive(Default)]
pub struct Module {
    constants: HashMap<ir::Constant, ValueId>,
    declares: HashMap<IR, ValueId>,
    nodes: Vec<IR>,
}

impl Module {
    fn get(&self, id: ValueId) -> &IR { &self.nodes[id.0 as usize] }

    fn resolve_access(&self, id: ValueId) -> Access {
        match self.get(id) {
            IR::Constant(ir::Constant::Access(a)) => *a,
            _ => panic!("{id} is not an Access constant"),
        }
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

    fn lower_access(&mut self, access: Access) -> ValueId { self.lower_constant(ir::Constant::Access(access)) }

    fn lower_array(&mut self, v: Vec<ValueId>) -> ValueId { self.emit(IR::Array(v)) }

    fn lower_image_attachment(&mut self, attachment: &ImageAttachment) -> ValueId {
        let extent = self.lower_constant(ir::Constant::Extent3D(attachment.extent()));
        let base_level = self.lower_u32(attachment.base_level());
        let level_count = self.lower_u32(attachment.level_count());
        let base_layer = self.lower_u32(attachment.base_layer());
        let layer_count = self.lower_u32(attachment.layer_count());
        let declare_image = IR::DeclareImage {
            image: attachment.image().clone(),
            image_view: attachment.image_view(),
            view_type: vk::ImageViewType::from_raw(-1),
            extent,
            format: attachment.format(),
            samples: attachment.samples(),
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: vk::ImageUsageFlags::empty(),
        };

        let declare_id = if let Some(&existing) = self.declares.get(&declare_image) {
            existing
        } else {
            let id = self.emit(declare_image.clone());
            self.declares.insert(declare_image, id);
            id
        };

        self.emit(IR::ConstructImage { image: declare_id })
    }

    pub fn acquire_next_image(&mut self, swapchain: &SwapChain) -> ValueId {
        let attach_values = swapchain
            .attachments
            .iter()
            .map(|a| self.lower_image_attachment(a))
            .collect::<Vec<_>>();
        let attachments = self.lower_array(attach_values);
        self.emit(IR::AcquireNextImage {
            swapchain: swapchain.handle,
            attachments,
            present_semaphores: swapchain.semaphores.clone(),
        })
    }

    pub fn release(&mut self, value: ValueId, access: Access, dst_domain: DomainFlag) -> ValueId {
        let access = self.lower_access(access);
        self.emit(IR::Release {
            resource: value,
            access,
            dst_domain,
        })
    }

    pub fn present(&mut self, attachment: ValueId) -> ValueId {
        self.release(attachment, Access::None, DomainFlag::Present)
    }

    pub fn clear(&mut self, attachment: ValueId, color: ClearValue) -> ValueId {
        let color = self.lower_constant(ir::Constant::ClearValue(color));
        self.emit(IR::Clear { attachment, color })
    }

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
                match ir {
                    IR::Constant(_) => {},
                    IR::Array(ids) => ids.iter().rev().for_each(|v| stack.push(*v)),
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
                    IR::Acquire { access, resource, .. } => {
                        stack.push(*access);
                        stack.push(*resource);
                    },
                    IR::Release { access, resource, .. } => {
                        stack.push(*access);
                        stack.push(*resource);
                    },
                    IR::CallOpaque { args, returns, .. } => {
                        stack.push(*returns);
                        stack.push(*args);
                    },
                    IR::Clear { color, attachment } => {
                        stack.push(*color);
                        stack.push(*attachment);
                    },
                    IR::MemoryBarrier {
                        src_access_flags,
                        dst_access_flags,
                    } => {
                        stack.push(*dst_access_flags);
                        stack.push(*src_access_flags);
                    },
                    IR::ImageBarrier {
                        src_access_flags,
                        dst_access_flags,
                        value,
                        ..
                    } => {
                        stack.push(*value);
                        stack.push(*dst_access_flags);
                        stack.push(*src_access_flags);
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
        let mut result = Vec::with_capacity(nodes.len() * 2);
        let mut states: HashMap<ValueId, ResourceState> = HashMap::new();
        let mut next_id = self.nodes.len() as u32;

        macro_rules! alloc {
            () => {{
                let id = ValueId(next_id);
                next_id += 1;
                id
            }};
        }
        macro_rules! emit_access {
            ($access:expr) => {{
                let id = alloc!();
                result.push((id, IR::Constant(ir::Constant::Access($access))));
                id
            }};
        }
        macro_rules! emit_barrier {
            ($src:expr, $dst:expr, $old:expr, $new:expr, $res:expr) => {{
                let id = alloc!();
                result.push((
                    id,
                    IR::ImageBarrier {
                        src_access_flags: $src,
                        dst_access_flags: $dst,
                        old_layout: $old,
                        new_layout: $new,
                        value: $res,
                    },
                ));
            }};
        }

        let no_access_id = emit_access!(Access::None);
        let undefined = ResourceState {
            layout: vk::ImageLayout::UNDEFINED,
            last_access: no_access_id,
        };

        for (value_id, ir) in nodes {
            match &ir {
                IR::AcquireNextImage { .. } | IR::ConstructImage { .. } => {
                    states.insert(value_id, undefined);
                },

                IR::Acquire { resource, access } => {
                    let state = states.get(resource).copied().unwrap_or(undefined);
                    let new_layout = self.resolve_access(*access).into();
                    emit_barrier!(state.last_access, *access, state.layout, new_layout, *resource);
                    let new_state = ResourceState {
                        layout: new_layout,
                        last_access: *access,
                    };
                    states.insert(*resource, new_state);
                    states.insert(value_id, new_state);
                },

                IR::Clear { attachment, .. } => {
                    let state = states.get(attachment).copied().unwrap_or(undefined);
                    let access_id = emit_access!(Access::Clear);
                    emit_barrier!(
                        state.last_access,
                        access_id,
                        state.layout,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        *attachment
                    );
                    let new_state = ResourceState {
                        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        last_access: access_id,
                    };
                    states.insert(*attachment, new_state);
                    states.insert(value_id, new_state);
                },

                IR::Release {
                    resource,
                    access,
                    dst_domain,
                } => {
                    let state = states.get(resource).copied().unwrap_or(undefined);
                    let new_layout = layout_for_domain(*dst_domain);
                    emit_barrier!(state.last_access, *access, state.layout, new_layout, *resource);
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
}
