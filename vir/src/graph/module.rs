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
    types: HashMap<ir::Type, ValueId>,
    constants: HashMap<ir::Constant, ValueId>,
    instructions: Vec<IR>,
}

impl Module {
    fn get(&self, id: ValueId) -> &IR { &self.instructions[id.0 as usize] }

    fn resolve_access(&self, id: ValueId) -> Access {
        match self.get(id) {
            IR::Constant(ir::Constant::Access(a)) => *a,
            _ => panic!("{id} is not an Access constant"),
        }
    }

    fn emit(&mut self, ir: IR) -> ValueId {
        let id = ValueId(self.instructions.len() as u32);
        self.instructions.push(ir);
        id
    }

    fn lower_type(&mut self, ty: ir::Type) -> ValueId {
        if let Some(&id) = self.types.get(&ty) {
            return id;
        }
        let id = self.emit(IR::Type(ty));
        self.types.insert(ty, id);
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

    fn lower_array(&mut self, ty: ValueId, elements: Vec<ValueId>) -> ValueId { self.emit(IR::Array { ty, elements }) }

    fn lower_image_attachment(&mut self, attachment: &ImageAttachment) -> (ValueId, ValueId) {
        let extent = self.lower_constant(ir::Constant::Extent3D(attachment.extent()));
        let base_level = self.lower_u32(attachment.base_level());
        let level_count = self.lower_u32(attachment.level_count());
        let base_layer = self.lower_u32(attachment.base_layer());
        let layer_count = self.lower_u32(attachment.layer_count());

        let ty_instr = self.lower_type(ir::Type::Image {
            format: attachment.format(),
            samples: attachment.samples(),
        });
        let construct_instr = self.emit(IR::ConstructImage {
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
        });

        (ty_instr, construct_instr)
    }

    pub fn acquire_next_image(&mut self, swapchain: &SwapChain) -> ValueId {
        let (ty_instr, attach_instr): (Vec<_>, Vec<_>) = swapchain
            .attachments
            .iter()
            .map(|attach| self.lower_image_attachment(attach))
            .unzip();

        let ty = ty_instr[0];
        assert!(ty_instr.iter().all(|&i| i == ty));
        let attachments = self.lower_array(ty, attach_instr);

        self.emit(IR::AcquireNextImage {
            swapchain: swapchain.handle,
            attachments,
            present_semaphores: swapchain.semaphores.clone(),
        })
    }

    pub fn release(&mut self, resource: ValueId, access: Access, dst_domain: DomainFlag) -> ValueId {
        let access = self.lower_access(access);
        self.emit(IR::Release {
            resource,
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

    pub fn compile(&self, id: ValueId) -> Vec<ir::Instr> {
        let linearized = self.topo_sort(id);
        let synced = self.sync(linearized);
        self.infer(synced)
    }

    fn topo_sort(&self, value_id: ValueId) -> Vec<ir::Instr> {
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
                    IR::Type(_) => {},
                    IR::Array { ty, elements } => {
                        stack.push(*ty);
                        elements.iter().rev().for_each(|v| stack.push(*v));
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
                    IR::MemoryBarrier { src_access, dst_access } => {
                        stack.push(*dst_access);
                        stack.push(*src_access);
                    },
                    IR::ImageBarrier {
                        src_access,
                        dst_access,
                        value,
                        ..
                    } => {
                        stack.push(*value);
                        stack.push(*dst_access);
                        stack.push(*src_access);
                    },
                }
            } else {
                processed.insert(id);
                nodes.push((id, ir.clone()));
            }
        }

        nodes
    }

    fn sync(&self, nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        let mut result = Vec::with_capacity(nodes.len() * 2);
        let mut states: HashMap<ValueId, ResourceState> = HashMap::new();
        let mut next_id = self.instructions.len() as u32;

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
                        src_access: $src,
                        dst_access: $dst,
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

    fn infer(&self, mut nodes: Vec<ir::Instr>) -> Vec<ir::Instr> { nodes }
}
