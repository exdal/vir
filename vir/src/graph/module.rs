use std::{cell::RefCell, collections::HashMap};

use crate::{Access, DomainFlag, IR, ImageAttachment, SwapChain, ValueId, graph::ir};

thread_local! {
    static MODULE: RefCell<Module> = RefCell::new(Module::default());
}

pub struct Value {
    ir: IR,
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

struct Module {
    constants: HashMap<ir::Constant, ValueId>,
    nodes: Vec<Value>,
}

impl Module {
    fn default() -> Self {
        Self {
            constants: HashMap::new(),
            nodes: Vec::new(),
        }
    }

    fn add_dep(&mut self, src: ValueId, dep: ValueId) { self.nodes[src.0 as usize].deps.push(dep); }

    fn emit(&mut self, ir: IR) -> ValueId {
        let id = ValueId(self.nodes.len() as u32);
        self.nodes.push(Value::new(ir));
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

    fn lower_acquire_swapchain(&mut self, swapchain: &SwapChain) -> ValueId {
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

    fn lower_acquire_next_image(&mut self, swapchain: ValueId) -> ValueId {
        self.emit(IR::AcquireNextImage { swapchain })
    }

    fn lower_release(&mut self, value: ValueId, access: Access, dst_domain: DomainFlag) -> ValueId {
        self.emit(IR::Release {
            resource: value,
            access,
            dst_domain,
        })
    }
}

pub fn acquire_swapchain(swapchain: &SwapChain) -> ValueId {
    MODULE.with_borrow_mut(|x| x.lower_acquire_swapchain(swapchain))
}

pub fn acquire_next_image(swapchain: ValueId) -> ValueId {
    MODULE.with_borrow_mut(|x| x.lower_acquire_next_image(swapchain))
}

pub fn release(value: ValueId, access: Access, dst_domain: DomainFlag) -> ValueId {
    MODULE.with_borrow_mut(|x| x.lower_release(value, access, dst_domain))
}
