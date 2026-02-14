use std::{cell::RefCell, collections::HashMap};

use crate::{IR, ImageAttachment, SwapChain, ValueId, graph::ir};

thread_local! {
    static MODULE: RefCell<Module> = RefCell::new(Module::default());
}

struct Module {
    constants: HashMap<ir::Constant, ValueId>,
    nodes: Vec<IR>,
}

impl Module {
    fn default() -> Self {
        Self {
            constants: HashMap::new(),
            nodes: Vec::new(),
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

    fn acquire_swapchain(&mut self, swapchain: &SwapChain) {}
}

pub fn acquire_swapchain(swapchain: &SwapChain) { MODULE.with_borrow_mut(|x| x.acquire_swapchain(swapchain)); }
