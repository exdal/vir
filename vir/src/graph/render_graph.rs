use ash::vk;

use crate::{Access, AllocatorKind, DomainFlag, SwapChain, ValueId, graph::module::Module};

/// ```
/// let mut graph = RenderGraph::new();
/// let swapchain_attachment = graph.acquire_next_image(swapchain);
/// let swapchain_attachment = graph.clear(swapchain_attachment, vir::White::<f32>);
/// let swapchain_attachment = graph.present(swapchain_attachment);
/// graph.compile(swapchain_attachment);
/// graph.optimize();
/// graph.submit();
/// ```
pub struct RenderGraph {
    module: Module,
    compiled: Vec<ValueId>,
}

impl RenderGraph {
    pub fn new() -> Self {
        return Self {
            module: Module::default(),
            compiled: Vec::new(),
        };
    }

    pub fn acquire_next_image(&mut self, swapchain: &SwapChain) -> ValueId {
        let swapchain_value = self.module.lower_acquire_swapchain(swapchain);
        return self.module.lower_acquire_next_image(swapchain_value);
    }

    pub fn clear(&mut self, attachment: ValueId, color: vk::ClearValue) -> ValueId {
        self.module.lower_clear(attachment, color)
    }

    pub fn present(&mut self, attachment: ValueId) -> ValueId {
        self.module.lower_release(attachment, Access::None, DomainFlag::Present)
    }

    pub fn compile(&mut self, value_id: ValueId) {
        let values = self.module.topo_sort(value_id);
        self.compiled = values;
    }

    pub fn submit(&self, allocator: &mut AllocatorKind) {
        self.compiled.iter().for_each(|&value_id| {
            let value = self.module.get(value_id);
            match &value.ir {
                super::IR::Constant(constant) => todo!(),
                super::IR::Array(value_ids) => todo!(),
                super::IR::ConstructBuffer { buffer, size } => todo!(),
                super::IR::ConstructImage { image, image_view, extent, format, samples, base_level, level_count, base_layer, layer_count } => todo!(),
                super::IR::AcquireSwapChain { swapchain, attachments } => todo!(),
                super::IR::AcquireNextImage { swapchain } => todo!(),
                super::IR::Acquire { resource, access } => todo!(),
                super::IR::Release { resource, access, dst_domain } => todo!(),
                super::IR::CallOpaque { args, returns, callback, domain } => todo!(),
                super::IR::Clear { attachment, color } => todo!(),
            }
        });
    }

    pub fn dump(&self) {
        for &id in &self.compiled {
            let value = self.module.get(id);
            println!("%{} = {}", id.0, value.ir);
        }
    }
}
