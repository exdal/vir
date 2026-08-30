use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::IsTerminal,
    ptr::NonNull,
    sync::Arc,
};

use ash::vk::{self, Handle};

use crate::{
    Access,
    AcquiredSwapchain,
    AllocatorKind,
    Buffer,
    BufferImageCopy,
    BufferInfo,
    ClearValue,
    CommandBuffer,
    ComputePipelineInfo,
    Context,
    DescriptorBinding,
    DomainFlag,
    DynamicValues,
    GraphicsPipelineInfo,
    IR,
    ImageAttachment,
    ImageInfo,
    MemoryLocation,
    PassCallback,
    PipelineBindings,
    PipelineId,
    PipelineLayout,
    PipelineState,
    Program,
    PushConstants,
    Rect2D,
    ResolvedViewport,
    Value,
    ValueId,
    VertexLayout,
    Viewport,
    core::ScopedStack,
    graph::{
        ir::{self, scaled},
        value::FromValue,
    },
    resource::{
        self,
        descriptor::DescriptorArena,
        pipeline::{
            ComputePipelineRequest,
            PipelineRequest,
            create_compute_pipelines,
            create_pipelines,
            validate_descriptor_bindings,
        },
        shader::{self, Reflection},
    },
};

struct SemaphoreSubmitInfo {
    semaphore: vk::Semaphore,
    value: u64,
    access: Access,
}

impl From<&SemaphoreSubmitInfo> for vk::SemaphoreSubmitInfo<'_> {
    fn from(info: &SemaphoreSubmitInfo) -> Self {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(info.semaphore)
            .value(info.value)
            .stage_mask(info.access.into())
    }
}

struct Batch {
    cmd_buf: CommandBuffer,
}

impl Batch {
    fn new(cmd_buf: CommandBuffer) -> Result<Self, vk::Result> {
        cmd_buf.begin()?;
        Ok(Self { cmd_buf })
    }

    fn end(self) -> Result<CommandBuffer, vk::Result> {
        self.cmd_buf.end()?;
        Ok(self.cmd_buf)
    }

    fn cmd_buf(&self) -> &CommandBuffer { &self.cmd_buf }
}

struct Submit {
    domain: DomainFlag,
    wait_semas: Vec<SemaphoreSubmitInfo>,
    cmd_buffers: Vec<CommandBuffer>,
    signal_semas: Vec<SemaphoreSubmitInfo>,
}

impl Default for Submit {
    fn default() -> Self {
        Self {
            domain: DomainFlag::Graphics,
            wait_semas: Vec::new(),
            cmd_buffers: Vec::new(),
            signal_semas: Vec::new(),
        }
    }
}

impl Submit {
    fn seal_batch(&mut self, batch: Batch) -> Result<(), vk::Result> {
        self.cmd_buffers.push(batch.end()?);
        Ok(())
    }
}

struct PresentInfo {
    image_index: u32,
    swapchain: vk::SwapchainKHR,
    semaphore: vk::Semaphore,
}

impl PresentInfo {
    fn new(image_index: u32, swapchain: vk::SwapchainKHR, semaphore: vk::Semaphore) -> Self {
        Self {
            image_index,
            swapchain,
            semaphore,
        }
    }
}

fn image_type(view_type: vk::ImageViewType) -> vk::ImageType {
    match view_type {
        vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageType::TYPE_1D,
        vk::ImageViewType::TYPE_3D => vk::ImageType::TYPE_3D,
        _ => vk::ImageType::TYPE_2D,
    }
}

fn subresource_layers(attachment: &ImageAttachment) -> vk::ImageSubresourceLayers {
    let range = attachment.subresource_range();
    vk::ImageSubresourceLayers::default()
        .aspect_mask(range.aspect_mask)
        .mip_level(range.base_mip_level)
        .base_array_layer(range.base_array_layer)
        .layer_count(range.layer_count)
}

fn blit_offsets(extent: vk::Extent3D) -> [vk::Offset3D; 2] {
    [
        vk::Offset3D::default(),
        vk::Offset3D {
            x: extent.width as i32,
            y: extent.height as i32,
            z: extent.depth.max(1) as i32,
        },
    ]
}

enum PipelineKind {
    Graphics {
        info: GraphicsPipelineInfo,
        vertex: VertexLayout,
        variants: HashMap<PipelineState, vk::Pipeline>,
    },
    Compute {
        info: ComputePipelineInfo,
        handle: Option<vk::Pipeline>,
    },
}

struct DeclaredPipeline {
    reflections: Vec<Reflection>,
    bindings: Vec<DescriptorBinding>,
    layout: Option<PipelineLayout>,
    kind: PipelineKind,
}

fn merged_bindings(reflections: &[Reflection]) -> Vec<DescriptorBinding> {
    let mut merged: BTreeMap<(u32, u32), DescriptorBinding> = BTreeMap::new();
    for reflection in reflections {
        for binding in &reflection.bindings {
            merged
                .entry((binding.set, binding.binding))
                .and_modify(|shared| shared.stages |= binding.stages)
                .or_insert(*binding);
        }
    }

    merged.into_values().collect()
}

impl DeclaredPipeline {
    fn layout_handle(&self) -> vk::PipelineLayout {
        self.layout.as_ref().map_or(vk::PipelineLayout::null(), |l| l.handle)
    }

    fn bindless(&self) -> Option<crate::BindlessDescriptorSet> {
        match &self.kind {
            PipelineKind::Graphics { info, .. } => info.bindless,
            PipelineKind::Compute { info, .. } => info.bindless,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DescriptorSetKey {
    pipeline: PipelineId,
    set: u32,
    descriptors: Vec<ResolvedImageDescriptor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResolvedImageDescriptor {
    set: u32,
    binding: u32,
    image_view: vk::ImageView,
    sampler: Option<vk::Sampler>,
}

struct RetiredDescriptorArena {
    arena: DescriptorArena,
    waits: Vec<(vk::Semaphore, u64)>,
}

pub struct Recorder<'a> {
    graph: &'a mut RenderGraph,
    pipeline: PipelineId,
    descriptors: BTreeMap<(u32, u32), ResolvedImageDescriptor>,
    result: Result<(), vk::Result>,
}

impl Recorder<'_> {
    pub fn render_area(&self) -> vk::Rect2D { self.graph.render_area }

    pub fn set_viewport(&mut self, index: u32, viewport: impl Into<Viewport>) -> &mut Self {
        let resolved = viewport.into().resolve(self.graph.render_area);
        self.record(|graph| {
            graph.batch()?.set_viewport(index, &[resolved.into()]);
            graph.recorded_viewports.clear();
            Ok(())
        })
    }

    pub fn set_scissor(&mut self, index: u32, rect: Rect2D) -> &mut Self {
        let resolved = rect.resolve(self.graph.render_area);
        self.record(|graph| {
            graph.batch()?.set_scissor(index, &[resolved]);
            graph.recorded_scissors.clear();
            Ok(())
        })
    }

    pub fn push_constants<T: Copy>(&mut self, value: &T) -> &mut Self { self.push_constants_at(0, value) }

    pub fn push_constants_at<T: Copy>(&mut self, offset: u32, value: &T) -> &mut Self {
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.push_constant_bytes(offset, bytes)
    }

    pub fn push_constant_bytes(&mut self, offset: u32, data: &[u8]) -> &mut Self {
        let push = PushConstants {
            offset,
            data: data.to_vec(),
            source: None,
            addresses: Vec::new(),
        };

        let pipeline = self.pipeline;
        self.record(move |graph| graph.record_push_constants(pipeline, &push))
    }

    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) -> &mut Self {
        self.draw_range(vertex_count, instance_count, 0, 0)
    }

    pub fn draw_range(
        &mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32,
    ) -> &mut Self {
        let pipeline = self.pipeline;
        let descriptors = self.descriptors.values().copied().collect::<Vec<_>>();
        self.record(move |graph| {
            graph.bind_resolved_descriptor_sets(pipeline, vk::PipelineBindPoint::GRAPHICS, &descriptors)?;
            graph
                .batch()?
                .draw(vertex_count, instance_count, first_vertex, first_instance);
            Ok(())
        })
    }

    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32) -> &mut Self {
        self.draw_indexed_range(index_count, instance_count, 0, 0, 0)
    }

    pub fn draw_indexed_range(
        &mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32,
    ) -> &mut Self {
        let pipeline = self.pipeline;
        let descriptors = self.descriptors.values().copied().collect::<Vec<_>>();
        self.record(move |graph| {
            graph.bind_resolved_descriptor_sets(pipeline, vk::PipelineBindPoint::GRAPHICS, &descriptors)?;
            graph
                .batch()?
                .draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance);
            Ok(())
        })
    }

    pub fn buffer(&self, id: ValueId) -> Buffer { self.graph.get::<Buffer>(&id) }

    pub fn image(&self, id: ValueId) -> ImageAttachment { self.graph.get::<ImageAttachment>(&id) }

    pub fn bind_image(&mut self, set: u32, binding: u32, image: &ImageAttachment) -> &mut Self {
        self.descriptors.insert(
            (set, binding),
            ResolvedImageDescriptor {
                set,
                binding,
                image_view: image.image_view(),
                sampler: None,
            },
        );
        self
    }

    pub fn bind_texture(&mut self, set: u32, binding: u32, image: &ImageAttachment, sampler: vk::Sampler) -> &mut Self {
        self.descriptors.insert(
            (set, binding),
            ResolvedImageDescriptor {
                set,
                binding,
                image_view: image.image_view(),
                sampler: Some(sampler),
            },
        );
        self
    }

    fn record(&mut self, action: impl FnOnce(&mut RenderGraph) -> Result<(), vk::Result>) -> &mut Self {
        if self.result.is_ok() {
            self.result = action(self.graph);
        }
        self
    }
}

pub struct RenderGraph {
    device: NonNull<ash::Device>,
    pipelines: Vec<DeclaredPipeline>,
    warned_push_constants: HashSet<PipelineId>,
    descriptor_arena: Option<DescriptorArena>,
    descriptor_cache: HashMap<DescriptorSetKey, vk::DescriptorSet>,
    retired_descriptor_arenas: Vec<RetiredDescriptorArena>,
    values: Vec<Value>,
    bound_pipeline: Option<vk::Pipeline>,
    active_descriptors: Vec<ResolvedImageDescriptor>,
    bound_descriptors: Option<PipelineId>,
    render_area: vk::Rect2D,
    recorded_viewports: Vec<ResolvedViewport>,
    recorded_scissors: Vec<vk::Rect2D>,
    recorded_push_constants: Option<(vk::PipelineLayout, PushConstants)>,
    current_batch: Option<Batch>,
    current_submit: Submit,
    submits: Vec<Submit>,
    presents: Vec<PresentInfo>,
    resource_to_swapchain: HashMap<ValueId, PresentInfo>,
    timeline_signals: Vec<(vk::Semaphore, u64)>,
}

impl PipelineBindings for RenderGraph {
    fn bindings(&self, pipeline: PipelineId) -> Option<&[DescriptorBinding]> {
        self.pipelines
            .get(pipeline.0 as usize)
            .map(|declared| declared.bindings.as_slice())
    }

    fn bindless_set(&self, pipeline: PipelineId) -> Option<u32> {
        self.pipelines
            .get(pipeline.0 as usize)
            .and_then(|declared| declared.bindless())
            .map(|external| external.index)
    }
}

impl RenderGraph {
    pub fn new(ctx: &Context) -> Self {
        Self {
            device: NonNull::from(ctx.device()),
            pipelines: Vec::new(),
            warned_push_constants: HashSet::new(),
            descriptor_arena: None,
            descriptor_cache: HashMap::new(),
            retired_descriptor_arenas: Vec::new(),
            values: Vec::new(),
            bound_pipeline: None,
            active_descriptors: Vec::new(),
            bound_descriptors: None,
            render_area: vk::Rect2D::default(),
            recorded_viewports: Vec::new(),
            recorded_scissors: Vec::new(),
            recorded_push_constants: None,
            current_batch: None,
            current_submit: Submit::default(),
            submits: Vec::new(),
            presents: Vec::new(),
            resource_to_swapchain: HashMap::new(),
            timeline_signals: Vec::new(),
        }
    }

    pub fn declare_pipeline(&mut self, info: GraphicsPipelineInfo) -> Result<PipelineId, vk::Result> {
        if info.shaders.is_empty() {
            tracing::error!("pipeline declared without any shaders");
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }

        let reflections = info
            .shaders
            .iter()
            .map(|spirv| shader::reflect(spirv))
            .collect::<Result<Vec<_>, _>>()?;
        validate_descriptor_bindings(&reflections, info.bindless)?;

        let vertex = VertexLayout::interleaved(&reflections);

        let id = PipelineId(self.pipelines.len() as u32);
        assert!(id.is_valid(), "exhausted valid PipelineId values");
        self.pipelines.push(DeclaredPipeline {
            bindings: merged_bindings(&reflections),
            reflections,
            layout: None,
            kind: PipelineKind::Graphics {
                info,
                vertex,
                variants: HashMap::new(),
            },
        });

        Ok(id)
    }

    pub fn declare_compute_pipeline(&mut self, info: ComputePipelineInfo) -> Result<PipelineId, vk::Result> {
        if info.shader.is_empty() {
            tracing::error!("compute pipeline declared without a shader");
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }

        let reflection = shader::reflect(&info.shader)?;
        if reflection.stage != vk::ShaderStageFlags::COMPUTE {
            tracing::error!(stage = ?reflection.stage, "compute pipeline declared with a shader that is not compute");
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        validate_descriptor_bindings(std::slice::from_ref(&reflection), info.bindless)?;

        let id = PipelineId(self.pipelines.len() as u32);
        assert!(id.is_valid(), "exhausted valid PipelineId values");
        self.pipelines.push(DeclaredPipeline {
            bindings: merged_bindings(std::slice::from_ref(&reflection)),
            reflections: vec![reflection],
            layout: None,
            kind: PipelineKind::Compute { info, handle: None },
        });

        Ok(id)
    }

    fn construct_pipelines(&mut self, instructions: &[ir::Instr]) -> Result<(), vk::Result> {
        let mut graphics: Vec<(PipelineId, PipelineState)> = Vec::new();
        let mut compute: Vec<PipelineId> = Vec::new();

        for (value_id, ir) in instructions {
            let pipeline = match ir {
                IR::Draw { pipeline, .. }
                | IR::DrawIndexed { pipeline, .. }
                | IR::CallOpaque { pipeline, .. }
                | IR::Dispatch { pipeline, .. } => pipeline,
                _ => continue,
            };

            if pipeline.is_invalid() {
                tracing::error!(%value_id, "draw or dispatch with no pipeline bound");
                return Err(vk::Result::ERROR_UNKNOWN);
            }

            let Some(declared) = self.pipelines.get(pipeline.0 as usize) else {
                tracing::error!(%pipeline, "draw or dispatch with a pipeline that was never declared");
                return Err(vk::Result::ERROR_UNKNOWN);
            };

            match (ir, &declared.kind) {
                (
                    IR::Draw { state, .. } | IR::DrawIndexed { state, .. } | IR::CallOpaque { state, .. },
                    PipelineKind::Graphics { variants, .. },
                ) => {
                    if variants.contains_key(state) {
                        continue;
                    }
                    if graphics.iter().any(|(id, pending)| id == pipeline && pending == state) {
                        continue;
                    }

                    graphics.push((*pipeline, state.clone()));
                },
                (IR::Dispatch { .. }, PipelineKind::Compute { handle, .. }) => {
                    if handle.is_some() || compute.contains(pipeline) {
                        continue;
                    }

                    compute.push(*pipeline);
                },
                (IR::Dispatch { .. }, _) => {
                    tracing::error!(%pipeline, "dispatch with a graphics pipeline bound");
                    return Err(vk::Result::ERROR_UNKNOWN);
                },
                _ => {
                    tracing::error!(%pipeline, "draw with a compute pipeline bound");
                    return Err(vk::Result::ERROR_UNKNOWN);
                },
            }
        }

        if graphics.is_empty() && compute.is_empty() {
            return Ok(());
        }

        let device_ptr = self.device;
        let device = unsafe { device_ptr.as_ref() };

        for id in graphics.iter().map(|(id, _)| *id).chain(compute.iter().copied()) {
            let index = id.0 as usize;
            if self.pipelines[index].layout.is_some() {
                continue;
            }

            let layout = PipelineLayout::create(
                device,
                &self.pipelines[index].reflections,
                self.pipelines[index].bindless(),
            )?;
            self.pipelines[index].layout = Some(layout);
        }

        let mut requests = Vec::with_capacity(graphics.len());
        for (id, state) in &graphics {
            let declared = &self.pipelines[id.0 as usize];
            let PipelineKind::Graphics { info, vertex, .. } = &declared.kind else {
                continue;
            };

            requests.push(PipelineRequest {
                info,
                reflections: &declared.reflections,
                vertex,
                state,
                layout: declared.layout_handle(),
            });
        }

        tracing::debug!(count = requests.len(), "compiling pipeline permutations");
        let handles = create_pipelines(device, &requests)?;
        drop(requests);

        for ((id, state), handle) in graphics.into_iter().zip(handles) {
            if let PipelineKind::Graphics { variants, .. } = &mut self.pipelines[id.0 as usize].kind {
                variants.insert(state, handle);
            }
        }

        let mut requests = Vec::with_capacity(compute.len());
        for id in &compute {
            let declared = &self.pipelines[id.0 as usize];
            let (PipelineKind::Compute { info, .. }, Some(reflection)) = (&declared.kind, declared.reflections.first())
            else {
                continue;
            };

            requests.push(ComputePipelineRequest {
                info,
                reflection,
                layout: declared.layout_handle(),
            });
        }

        tracing::debug!(count = requests.len(), "compiling compute pipelines");
        let handles = create_compute_pipelines(device, &requests)?;
        drop(requests);

        for (id, compiled) in compute.into_iter().zip(handles) {
            if let PipelineKind::Compute { handle, .. } = &mut self.pipelines[id.0 as usize].kind {
                *handle = Some(compiled);
            }
        }

        Ok(())
    }

    fn reset(&mut self, program: &Program) -> Result<(), vk::Result> {
        let instructions = program.instructions();
        let value_count = instructions.iter().map(|(id, _)| id.0 as usize + 1).max().unwrap_or(0);

        self.values.clear();
        self.values.resize(value_count, Value::None);

        for (value_id, ir) in instructions {
            let IR::Variable { slot, .. } = ir else {
                continue;
            };
            if let Some(variable) = program.variable(*slot) {
                self.set_value(value_id, variable.value.clone());
            }
        }

        for (resource, variable) in program.bound_variables() {
            if matches!(variable.value, Value::None) {
                tracing::error!(%resource, name = ?variable.name, "nothing was bound to this slot");
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }

            self.set_value(&resource, variable.value.clone());
        }

        self.bound_pipeline = None;
        self.active_descriptors.clear();
        self.bound_descriptors = None;
        self.render_area = vk::Rect2D::default();
        self.recorded_viewports.clear();
        self.recorded_scissors.clear();
        self.recorded_push_constants = None;
        self.current_batch = None;
        self.current_submit = Submit::default();
        self.submits.clear();
        self.presents.clear();
        self.resource_to_swapchain.clear();
        self.timeline_signals.clear();

        Ok(())
    }

    fn set_value(&mut self, value_id: &ValueId, value: Value) {
        let index = value_id.0 as usize;
        if index >= self.values.len() {
            self.values.resize(index + 1, Value::None);
        }
        self.values[index] = value;
    }

    fn get<T: FromValue>(&self, id: &ValueId) -> T {
        match self.get_value(id) {
            Value::Reference(v) => self.get::<T>(v),
            value => T::from_value(value),
        }
    }

    fn get_value(&self, value_id: &ValueId) -> &Value { self.values.get(value_id.0 as usize).unwrap() }

    fn resolve_id(&self, value_id: &ValueId) -> ValueId {
        match self.get_value(value_id) {
            Value::Reference(inner) => self.resolve_id(inner),
            _ => *value_id,
        }
    }

    fn device(&self) -> &ash::Device { unsafe { self.device.as_ref() } }

    fn ensure_batch(&mut self, ctx: &Context, allocator: &mut AllocatorKind) -> Result<(), vk::Result> {
        if self.current_batch.is_some() {
            return Ok(());
        }

        let queue = ctx
            .command_queue_by_domain(self.current_submit.domain)
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        let cmd_buf = allocator.allocate_command_buffer(queue.family_index())?;
        self.current_batch = Some(Batch::new(cmd_buf)?);
        self.bound_pipeline = None;
        self.bound_descriptors = None;
        self.recorded_viewports.clear();
        self.recorded_scissors.clear();
        self.recorded_push_constants = None;
        Ok(())
    }

    fn batch(&self) -> Result<&CommandBuffer, vk::Result> {
        self.current_batch
            .as_ref()
            .map(|b| b.cmd_buf())
            .ok_or(vk::Result::ERROR_UNKNOWN)
    }

    fn flush_submit(&mut self, signal_sema: Option<SemaphoreSubmitInfo>) -> Result<(), vk::Result> {
        if let Some(signal) = signal_sema {
            self.current_submit.signal_semas.push(signal);
        }
        if let Some(batch) = self.current_batch.take() {
            self.current_submit.seal_batch(batch)?;
        }
        let submit = std::mem::take(&mut self.current_submit);
        self.submits.push(submit);
        Ok(())
    }

    fn prepare_descriptor_arena(&mut self, instructions: &[ir::Instr]) -> Result<(), vk::Result> {
        if self.descriptor_arena.is_some() {
            tracing::error!("a previous descriptor execution was not retired");
            return Err(vk::Result::ERROR_UNKNOWN);
        }

        let mut max_sets = 0;
        let mut totals: BTreeMap<i32, (vk::DescriptorType, u32)> = BTreeMap::new();
        for (_, instruction) in instructions {
            let pipeline = match instruction {
                IR::Draw { pipeline, .. }
                | IR::DrawIndexed { pipeline, .. }
                | IR::CallOpaque { pipeline, .. }
                | IR::Dispatch { pipeline, .. } => *pipeline,
                _ => PipelineId::INVALID,
            };
            let Some(layout) = self
                .pipelines
                .get(pipeline.0 as usize)
                .and_then(|pipeline| pipeline.layout.as_ref())
            else {
                continue;
            };

            for set in &layout.sets {
                if set.sizes.is_empty() {
                    continue;
                }
                max_sets += 1;
                for (descriptor_type, count) in &set.sizes {
                    let total = totals.entry(descriptor_type.as_raw()).or_insert((*descriptor_type, 0));
                    total.1 += count;
                }
            }
        }

        self.descriptor_cache.clear();
        self.descriptor_arena = DescriptorArena::create(self.device(), max_sets, &totals)?;
        Ok(())
    }

    fn retire_descriptor_arena(&mut self, waits: Vec<(vk::Semaphore, u64)>) {
        self.descriptor_cache.clear();
        let Some(arena) = self.descriptor_arena.take() else {
            return;
        };
        if waits.is_empty() {
            arena.destroy(self.device());
        } else {
            self.retired_descriptor_arenas
                .push(RetiredDescriptorArena { arena, waits });
        }
    }

    fn collect_descriptor_arenas(&mut self) -> Result<(), vk::Result> {
        let mut pending = Vec::new();
        let mut retired = std::mem::take(&mut self.retired_descriptor_arenas).into_iter();
        while let Some(item) = retired.next() {
            let complete = item.waits.iter().try_fold(true, |complete, (semaphore, value)| {
                let reached = unsafe { self.device().get_semaphore_counter_value(*semaphore) }?;
                Ok::<_, vk::Result>(complete && reached >= *value)
            });
            match complete {
                Ok(true) => item.arena.destroy(self.device()),
                Ok(false) => pending.push(item),
                Err(err) => {
                    pending.push(item);
                    pending.extend(retired);
                    self.retired_descriptor_arenas = pending;
                    return Err(err);
                },
            }
        }
        self.retired_descriptor_arenas = pending;
        Ok(())
    }

    pub fn execute(
        &mut self, ctx: &Context, program: &Program, allocator: &mut AllocatorKind,
    ) -> Result<(), vk::Result> {
        let instructions = program.instructions();
        self.collect_descriptor_arenas()?;
        self.construct_pipelines(instructions)?;
        self.reset(program)?;
        self.prepare_descriptor_arena(instructions)?;

        let result = (|| {
            let mut pc = 0;
            let mut block = None;
            let mut arrived_from = None;
            while let Some((value_id, node)) = instructions.get(pc) {
                match node {
                    IR::Label { label, .. } => block = Some(*label),
                    IR::SelectionMerge { .. } => {},
                    IR::Branch { target, .. } => {
                        arrived_from = block;
                        pc = program.label_index(*target).ok_or(vk::Result::ERROR_UNKNOWN)?;
                        continue;
                    },
                    IR::BranchConditional {
                        condition,
                        true_label,
                        false_label,
                        ..
                    } => {
                        let taken = match self.get::<bool>(condition) {
                            true => *true_label,
                            false => *false_label,
                        };
                        arrived_from = block;
                        pc = program.label_index(taken).ok_or(vk::Result::ERROR_UNKNOWN)?;
                        continue;
                    },
                    IR::Return => break,
                    // a bound resource was placed by reset, there is nothing here to allocate
                    IR::ConstructBuffer { .. } | IR::ConstructImage { .. } if program.bound(value_id).is_some() => {},
                    IR::Phi { incoming, .. } => {
                        let chosen = incoming
                            .iter()
                            .find(|(_, from)| Some(*from) == arrived_from)
                            .map(|(value, _)| *value)
                            .ok_or(vk::Result::ERROR_UNKNOWN)?;
                        self.set_value(value_id, Value::Reference(chosen));
                    },
                    node => self.execute_one(ctx, value_id, node, allocator)?,
                }
                pc += 1;
            }

            if self.current_batch.is_some() || !self.current_submit.cmd_buffers.is_empty() {
                self.flush_submit(None)?;
            }

            for submit in self.submits.drain(..) {
                let queue = ctx
                    .command_queue_by_domain(submit.domain)
                    .ok_or(vk::Result::ERROR_UNKNOWN)?;

                let timeline_value = queue.next_timeline_value();
                let timeline_signal = SemaphoreSubmitInfo {
                    semaphore: *queue.semaphore(),
                    value: timeline_value,
                    access: Access::MemoryRW,
                };

                let stack = ScopedStack::new();
                let wait_semas = stack.alloc_slice::<vk::SemaphoreSubmitInfo>(submit.wait_semas.len());
                let cmd_buf_infos = stack.alloc_slice::<vk::CommandBufferSubmitInfo>(submit.cmd_buffers.len());
                let signal_semas = stack.alloc_slice::<vk::SemaphoreSubmitInfo>(submit.signal_semas.len() + 1);

                for (dst, src) in wait_semas.iter_mut().zip(&submit.wait_semas) {
                    *dst = src.into();
                }
                for (dst, src) in cmd_buf_infos.iter_mut().zip(&submit.cmd_buffers) {
                    *dst = vk::CommandBufferSubmitInfo::default().command_buffer(src.into());
                }
                for (dst, src) in signal_semas
                    .iter_mut()
                    .zip(submit.signal_semas.iter().chain(std::iter::once(&timeline_signal)))
                {
                    *dst = src.into();
                }

                let submit_info = vk::SubmitInfo2::default()
                    .wait_semaphore_infos(wait_semas)
                    .command_buffer_infos(cmd_buf_infos)
                    .signal_semaphore_infos(signal_semas);
                queue.submit(&[submit_info])?;

                allocator.add_timeline_wait(*queue.semaphore(), timeline_value);
                self.timeline_signals.push((*queue.semaphore(), timeline_value));
            }

            for present in self.presents.drain(..) {
                let queue = ctx
                    .command_queue_by_domain(DomainFlag::Graphics)
                    .ok_or(vk::Result::ERROR_UNKNOWN)?;

                let present_info = vk::PresentInfoKHR::default()
                    .wait_semaphores(std::slice::from_ref(&present.semaphore))
                    .swapchains(std::slice::from_ref(&present.swapchain))
                    .image_indices(std::slice::from_ref(&present.image_index));
                ctx.present(queue, &present_info)?;
            }

            Ok(())
        })();

        let waits = self.timeline_signals.clone();
        self.retire_descriptor_arena(waits);
        result
    }

    pub fn wait(&self) -> Result<(), vk::Result> {
        if self.timeline_signals.is_empty() {
            return Ok(());
        }

        let (semaphores, values): (Vec<_>, Vec<_>) = self.timeline_signals.iter().copied().unzip();
        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);

        unsafe { self.device().wait_semaphores(&wait_info, u64::MAX) }
    }

    pub fn execute_blocking(
        &mut self, ctx: &Context, program: &Program, allocator: &mut AllocatorKind,
    ) -> Result<(), vk::Result> {
        self.execute(ctx, program, allocator)?;
        self.wait()?;
        self.collect_descriptor_arenas()?;
        allocator.free_command_buffers();

        Ok(())
    }

    fn record_push_constants(&mut self, pipeline: PipelineId, push: &PushConstants) -> Result<(), vk::Result> {
        if push.is_empty() {
            return Ok(());
        }

        let mut resolved;
        let push = match push.source {
            None => push,
            Some(source) => {
                let bytes = self.get::<Arc<[u8]>>(&source);
                if bytes.len() != push.data.len() {
                    tracing::error!(
                        %pipeline,
                        expected = push.data.len(),
                        got = bytes.len(),
                        "push constant variable does not hold the number of bytes it was declared with"
                    );
                    return Err(vk::Result::ERROR_UNKNOWN);
                }

                resolved = push.clone();
                resolved.data = bytes.to_vec();
                &resolved
            },
        };

        // a transient buffer has no address until it is allocated, so the block only gets one
        // here, over whatever the bytes above left in that range
        let push = match push.addresses.is_empty() {
            true => push,
            false => {
                let mut patched = push.clone();
                for (offset, buffer) in push.addresses.iter() {
                    let address = self.get::<Buffer>(buffer).device_address();
                    let Some(head) = offset.checked_sub(patched.offset).map(|head| head as usize) else {
                        continue;
                    };
                    let Some(slot) = patched.data.get_mut(head..head + size_of::<vk::DeviceAddress>()) else {
                        tracing::error!(
                            %pipeline,
                            offset,
                            "a pushed device address does not fit the block it was declared in"
                        );
                        return Err(vk::Result::ERROR_UNKNOWN);
                    };
                    slot.copy_from_slice(&address.to_ne_bytes());
                }

                resolved = patched;
                &resolved
            },
        };

        let cmd_buf = self.batch()?.clone();
        let layout = self
            .pipelines
            .get(pipeline.0 as usize)
            .and_then(|declared| declared.layout.as_ref())
            .ok_or(vk::Result::ERROR_UNKNOWN)?;

        let unchanged = self
            .recorded_push_constants
            .as_ref()
            .is_some_and(|(handle, recorded)| *handle == layout.handle && recorded == push);
        if unchanged {
            return Ok(());
        }

        let mut covered = 0;
        for (stages, offset, size) in layout.cover(push.offset, push.size()) {
            let head = (offset - push.offset) as usize;
            cmd_buf.push_constants(layout.handle, stages, offset, &push.data[head..head + size as usize]);
            covered += size;
        }

        let handle = layout.handle;
        if covered < push.size() && self.warned_push_constants.insert(pipeline) {
            tracing::warn!(
                %pipeline,
                offset = push.offset,
                size = push.size(),
                covered,
                "pushed bytes the pipeline's shaders do not declare; they are dropped"
            );
        }

        self.recorded_push_constants = Some((handle, push.clone()));
        Ok(())
    }

    /// A pass starts and ends with nothing written, so neither can inherit the other's sets.
    fn open_descriptors(&mut self) {
        self.active_descriptors.clear();
        self.bound_descriptors = None;
    }

    /// Binds the writes still standing, unless the same pipeline already has them bound.
    fn bind_active_descriptors(
        &mut self, pipeline: PipelineId, bind_point: vk::PipelineBindPoint,
    ) -> Result<(), vk::Result> {
        if self.bound_descriptors == Some(pipeline) {
            return Ok(());
        }

        let descriptors = std::mem::take(&mut self.active_descriptors);
        let result = self.bind_resolved_descriptor_sets(pipeline, bind_point, &descriptors);
        self.active_descriptors = descriptors;

        if result.is_ok() {
            self.bound_descriptors = Some(pipeline);
        }
        result
    }

    fn bind_resolved_descriptor_sets(
        &mut self, pipeline: PipelineId, bind_point: vk::PipelineBindPoint, descriptors: &[ResolvedImageDescriptor],
    ) -> Result<(), vk::Result> {
        let Some(declared) = self.pipelines.get(pipeline.0 as usize) else {
            return Err(vk::Result::ERROR_UNKNOWN);
        };
        let Some(layout) = declared.layout.as_ref() else {
            return Err(vk::Result::ERROR_UNKNOWN);
        };

        let layout_handle = layout.handle;
        let set_layouts = layout.sets.iter().map(|set| set.handle).collect::<Vec<_>>();
        let external = layout.bindless;

        // A pass can draw with several pipelines. Its descriptor table is their union, while
        // each pipeline only consumes the locations present in its own reflected layout.
        let mut ordinary = BTreeMap::<u32, Vec<(ResolvedImageDescriptor, vk::DescriptorType)>>::new();
        for expected in &declared.bindings {
            let (set, binding, descriptor_type) = (expected.set, expected.binding, expected.descriptor_type);
            if external.is_some_and(|external| external.index == set) {
                continue;
            }

            let found = descriptors
                .binary_search_by_key(&(set, binding), |provided| (provided.set, provided.binding))
                .ok()
                .map(|at| descriptors[at]);
            let Some(descriptor) = found else {
                tracing::error!(%pipeline, set, binding, "pipeline descriptor has no image bound for this pass");
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            };
            if descriptor.image_view.is_null() {
                tracing::error!(%pipeline, set, binding, "image descriptor uses a null image view");
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            if descriptor.sampler.is_some_and(|sampler| sampler.is_null()) {
                tracing::error!(%pipeline, set, binding, "combined image sampler uses a null sampler");
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            let combined = descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
            if combined != descriptor.sampler.is_some() {
                tracing::error!(
                    %pipeline,
                    set,
                    binding,
                    descriptor_type = ?descriptor_type,
                    "the image binding method does not match the reflected descriptor type"
                );
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            ordinary.entry(set).or_default().push((descriptor, descriptor_type));
        }

        let device_ptr = self.device;
        let device = unsafe { device_ptr.as_ref() };
        let mut sets = Vec::new();
        for (set, bindings) in ordinary {
            let key = DescriptorSetKey {
                pipeline,
                set,
                descriptors: bindings.iter().map(|(descriptor, _)| *descriptor).collect(),
            };
            let descriptor_set = match self.descriptor_cache.get(&key).copied() {
                Some(descriptor_set) => descriptor_set,
                None => {
                    let set_layout = set_layouts
                        .get(set as usize)
                        .copied()
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let arena = self
                        .descriptor_arena
                        .as_mut()
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let descriptor_set = arena.allocate(device, set_layout)?;

                    let infos = bindings
                        .iter()
                        .map(|(descriptor, _)| {
                            vk::DescriptorImageInfo::default()
                                .sampler(descriptor.sampler.unwrap_or(vk::Sampler::null()))
                                .image_view(descriptor.image_view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        })
                        .collect::<Vec<_>>();
                    let writes = bindings
                        .iter()
                        .zip(&infos)
                        .map(|((descriptor, descriptor_type), info)| {
                            vk::WriteDescriptorSet::default()
                                .dst_set(descriptor_set)
                                .dst_binding(descriptor.binding)
                                .descriptor_type(*descriptor_type)
                                .image_info(std::slice::from_ref(info))
                        })
                        .collect::<Vec<_>>();
                    unsafe { device.update_descriptor_sets(&writes, &[]) };
                    self.descriptor_cache.insert(key, descriptor_set);
                    descriptor_set
                },
            };
            sets.push((set, descriptor_set));
        }

        if let Some(external) = external {
            sets.push((external.index, external.set));
        }
        sets.sort_unstable_by_key(|(set, _)| *set);

        let cmd_buf = self.batch()?.clone();
        for (set, descriptor_set) in sets {
            cmd_buf.bind_descriptor_sets(bind_point, layout_handle, set, &[descriptor_set]);
        }

        Ok(())
    }

    fn prepare_draw(
        &mut self, pipeline: PipelineId, state: &PipelineState, dynamic: &DynamicValues, bind_descriptors: bool,
    ) -> Result<(), vk::Result> {
        if pipeline.is_invalid() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        let handle = self
            .pipelines
            .get(pipeline.0 as usize)
            .and_then(|declared| match &declared.kind {
                PipelineKind::Graphics { variants, .. } => variants.get(state).copied(),
                PipelineKind::Compute { .. } => None,
            })
            .ok_or(vk::Result::ERROR_UNKNOWN)?;

        if self.bound_pipeline != Some(handle) {
            self.batch()?.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, handle);
            self.bound_pipeline = Some(handle);
        }

        if bind_descriptors {
            self.bind_active_descriptors(pipeline, vk::PipelineBindPoint::GRAPHICS)?;
        }

        let viewports = dynamic.viewports(self.render_area);
        if !viewports.is_empty() && viewports != self.recorded_viewports {
            let handles = viewports.iter().copied().map(vk::Viewport::from).collect::<Vec<_>>();
            self.batch()?.set_viewport(0, &handles);
            self.recorded_viewports = viewports;
        }

        let scissors = dynamic.scissors(self.render_area);
        if !scissors.is_empty() && scissors != self.recorded_scissors {
            self.batch()?.set_scissor(0, &scissors);
            self.recorded_scissors = scissors;
        }

        self.record_push_constants(pipeline, &dynamic.push_constants)
    }

    fn local_size(&self, pipeline: PipelineId) -> [u32; 3] {
        self.pipelines
            .get(pipeline.0 as usize)
            .and_then(|declared| declared.reflections.first())
            .map_or([1; 3], |reflection| reflection.local_size)
    }

    fn groups_for(&self, invocations: [u32; 3], pipeline: PipelineId) -> [u32; 3] {
        let local_size = match pipeline.is_valid() {
            true => self.local_size(pipeline),
            false => [1; 3],
        };
        [
            invocations[0].div_ceil(local_size[0]),
            invocations[1].div_ceil(local_size[1]),
            invocations[2].div_ceil(local_size[2]),
        ]
    }

    fn prepare_dispatch(&mut self, pipeline: PipelineId, push_constants: &PushConstants) -> Result<(), vk::Result> {
        if pipeline.is_invalid() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        let handle = self
            .pipelines
            .get(pipeline.0 as usize)
            .and_then(|declared| match &declared.kind {
                PipelineKind::Compute { handle, .. } => *handle,
                PipelineKind::Graphics { .. } => None,
            })
            .ok_or(vk::Result::ERROR_UNKNOWN)?;

        if self.bound_pipeline != Some(handle) {
            self.batch()?.bind_pipeline(vk::PipelineBindPoint::COMPUTE, handle);
            self.bound_pipeline = Some(handle);
        }

        self.bind_active_descriptors(pipeline, vk::PipelineBindPoint::COMPUTE)?;

        self.record_push_constants(pipeline, push_constants)
    }

    /// Prints the program, highlighted when stdout is a terminal that has not opted out with
    /// `NO_COLOR`.
    pub fn dump(program: &Program) {
        let highlight = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        println!("{}", program.dump_with(highlight));
    }

    fn execute_one(
        &mut self, ctx: &Context, value_id: &ValueId, ir: &IR, allocator: &mut AllocatorKind,
    ) -> Result<(), vk::Result> {
        match ir {
            IR::Type(_) => {},
            IR::Constant(c) => match c {
                ir::Constant::I32(v) => self.set_value(value_id, Value::I32(*v)),
                ir::Constant::U32(v) => self.set_value(value_id, Value::U32(*v)),
                ir::Constant::Size(v) => self.set_value(value_id, Value::Size(*v)),
                ir::Constant::Extent2D(v) => self.set_value(value_id, Value::Extent2D(*v)),
                ir::Constant::Extent3D(v) => self.set_value(value_id, Value::Extent3D(*v)),
                ir::Constant::String(v) => self.set_value(value_id, Value::String(v.clone())),
                ir::Constant::Access(v) => self.set_value(value_id, Value::Access(*v)),
                ir::Constant::ClearValue(v) => self.set_value(value_id, Value::ClearValue(*v)),
            },
            IR::Variable { .. } => {},
            IR::Array { elements, .. } => self.set_value(value_id, Value::Slice(elements.clone())),
            IR::Index { array, index } => {
                let elements = self.get::<Vec<ValueId>>(array);
                let index = self.get::<u32>(index) as usize;
                let element = *elements.get(index).ok_or(vk::Result::ERROR_UNKNOWN)?;
                self.set_value(value_id, Value::Reference(element));
            },
            IR::ConstructBuffer {
                buffer,
                size,
                usage,
                location,
                name,
                ..
            } => {
                let buffer = if buffer.is_null() {
                    allocator.allocate_buffer(&BufferInfo {
                        size: match size.is_valid() {
                            true => self.get::<usize>(size) as u64,
                            false => 0,
                        },
                        usage: *usage,
                        location: *location,
                        name: match name.is_valid() {
                            true => self.get::<Arc<str>>(name).to_string(),
                            false => format!("transient buffer {value_id}"),
                        },
                    })?
                } else {
                    *buffer
                };

                self.set_value(value_id, Value::Buffer(buffer));
            },
            IR::ConstructImage {
                image,
                image_view,
                view_type,
                extent,
                format,
                samples,
                base_level,
                level_count,
                base_layer,
                layer_count,
                usage,
                initial_layout,
                name,
            } => {
                let extent = match extent.is_valid() {
                    true => self.get::<vk::Extent3D>(extent),
                    false => vk::Extent3D::default(),
                };
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: resource::aspect_mask(*format),
                    base_mip_level: self.get::<u32>(base_level),
                    level_count: self.get::<u32>(level_count),
                    base_array_layer: self.get::<u32>(base_layer),
                    layer_count: self.get::<u32>(layer_count),
                };

                let image = if image.is_null() {
                    allocator.allocate_image(&ImageInfo {
                        extent,
                        format: *format,
                        usage: *usage,
                        image_type: image_type(*view_type),
                        mip_levels: subresource_range.level_count,
                        array_layers: subresource_range.layer_count,
                        samples: *samples,
                        location: MemoryLocation::GpuOnly,
                        name: match name.is_valid() {
                            true => self.get::<Arc<str>>(name).to_string(),
                            false => format!("transient image {value_id}"),
                        },
                    })?
                } else {
                    *image
                };

                let image_view = if image_view.is_null() {
                    allocator.allocate_image_view(image.handle(), *format, *view_type, subresource_range)?
                } else {
                    *image_view
                };

                let attachment = ImageAttachment::new(image, *format, extent, *samples, *initial_layout)
                    .with_image_view(image_view)
                    .with_subresource_range(subresource_range);
                self.set_value(value_id, Value::ImageAttachment(attachment));
            },
            IR::AcquireNextImage { swapchain } => {
                let handle = self.get::<AcquiredSwapchain>(swapchain).handle;
                let acquire_semaphore = allocator.allocate_binary_semaphore()?;
                let image_index = ctx.acquire_next_image(handle, acquire_semaphore)?;

                self.current_submit.wait_semas.push(SemaphoreSubmitInfo {
                    semaphore: acquire_semaphore,
                    value: 0,
                    access: Access::MemoryRW,
                });

                self.set_value(value_id, Value::U32(image_index));
            },
            IR::SwapchainImage { swapchain, acquire, .. } => {
                let image_index = self.get::<u32>(acquire);
                let binding = self.get::<AcquiredSwapchain>(swapchain);

                let attachment = binding
                    .attachments
                    .get(image_index as usize)
                    .ok_or(vk::Result::ERROR_UNKNOWN)?
                    .clone();
                let present_semaphore = *binding
                    .present_semaphores
                    .get(image_index as usize)
                    .ok_or(vk::Result::ERROR_UNKNOWN)?;

                self.resource_to_swapchain.insert(
                    *value_id,
                    PresentInfo::new(image_index, binding.handle, present_semaphore),
                );
                self.set_value(value_id, Value::ImageAttachment(attachment));
            },
            IR::Acquire { .. } => todo!(),
            IR::Release {
                resource,
                access,
                dst_domain,
            } => {
                let access = self.get::<Access>(access);
                let resolved = self.resolve_id(resource);
                if dst_domain.contains(DomainFlag::Present) {
                    let present = self
                        .resource_to_swapchain
                        .remove(&resolved)
                        .ok_or(vk::Result::ERROR_UNKNOWN)?;

                    self.flush_submit(Some(SemaphoreSubmitInfo {
                        semaphore: present.semaphore,
                        value: 0,
                        access,
                    }))?;

                    self.presents.push(present);
                } else if *dst_domain != self.current_submit.domain {
                    self.flush_submit(None)?;
                }
            },
            IR::Clear { attachment, color } => {
                self.ensure_batch(ctx, allocator)?;
                self.set_value(value_id, Value::Reference(*attachment));
                let attachment = self.get::<ImageAttachment>(attachment);
                let color = self.get::<ClearValue>(color);
                let subresource_range = attachment.subresource_range();
                let image = attachment.image().into();
                let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                let batch = self.batch()?;

                if subresource_range.aspect_mask.contains(vk::ImageAspectFlags::DEPTH) {
                    batch.clear_depth_stencil(image, layout, &color, &[subresource_range]);
                } else {
                    batch.clear_color(image, layout, &color, &[subresource_range]);
                }
            },
            IR::Blit { src, dst, filter } => {
                self.ensure_batch(ctx, allocator)?;
                self.set_value(value_id, Value::Reference(*dst));

                let src = self.get::<ImageAttachment>(src);
                let dst = self.get::<ImageAttachment>(dst);
                let region = vk::ImageBlit::default()
                    .src_subresource(subresource_layers(&src))
                    .src_offsets(blit_offsets(src.extent()))
                    .dst_subresource(subresource_layers(&dst))
                    .dst_offsets(blit_offsets(dst.extent()));

                self.batch()?.blit_image(
                    src.image().into(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.image().into(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                    *filter,
                );
            },
            IR::CopyBufferToImage { buffer, image, region } => {
                self.ensure_batch(ctx, allocator)?;
                self.set_value(value_id, Value::Reference(*image));

                let buffer = self.get::<Buffer>(buffer);
                let attachment = self.get::<ImageAttachment>(image);
                let subresource_range = attachment.subresource_range();
                let region = region.unwrap_or_else(|| BufferImageCopy::whole(attachment.extent()));

                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(region.buffer_offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(subresource_range.aspect_mask)
                            .mip_level(region.mip_level)
                            .base_array_layer(subresource_range.base_array_layer)
                            .layer_count(subresource_range.layer_count),
                    )
                    .image_offset(region.image_offset)
                    .image_extent(region.image_extent);

                self.batch()?.copy_buffer_to_image(
                    buffer.handle(),
                    attachment.image().into(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
            },
            IR::BeginRendering {
                color_attachments,
                depth_attachment,
                render_area,
                ..
            } => {
                self.ensure_batch(ctx, allocator)?;
                self.open_descriptors();

                let attachments = color_attachments
                    .iter()
                    .map(|id| self.get::<ImageAttachment>(id))
                    .collect::<Vec<_>>();
                let depth = depth_attachment
                    .is_valid()
                    .then(|| self.get::<ImageAttachment>(depth_attachment));

                let extent = match render_area.is_valid() {
                    true => self.get::<vk::Extent2D>(render_area),
                    // a depth-only region sizes itself off the depth attachment instead
                    false => attachments
                        .first()
                        .or(depth.as_ref())
                        .map(|attachment| vk::Extent2D {
                            width: attachment.extent().width,
                            height: attachment.extent().height,
                        })
                        .ok_or(vk::Result::ERROR_UNKNOWN)?,
                };

                let render_area = vk::Rect2D::default().extent(extent);
                self.render_area = render_area;

                let attachment_infos = attachments
                    .iter()
                    .map(|attachment| {
                        vk::RenderingAttachmentInfo::default()
                            .image_view(attachment.image_view())
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)
                    })
                    .collect::<Vec<_>>();

                let depth_info = depth.as_ref().map(|attachment| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(attachment.image_view())
                        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                });

                self.batch()?
                    .begin_rendering(render_area, &attachment_infos, depth_info.as_ref());

                match color_attachments
                    .first()
                    .copied()
                    .or_else(|| depth_attachment.is_valid().then_some(*depth_attachment))
                {
                    Some(first) => self.set_value(value_id, Value::Reference(first)),
                    None => self.set_value(value_id, Value::None),
                }
            },
            IR::BindPipeline { pass, .. } | IR::SetState { pass, .. } => {
                self.set_value(value_id, Value::Reference(*pass));
            },
            IR::WriteDescriptor {
                pass,
                set,
                binding,
                descriptor,
                ..
            } => {
                self.set_value(value_id, Value::Reference(*pass));

                let image = self.get::<ImageAttachment>(&descriptor.image());
                let written = ResolvedImageDescriptor {
                    set: *set,
                    binding: *binding,
                    image_view: image.image_view(),
                    sampler: descriptor.sampler(),
                };

                match self
                    .active_descriptors
                    .binary_search_by_key(&(*set, *binding), |bound| (bound.set, bound.binding))
                {
                    Ok(at) if self.active_descriptors[at] == written => return Ok(()),
                    Ok(at) => self.active_descriptors[at] = written,
                    Err(at) => self.active_descriptors.insert(at, written),
                }

                self.bound_descriptors = None;
            },
            IR::BindVertexBuffers {
                pass,
                first_binding,
                buffers,
                offsets,
            } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.ensure_batch(ctx, allocator)?;

                let handles = buffers
                    .iter()
                    .map(|id| self.get::<Buffer>(id).handle())
                    .collect::<Vec<_>>();
                self.batch()?.bind_vertex_buffers(*first_binding, &handles, offsets);
            },
            IR::BindIndexBuffer {
                pass,
                buffer,
                offset,
                index_type,
            } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.ensure_batch(ctx, allocator)?;

                let handle = self.get::<Buffer>(buffer).handle();
                self.batch()?.bind_index_buffer(handle, *offset, *index_type);
            },
            IR::Draw {
                pass,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
                pipeline,
                state,
                dynamic,
            } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.prepare_draw(*pipeline, state, dynamic, true)?;

                let vertex_count = self.get::<u32>(vertex_count);
                let instance_count = self.get::<u32>(instance_count);
                let first_vertex = self.get::<u32>(first_vertex);
                let first_instance = self.get::<u32>(first_instance);
                self.batch()?
                    .draw(vertex_count, instance_count, first_vertex, first_instance);
            },
            IR::DrawIndexed {
                pass,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
                pipeline,
                state,
                dynamic,
            } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.prepare_draw(*pipeline, state, dynamic, true)?;

                let index_count = self.get::<u32>(index_count);
                let instance_count = self.get::<u32>(instance_count);
                let first_index = self.get::<u32>(first_index);
                let vertex_offset = self.get::<i32>(vertex_offset);
                let first_instance = self.get::<u32>(first_instance);
                self.batch()?
                    .draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance);
            },
            IR::CallOpaque {
                pass,
                body,
                pipeline,
                state,
                dynamic,
            } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.prepare_draw(*pipeline, state, dynamic, false)?;

                let body = self.get::<PassCallback>(body);
                let pipeline = *pipeline;
                let descriptors = self
                    .active_descriptors
                    .iter()
                    .map(|descriptor| ((descriptor.set, descriptor.binding), *descriptor))
                    .collect();
                let mut recorder = Recorder {
                    graph: self,
                    pipeline,
                    descriptors,
                    result: Ok(()),
                };
                body.call(&mut recorder);
                let result = recorder.result;

                // the callback bound whatever it drew with, so nothing the pass wrote is still
                // in force
                self.bound_pipeline = None;
                self.bound_descriptors = None;
                self.recorded_viewports.clear();
                self.recorded_scissors.clear();
                self.recorded_push_constants = None;

                result?;
            },
            IR::EndRendering { pass } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.batch()?.end_rendering();
                self.open_descriptors();
            },
            IR::BeginCompute { declared_access, .. } => {
                self.ensure_batch(ctx, allocator)?;
                self.open_descriptors();
                match declared_access.first() {
                    Some((first, _)) => self.set_value(value_id, Value::Reference(*first)),
                    None => self.set_value(value_id, Value::None),
                }
            },
            IR::EndCompute { pass } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.open_descriptors();
            },
            IR::Dispatch {
                pass,
                size,
                pipeline,
                push_constants,
            } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.ensure_batch(ctx, allocator)?;
                self.prepare_dispatch(*pipeline, push_constants)?;

                match size {
                    ir::DispatchSize::Groups { x, y, z } => {
                        let groups = [self.get::<u32>(x), self.get::<u32>(y), self.get::<u32>(z)];
                        self.batch()?.dispatch(groups[0], groups[1], groups[2]);
                    },
                    ir::DispatchSize::Invocations { x, y, z } => {
                        let invocations = [self.get::<u32>(x), self.get::<u32>(y), self.get::<u32>(z)];
                        let groups = self.groups_for(invocations, *pipeline);
                        self.batch()?.dispatch(groups[0], groups[1], groups[2]);
                    },
                    ir::DispatchSize::InvocationsPerPixel { image, scale } => {
                        let extent = self.get::<ImageAttachment>(image).extent();
                        let scale = ir::DispatchSize::scale(*scale);
                        let invocations = [
                            scaled(extent.width as u64, scale[0]),
                            scaled(extent.height as u64, scale[1]),
                            scaled(extent.depth as u64, scale[2]),
                        ];
                        let groups = self.groups_for(invocations, *pipeline);
                        self.batch()?.dispatch(groups[0], groups[1], groups[2]);
                    },
                    ir::DispatchSize::InvocationsPerElement {
                        buffer,
                        element_size,
                        scale,
                    } => {
                        let size = self.get::<Buffer>(buffer).size();
                        let invocations = [scaled(size / element_size.max(&1), f32::from_bits(*scale)), 1, 1];
                        let groups = self.groups_for(invocations, *pipeline);
                        self.batch()?.dispatch(groups[0], groups[1], groups[2]);
                    },
                    ir::DispatchSize::Indirect { buffer, offset } => {
                        let handle = self.get::<Buffer>(buffer).handle();
                        self.batch()?.dispatch_indirect(handle, *offset);
                    },
                }
            },
            IR::Label { .. }
            | IR::SelectionMerge { .. }
            | IR::Branch { .. }
            | IR::BranchConditional { .. }
            | IR::Return
            | IR::Phi { .. } => {},
            IR::MemoryBarrier { src_access, dst_access } => {
                self.ensure_batch(ctx, allocator)?;
                let src_access_flags = self.get::<Access>(src_access);
                let dst_access_flags = self.get::<Access>(dst_access);
                self.batch()?.memory_barrier(src_access_flags, dst_access_flags);
            },
            IR::ImageBarrier {
                src_access,
                dst_access,
                old_layout,
                new_layout,
                value,
            } => {
                self.ensure_batch(ctx, allocator)?;
                let src_access_flags = self.get::<Access>(src_access);
                let dst_access_flags = self.get::<Access>(dst_access);
                let attachment = self.get::<ImageAttachment>(value);
                let subresource_range = attachment.subresource_range();
                self.batch()?.image_barrier(
                    attachment.image().into(),
                    src_access_flags,
                    dst_access_flags,
                    *old_layout,
                    *new_layout,
                    subresource_range,
                );
            },
        }

        Ok(())
    }
}

impl Drop for RenderGraph {
    fn drop(&mut self) {
        let device_ptr = self.device;
        let device = unsafe { device_ptr.as_ref() };

        if let Err(err) = unsafe { device.device_wait_idle() } {
            tracing::error!(?err, "failed to wait for the device before tearing down pipelines");
        }

        if let Some(arena) = self.descriptor_arena.take() {
            arena.destroy(device);
        }
        for retired in std::mem::take(&mut self.retired_descriptor_arenas) {
            retired.arena.destroy(device);
        }

        for declared in &self.pipelines {
            match &declared.kind {
                PipelineKind::Graphics { variants, .. } => {
                    for handle in variants.values() {
                        unsafe { device.destroy_pipeline(*handle, None) };
                    }
                },
                PipelineKind::Compute {
                    handle: Some(handle), ..
                } => {
                    unsafe { device.destroy_pipeline(*handle, None) };
                },
                PipelineKind::Compute { .. } => {},
            }

            if let Some(layout) = &declared.layout {
                layout.destroy(device);
            }
        }
    }
}
