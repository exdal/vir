use std::{
    collections::{HashMap, HashSet},
    ptr::NonNull,
};

use ash::vk::{self, Handle};

use crate::{
    Access,
    AllocatorKind,
    Buffer,
    ClearValue,
    CommandBuffer,
    ComputePipelineInfo,
    Context,
    DescriptorSets,
    DomainFlag,
    DynamicValues,
    GraphicsPipelineInfo,
    IR,
    ImageAttachment,
    ImageInfo,
    MemoryLocation,
    PipelineId,
    PipelineLayout,
    PipelineState,
    PushConstants,
    ResolvedViewport,
    TextureId,
    Value,
    ValueId,
    VertexLayout,
    core::ScopedStack,
    graph::{dump, ir, value::FromValue},
    resource::{
        self,
        pipeline::{ComputePipelineRequest, PipelineRequest, create_compute_pipelines, create_pipelines},
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
    layout: Option<PipelineLayout>,
    descriptors: Option<DescriptorSets>,
    written_textures: u64,
    kind: PipelineKind,
}

impl DeclaredPipeline {
    fn layout_handle(&self) -> vk::PipelineLayout {
        self.layout.as_ref().map_or(vk::PipelineLayout::null(), |l| l.handle)
    }
}

#[derive(Clone, Copy)]
struct Texture {
    image_view: vk::ImageView,
    sampler: vk::Sampler,
}

pub const DEFAULT_VARIABLE_DESCRIPTOR_COUNT: u32 = 1024;

pub struct RenderGraph {
    device: NonNull<ash::Device>,
    pipelines: Vec<DeclaredPipeline>,
    variable_descriptor_count: u32,
    warned_push_constants: HashSet<PipelineId>,
    textures: Vec<Option<Texture>>,
    free_textures: Vec<u32>,
    texture_revision: u64,
    values: Vec<Value>,
    bound_pipeline: Option<vk::Pipeline>,
    bound_layout: Option<(vk::PipelineBindPoint, vk::PipelineLayout)>,
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

impl RenderGraph {
    pub fn new(ctx: &Context) -> Self {
        Self {
            device: NonNull::from(ctx.device()),
            pipelines: Vec::new(),
            variable_descriptor_count: DEFAULT_VARIABLE_DESCRIPTOR_COUNT,
            warned_push_constants: HashSet::new(),
            textures: Vec::new(),
            free_textures: Vec::new(),
            texture_revision: 0,
            values: Vec::new(),
            bound_pipeline: None,
            bound_layout: None,
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

    pub fn with_variable_descriptor_count(mut self, count: u32) -> Self {
        self.variable_descriptor_count = count;
        self
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

        let vertex = VertexLayout::interleaved(&reflections);

        let id = PipelineId(self.pipelines.len() as u32);
        self.pipelines.push(DeclaredPipeline {
            reflections,
            layout: None,
            descriptors: None,
            written_textures: 0,
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

        let id = PipelineId(self.pipelines.len() as u32);
        self.pipelines.push(DeclaredPipeline {
            reflections: vec![reflection],
            layout: None,
            descriptors: None,
            written_textures: 0,
            kind: PipelineKind::Compute { info, handle: None },
        });

        Ok(id)
    }

    pub fn register_texture(&mut self, image_view: vk::ImageView, sampler: vk::Sampler) -> TextureId {
        let texture = Some(Texture { image_view, sampler });
        self.texture_revision += 1;

        match self.free_textures.pop() {
            Some(index) => {
                self.textures[index as usize] = texture;
                TextureId(index)
            },
            None => {
                self.textures.push(texture);
                TextureId(self.textures.len() as u32 - 1)
            },
        }
    }

    pub fn unregister_texture(&mut self, id: TextureId) {
        let Some(slot) = self.textures.get_mut(id.0 as usize) else {
            tracing::error!(%id, "texture was never registered with this graph");
            return;
        };
        if slot.take().is_none() {
            tracing::error!(%id, "texture slot is already free");
            return;
        }

        self.free_textures.push(id.0);
        self.texture_revision += 1;
    }

    fn update_descriptors(&mut self) {
        let device_ptr = self.device;
        let device = unsafe { device_ptr.as_ref() };
        let revision = self.texture_revision;

        let mut runs: Vec<(u32, Vec<vk::DescriptorImageInfo>)> = Vec::new();
        for (index, texture) in self.textures.iter().enumerate() {
            let Some(texture) = texture else {
                continue;
            };
            let info = vk::DescriptorImageInfo::default()
                .sampler(texture.sampler)
                .image_view(texture.image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            match runs.last_mut() {
                Some((first, infos)) if *first as usize + infos.len() == index => infos.push(info),
                _ => runs.push((index as u32, vec![info])),
            }
        }

        for pipeline in &mut self.pipelines {
            if pipeline.written_textures == revision {
                continue;
            }

            let (Some(layout), Some(descriptors)) = (&pipeline.layout, &pipeline.descriptors) else {
                continue;
            };
            let Some(table) = layout.texture_binding else {
                continue;
            };
            let Some(set) = descriptors.set(table.set) else {
                continue;
            };

            let writes = runs
                .iter()
                .map(|(first, infos)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(table.binding)
                        .dst_array_element(*first)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(infos)
                })
                .collect::<Vec<_>>();

            if !writes.is_empty() {
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
            pipeline.written_textures = revision;
        }
    }

    fn construct_pipelines(&mut self, instructions: &[ir::Instr]) -> Result<(), vk::Result> {
        let mut graphics: Vec<(PipelineId, PipelineState)> = Vec::new();
        let mut compute: Vec<PipelineId> = Vec::new();

        for (value_id, ir) in instructions {
            let pipeline = match ir {
                IR::Draw { pipeline, .. } | IR::DrawIndexed { pipeline, .. } | IR::Dispatch { pipeline, .. } => {
                    pipeline
                },
                _ => continue,
            };

            let Some(pipeline) = pipeline else {
                tracing::error!(%value_id, "draw or dispatch with no pipeline bound");
                return Err(vk::Result::ERROR_UNKNOWN);
            };

            let Some(declared) = self.pipelines.get(pipeline.0 as usize) else {
                tracing::error!(%pipeline, "draw or dispatch with a pipeline that was never declared");
                return Err(vk::Result::ERROR_UNKNOWN);
            };

            match (ir, &declared.kind) {
                (IR::Draw { state, .. } | IR::DrawIndexed { state, .. }, PipelineKind::Graphics { variants, .. }) => {
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
                self.variable_descriptor_count,
            )?;
            self.pipelines[index].descriptors = DescriptorSets::create(device, &layout)?;
            self.pipelines[index].written_textures = 0;
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

    fn reset(&mut self, instructions: &[ir::Instr]) {
        let value_count = instructions.iter().map(|(id, _)| id.0 as usize + 1).max().unwrap_or(0);

        self.values.clear();
        self.values.resize(value_count, Value::None);
        self.bound_pipeline = None;
        self.bound_layout = None;
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
        self.bound_layout = None;
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

    pub fn execute(
        &mut self, ctx: &Context, instructions: &[ir::Instr], allocator: &mut AllocatorKind,
    ) -> Result<(), vk::Result> {
        self.construct_pipelines(instructions)?;
        self.update_descriptors();
        self.reset(instructions);

        for (value_id, node) in instructions {
            self.execute_one(ctx, value_id, node, allocator)?;
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
        &mut self, ctx: &Context, instructions: &[ir::Instr], allocator: &mut AllocatorKind,
    ) -> Result<(), vk::Result> {
        self.execute(ctx, instructions, allocator)?;
        self.wait()?;
        allocator.free_command_buffers();

        Ok(())
    }

    fn record_push_constants(&mut self, pipeline: PipelineId, push: &PushConstants) -> Result<(), vk::Result> {
        if push.is_empty() {
            return Ok(());
        }

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

    fn bind_descriptor_sets(
        &mut self, pipeline: PipelineId, bind_point: vk::PipelineBindPoint,
    ) -> Result<(), vk::Result> {
        let bound = self.pipelines.get(pipeline.0 as usize).and_then(|declared| {
            let layout = declared.layout.as_ref()?;
            Some((layout.handle, declared.descriptors.as_ref().map(|d| d.sets().to_vec())))
        });

        let Some((handle, sets)) = bound else {
            return Ok(());
        };
        if self.bound_layout == Some((bind_point, handle)) {
            return Ok(());
        }
        self.bound_layout = Some((bind_point, handle));

        if let Some(sets) = sets {
            self.batch()?.bind_descriptor_sets(bind_point, handle, 0, &sets);
        }

        Ok(())
    }

    fn prepare_draw(
        &mut self, pipeline: Option<PipelineId>, state: &PipelineState, dynamic: &DynamicValues,
    ) -> Result<(), vk::Result> {
        let pipeline = pipeline.ok_or(vk::Result::ERROR_UNKNOWN)?;
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

        self.bind_descriptor_sets(pipeline, vk::PipelineBindPoint::GRAPHICS)?;

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

    fn prepare_dispatch(
        &mut self, pipeline: Option<PipelineId>, push_constants: &PushConstants,
    ) -> Result<(), vk::Result> {
        let pipeline = pipeline.ok_or(vk::Result::ERROR_UNKNOWN)?;
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

        self.bind_descriptor_sets(pipeline, vk::PipelineBindPoint::COMPUTE)?;

        self.record_push_constants(pipeline, push_constants)
    }

    pub fn dump(instructions: &[ir::Instr]) {
        println!("{}", dump::dump(instructions));
    }

    fn execute_one(
        &mut self, ctx: &Context, value_id: &ValueId, ir: &IR, allocator: &mut AllocatorKind,
    ) -> Result<(), vk::Result> {
        match ir {
            IR::Type(_) => {},
            IR::Constant(c) => match c {
                ir::Constant::I32(v) => self.set_value(value_id, Value::I32(*v)),
                ir::Constant::U32(v) => self.set_value(value_id, Value::U32(*v)),
                ir::Constant::Extent2D(v) => self.set_value(value_id, Value::Extent2D(*v)),
                ir::Constant::Extent3D(v) => self.set_value(value_id, Value::Extent3D(*v)),
                ir::Constant::Access(v) => self.set_value(value_id, Value::Access(*v)),
                ir::Constant::ClearValue(v) => self.set_value(value_id, Value::ClearValue(*v)),
            },
            IR::Array { ty: _, elements } => self.set_value(value_id, Value::Slice(elements.clone())),
            IR::Index { array, index } => {
                let elements = self.get::<Vec<ValueId>>(array);
                let index = self.get::<u32>(index) as usize;
                let element = *elements.get(index).ok_or(vk::Result::ERROR_UNKNOWN)?;
                self.set_value(value_id, Value::Reference(element));
            },
            IR::ConstructBuffer { buffer, .. } => {
                self.set_value(value_id, Value::Buffer(*buffer));
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
                let extent = self.get::<vk::Extent3D>(extent);
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
                        name: match name {
                            Some(name) => name.to_string(),
                            None => format!("transient image {value_id}"),
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
            IR::AcquireNextImage {
                swapchain,
                attachments,
                present_semaphores,
            } => {
                let acquire_semaphore = allocator.allocate_binary_semaphore()?;
                let image_index = ctx.acquire_next_image(*swapchain, acquire_semaphore)?;

                self.current_submit.wait_semas.push(SemaphoreSubmitInfo {
                    semaphore: acquire_semaphore,
                    value: 0,
                    access: Access::MemoryRW,
                });

                let elements = self.get::<Vec<ValueId>>(attachments);
                let element = *elements.get(image_index as usize).ok_or(vk::Result::ERROR_UNKNOWN)?;
                let present_semaphore = *present_semaphores
                    .get(image_index as usize)
                    .ok_or(vk::Result::ERROR_UNKNOWN)?;
                self.resource_to_swapchain
                    .insert(element, PresentInfo::new(image_index, *swapchain, present_semaphore));

                self.set_value(value_id, Value::U32(image_index));
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
            IR::CallOpaque { .. } => todo!(),
            IR::Clear { attachment, color } => {
                self.ensure_batch(ctx, allocator)?;
                self.set_value(value_id, Value::Reference(*attachment));
                let attachment = self.get::<ImageAttachment>(attachment);
                let color = self.get::<ClearValue>(color);
                let subresource_range = attachment.subresource_range();
                self.batch()?.clear_color(
                    attachment.image().into(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &color,
                    &[subresource_range],
                );
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
                render_area,
                ..
            } => {
                self.ensure_batch(ctx, allocator)?;

                let attachments = color_attachments
                    .iter()
                    .map(|id| self.get::<ImageAttachment>(id))
                    .collect::<Vec<_>>();

                let extent = match render_area {
                    Some(id) => self.get::<vk::Extent2D>(id),
                    None => attachments
                        .first()
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

                self.batch()?.begin_rendering(render_area, &attachment_infos);

                match color_attachments.first() {
                    Some(first) => self.set_value(value_id, Value::Reference(*first)),
                    None => self.set_value(value_id, Value::None),
                }
            },
            IR::BindPipeline { pass, .. } | IR::SetState { pass, .. } | IR::SampleImage { pass, .. } => {
                self.set_value(value_id, Value::Reference(*pass));
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
                self.prepare_draw(*pipeline, state, dynamic)?;

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
                self.prepare_draw(*pipeline, state, dynamic)?;

                let index_count = self.get::<u32>(index_count);
                let instance_count = self.get::<u32>(instance_count);
                let first_index = self.get::<u32>(first_index);
                let vertex_offset = self.get::<i32>(vertex_offset);
                let first_instance = self.get::<u32>(first_instance);
                self.batch()?
                    .draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance);
            },
            IR::EndRendering { pass } => {
                self.set_value(value_id, Value::Reference(*pass));
                self.batch()?.end_rendering();
            },
            IR::BeginCompute { resources, .. } => {
                self.ensure_batch(ctx, allocator)?;
                match resources.first() {
                    Some((first, _)) => self.set_value(value_id, Value::Reference(*first)),
                    None => self.set_value(value_id, Value::None),
                }
            },
            IR::EndCompute { pass } => self.set_value(value_id, Value::Reference(*pass)),
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
                        let local_size = pipeline.map_or([1; 3], |pipeline| self.local_size(pipeline));
                        let groups = [
                            invocations[0].div_ceil(local_size[0]),
                            invocations[1].div_ceil(local_size[1]),
                            invocations[2].div_ceil(local_size[2]),
                        ];
                        self.batch()?.dispatch(groups[0], groups[1], groups[2]);
                    },
                    ir::DispatchSize::Indirect { buffer, offset } => {
                        let handle = self.get::<Buffer>(buffer).handle();
                        self.batch()?.dispatch_indirect(handle, *offset);
                    },
                }
            },
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
        let device = self.device();

        if let Err(err) = unsafe { device.device_wait_idle() } {
            tracing::error!(?err, "failed to wait for the device before tearing down pipelines");
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

            if let Some(descriptors) = &declared.descriptors {
                descriptors.destroy(device);
            }

            if let Some(layout) = &declared.layout {
                layout.destroy(device);
            }
        }
    }
}
