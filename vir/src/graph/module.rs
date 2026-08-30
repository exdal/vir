use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use ash::vk;

use crate::{
    Access,
    AcquiredSwapchain,
    Buffer,
    BufferImageCopy,
    BufferInfo,
    ClearValue,
    ColorBlendAttachmentState,
    DepthState,
    DomainFlag,
    DynamicStateFlags,
    IR,
    Image,
    ImageAttachment,
    ImageInfo,
    LabelId,
    MemoryLocation,
    PassCallback,
    PassState,
    PipelineId,
    Program,
    RasterizationState,
    Rect2D,
    RenderingState,
    StateChange,
    SwapChain,
    Value,
    ValueId,
    Viewport,
    graph::{
        analysis::{PipelineBindings, analyze_descriptors},
        ir::{self, scaled},
        program::Variable,
    },
};

#[derive(Clone, Copy)]
struct ImageState {
    layout: vk::ImageLayout,
    last_access: ValueId,
    access: Access,
}

#[derive(Clone, Copy)]
struct BufferState {
    last_access: ValueId,
    access: Access,
}

struct Construct {
    merge: LabelId,
    entry_images: HashMap<ValueId, ImageState>,
    entry_buffers: HashMap<ValueId, BufferState>,
    arms: Vec<Arm>,
}

struct Arm {
    at: usize,
    images: HashMap<ValueId, ImageState>,
    buffers: HashMap<ValueId, BufferState>,
}

fn resolve_descriptor_access(
    mut nodes: Vec<ir::Instr>, pipelines: &impl PipelineBindings, next_id: &mut u32,
) -> Vec<ir::Instr> {
    let mut written: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut open = false;
    let mut bound: Option<PipelineId> = None;
    let mut resolved: BTreeMap<usize, Access> = BTreeMap::new();

    for (index, (_, ir)) in nodes.iter().enumerate() {
        match ir {
            IR::BeginRendering { .. } | IR::BeginCompute { .. } => {
                written.clear();
                open = true;
                bound = None;
            },
            IR::EndRendering { .. } | IR::EndCompute { .. } => {
                open = false;
                bound = None;
            },
            IR::BindPipeline { pipeline, .. } => bound = Some(*pipeline),

            IR::WriteDescriptor { set, binding, .. } if open => {
                written.insert((*set, *binding), index);
            },

            IR::Draw { .. } | IR::DrawIndexed { .. } | IR::CallOpaque { .. } | IR::Dispatch { .. } => {
                let Some(bindings) = bound.and_then(|pipeline| pipelines.bindings(pipeline)) else {
                    continue;
                };

                for binding in bindings {
                    if let Some(at) = written.get(&(binding.set, binding.binding)) {
                        *resolved.entry(*at).or_insert(Access::empty()) |= Access::sampled_by(binding.stages);
                    }
                }
            },

            _ => {},
        }
    }

    let mut access_ids = nodes
        .iter()
        .filter_map(|(id, ir)| match ir {
            IR::Constant(ir::Constant::Access(access)) => Some((*access, *id)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let mut constants = Vec::new();

    for (index, access) in resolved {
        let access_id = match access_ids.get(&access).copied() {
            Some(id) => id,
            None => {
                let id = alloc_id(next_id);
                constants.push((id, IR::Constant(ir::Constant::Access(access))));
                access_ids.insert(access, id);
                id
            },
        };
        if let IR::WriteDescriptor { access: slot, .. } = &mut nodes[index].1 {
            *slot = access_id;
        }
    }

    constants.extend(nodes);
    constants
}

fn globals_first(nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
    let (mut types, rest): (Vec<_>, Vec<_>) = nodes.into_iter().partition(|(_, ir)| matches!(ir, IR::Type(_)));
    let (constants, rest): (Vec<_>, Vec<_>) = rest.into_iter().partition(|(_, ir)| matches!(ir, IR::Constant(_)));
    types.extend(constants);
    types.extend(rest);
    types
}

fn alloc_id(next_id: &mut u32) -> ValueId {
    assert_ne!(*next_id, u32::MAX, "exhausted valid ValueId values");
    let id = ValueId(*next_id);
    *next_id += 1;
    id
}

#[derive(Clone, Copy)]
struct ImageType {
    format: vk::Format,
    samples: vk::SampleCountFlags,
    extent: Option<vk::Extent3D>,
}

fn name_of(name: &str) -> ir::Name { (!name.is_empty()).then(|| Arc::from(name)) }

#[derive(Clone, Copy, Default)]
enum Terminator {
    #[default]
    Return,
    Branch {
        target: LabelId,
    },
    BranchConditional {
        condition: ValueId,
        merge: LabelId,
        true_label: LabelId,
        false_label: LabelId,
    },
}

struct Block {
    label: LabelId,
    terminator: Terminator,
}

fn is_global(ir: &IR) -> bool {
    matches!(
        ir,
        IR::Type(_)
            | IR::Constant(_)
            | IR::Variable { .. }
            | IR::Array { .. }
            | IR::ConstructBuffer { .. }
            | IR::ConstructImage { .. }
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PassKind {
    Rendering,
    Compute,
}

impl PassKind {
    fn name(self) -> &'static str {
        match self {
            PassKind::Rendering => "rendering",
            PassKind::Compute => "compute",
        }
    }
}

#[derive(Clone, Copy)]
struct OngoingPass {
    kind: PassKind,
    begin: ValueId,
    tip: ValueId,
}

pub struct Module {
    types: HashMap<ir::Type, ValueId>,
    constants: HashMap<ir::Constant, ValueId>,
    instructions: Vec<IR>,
    variables: Vec<Variable>,
    variable_ids: Vec<ValueId>,
    blocks: Vec<Block>,
    instruction_block: Vec<Option<LabelId>>,
    label_count: u32,
    current_block: Option<LabelId>,
    ongoing_pass: Option<OngoingPass>,
}

impl Default for Module {
    fn default() -> Self {
        let entry = LabelId(0);
        Self {
            types: HashMap::default(),
            constants: HashMap::default(),
            instructions: Vec::new(),
            variables: Vec::new(),
            variable_ids: Vec::new(),
            blocks: vec![Block {
                label: entry,
                terminator: Terminator::default(),
            }],
            instruction_block: Vec::new(),
            label_count: 1,
            current_block: Some(entry),
            ongoing_pass: None,
        }
    }
}

impl Module {
    fn get(&self, id: ValueId) -> &IR { &self.instructions[id.0 as usize] }

    fn block_of(&self, id: ValueId) -> Option<LabelId> { self.instruction_block.get(id.0 as usize).copied().flatten() }

    fn resolve_access(&self, id: ValueId) -> Access {
        match self.get(id) {
            IR::Constant(ir::Constant::Access(a)) => *a,
            _ => panic!("{id} is not an Access constant"),
        }
    }

    fn emit(&mut self, ir: IR) -> ValueId {
        let id = ValueId(self.instructions.len() as u32);
        assert!(id.is_valid(), "exhausted valid ValueId values");
        self.instructions.push(ir);
        self.instruction_block.push(self.current_block);
        id
    }

    fn declare_label(&mut self) -> LabelId {
        let label = LabelId(self.label_count);
        self.label_count += 1;
        label
    }

    fn place_label(&mut self, label: LabelId) {
        if self.blocks.iter().any(|block| block.label == label) {
            tracing::error!(%label, "label is placed more than once");
            return;
        }

        self.blocks.push(Block {
            label,
            terminator: Terminator::default(),
        });
        self.current_block = Some(label);
    }

    fn branch(&mut self, target: LabelId) { self.terminate(Terminator::Branch { target }) }

    fn branch_conditional(&mut self, condition: ValueId, merge: LabelId, true_label: LabelId, false_label: LabelId) {
        self.terminate(Terminator::BranchConditional {
            condition,
            merge,
            true_label,
            false_label,
        })
    }

    fn terminate(&mut self, terminator: Terminator) {
        match self.blocks.last_mut() {
            Some(block) if matches!(block.terminator, Terminator::Return) => block.terminator = terminator,
            Some(block) => tracing::error!(label = %block.label, "block is terminated twice"),
            None => tracing::error!("a branch was recorded before any block was opened"),
        }
    }

    fn phi(&mut self, incoming: &[(ValueId, LabelId)]) -> ValueId {
        self.emit(IR::Phi {
            incoming: incoming.to_vec(),
        })
    }

    pub fn set_condition(
        &mut self, condition: ValueId, then: impl FnOnce(&mut Module) -> ValueId,
        otherwise: impl FnOnce(&mut Module) -> ValueId,
    ) -> ValueId {
        self.expect_no_open_pass("a selection is recorded");

        let merge = self.declare_label();
        let true_label = self.declare_label();
        let false_label = self.declare_label();

        self.branch_conditional(condition, merge, true_label, false_label);

        self.place_label(true_label);
        let taken = then(self);
        self.expect_no_open_pass("an arm ends");
        self.branch(merge);

        self.place_label(false_label);
        let skipped = otherwise(self);
        self.expect_no_open_pass("an arm ends");
        self.branch(merge);

        self.place_label(merge);
        self.phi(&[(taken, true_label), (skipped, false_label)])
    }

    fn declare_var(&mut self, kind: ir::VariableKind, name: &str, value: Value) -> ValueId {
        self.declare_var_at(kind, name, value, ir::SourceLocation::NONE)
    }

    fn declare_var_at(
        &mut self, kind: ir::VariableKind, name: &str, value: Value, location: ir::SourceLocation,
    ) -> ValueId {
        let slot = self.variables.len() as u32;
        let name: ir::Name = Some(Arc::from(name));
        let lowered_name = self.lower_name(name.clone());
        let id = self.emit(IR::Variable {
            slot,
            kind,
            name: lowered_name,
            location,
        });
        self.variables.push(Variable {
            kind,
            name,
            value,
            resource: None,
        });
        self.variable_ids.push(id);
        id
    }

    fn declare_resource_var(&mut self, kind: ir::VariableKind, name: &str, resource: ValueId) -> ValueId {
        self.variables.push(Variable {
            kind,
            name: name_of(name),
            value: Value::None,
            resource: Some(resource),
        });
        self.variable_ids.push(resource);
        resource
    }

    pub fn declare_buffer_var(&mut self, name: &str, access: Access) -> ValueId {
        self.lower_type(ir::Type::Buffer);
        let initial_access = self.lower_access(access);
        let lowered_name = self.lower_name(name_of(name));
        let resource = self.emit(IR::ConstructBuffer {
            buffer: Buffer::default(),
            size: ValueId::INVALID,
            usage: vk::BufferUsageFlags::empty(),
            location: MemoryLocation::Unknown,
            initial_access,
            name: lowered_name,
        });

        self.declare_resource_var(ir::VariableKind::Buffer, name, resource)
    }

    pub fn declare_image_var(
        &mut self, name: &str, format: vk::Format, samples: vk::SampleCountFlags, layout: vk::ImageLayout,
    ) -> ValueId {
        let base_level = self.lower_u32(0);
        let level_count = self.lower_u32(1);
        let base_layer = self.lower_u32(0);
        let layer_count = self.lower_u32(1);
        self.lower_type(ir::Type::Image { format, samples });
        let lowered_name = self.lower_name(name_of(name));

        let resource = self.emit(IR::ConstructImage {
            image: Image::default(),
            image_view: vk::ImageView::null(),
            view_type: vk::ImageViewType::TYPE_2D,
            extent: ValueId::INVALID,
            format,
            samples,
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: vk::ImageUsageFlags::empty(),
            initial_layout: layout,
            name: lowered_name,
        });

        self.declare_resource_var(ir::VariableKind::ImageAttachment, name, resource)
    }

    pub fn constant_u32(&mut self, value: u32) -> ValueId { self.lower_u32(value) }

    pub fn declare_bool_var(&mut self, name: &str, default: bool) -> ValueId {
        self.declare_var(ir::VariableKind::Bool, name, Value::Bool(default))
    }

    pub fn declare_i32_var(&mut self, name: &str, default: i32) -> ValueId {
        self.declare_var(ir::VariableKind::I32, name, Value::I32(default))
    }

    pub fn declare_u32_var(&mut self, name: &str, default: u32) -> ValueId {
        self.declare_var(ir::VariableKind::U32, name, Value::U32(default))
    }

    pub fn declare_extent_2d_var(&mut self, name: &str, default: vk::Extent2D) -> ValueId {
        self.declare_var(ir::VariableKind::Extent2D, name, Value::Extent2D(default))
    }

    pub fn declare_extent_3d_var(&mut self, name: &str, default: vk::Extent3D) -> ValueId {
        self.declare_var(ir::VariableKind::Extent3D, name, Value::Extent3D(default))
    }

    pub fn declare_clear_var(&mut self, name: &str, default: ClearValue) -> ValueId {
        self.declare_var(ir::VariableKind::ClearValue, name, Value::ClearValue(default))
    }

    #[track_caller]
    pub fn declare_callback_var(&mut self, name: &str) -> ValueId {
        self.declare_var_at(
            ir::VariableKind::Callback,
            name,
            Value::Callback(PassCallback::empty()),
            ir::SourceLocation::caller(),
        )
    }

    pub fn declare_bytes_var(&mut self, name: &str, size: u32) -> ValueId {
        let zeroed = Value::Bytes(vec![0u8; size as usize].into());
        self.declare_var(ir::VariableKind::Bytes(size), name, zeroed)
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
        let id = self.emit(IR::Constant(constant.clone()));
        self.constants.insert(constant, id);
        id
    }

    fn lower_i32(&mut self, v: i32) -> ValueId { self.lower_constant(ir::Constant::I32(v)) }

    fn lower_u32(&mut self, v: u32) -> ValueId { self.lower_constant(ir::Constant::U32(v)) }

    fn lower_access(&mut self, access: Access) -> ValueId { self.lower_constant(ir::Constant::Access(access)) }

    fn lower_string(&mut self, string: Arc<str>) -> ValueId { self.lower_constant(ir::Constant::String(string)) }

    fn lower_name(&mut self, name: ir::Name) -> ValueId {
        name.map_or(ValueId::INVALID, |name| self.lower_string(name))
    }

    fn lower_image_attachment(&mut self, attachment: &ImageAttachment, name: ir::Name) -> (ValueId, ValueId) {
        let extent = self.lower_constant(ir::Constant::Extent3D(attachment.extent()));
        let base_level = self.lower_u32(attachment.base_level());
        let level_count = self.lower_u32(attachment.level_count());
        let base_layer = self.lower_u32(attachment.base_layer());
        let layer_count = self.lower_u32(attachment.layer_count());
        let name = self.lower_name(name);

        let ty_instr = self.lower_type(ir::Type::Image {
            format: attachment.format(),
            samples: attachment.samples(),
        });
        let view_type = if attachment.layer_count() > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let construct_instr = self.emit(IR::ConstructImage {
            image: *attachment.image(),
            image_view: attachment.image_view(),
            view_type,
            extent,
            format: attachment.format(),
            samples: attachment.samples(),
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: attachment.image().usage(),
            initial_layout: attachment.layout(),
            name,
        });

        (ty_instr, construct_instr)
    }

    pub fn transient_image(&mut self, info: &ImageInfo) -> ValueId {
        let extent = self.lower_constant(ir::Constant::Extent3D(info.extent));
        self.transient_image_sized(info, extent)
    }

    pub fn transient_image_sized(&mut self, info: &ImageInfo, extent: ValueId) -> ValueId {
        let base_level = self.lower_u32(0);
        let level_count = self.lower_u32(info.mip_levels);
        let base_layer = self.lower_u32(0);
        let layer_count = self.lower_u32(info.array_layers);
        let name = self.lower_name(name_of(&info.name));

        self.lower_type(ir::Type::Image {
            format: info.format,
            samples: info.samples,
        });
        let view_type = if info.array_layers > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };

        self.emit(IR::ConstructImage {
            image: Image::default(),
            image_view: vk::ImageView::null(),
            view_type,
            extent,
            format: info.format,
            samples: info.samples,
            base_level,
            level_count,
            base_layer,
            layer_count,
            usage: info.usage,
            initial_layout: vk::ImageLayout::UNDEFINED,
            name,
        })
    }

    pub fn import_image(&mut self, image: &Image, layout: vk::ImageLayout) -> ValueId {
        self.import_attachment(&ImageAttachment::from_image(image, layout))
    }

    pub fn import_attachment(&mut self, attachment: &ImageAttachment) -> ValueId {
        self.lower_image_attachment(attachment, None).1
    }

    pub fn transient_buffer(&mut self, info: &BufferInfo) -> ValueId {
        let size = self.lower_constant(ir::Constant::Size(info.size as usize));
        let initial_access = self.lower_access(Access::None);
        let name = self.lower_name(name_of(&info.name));
        self.lower_type(ir::Type::Buffer);
        self.emit(IR::ConstructBuffer {
            buffer: Buffer::default(),
            size,
            usage: info.usage,
            location: info.location,
            initial_access,
            name,
        })
    }

    pub fn import_buffer(&mut self, buffer: &Buffer, access: Access) -> ValueId {
        self.lower_type(ir::Type::Buffer);
        let size = self.lower_u32(buffer.size() as u32);
        let initial_access = self.lower_access(access);
        self.emit(IR::ConstructBuffer {
            buffer: *buffer,
            size,
            usage: vk::BufferUsageFlags::empty(),
            location: MemoryLocation::Unknown,
            initial_access,
            name: ValueId::INVALID,
        })
    }

    pub fn set_name(&mut self, id: ValueId, name: impl Into<Arc<str>>) -> ValueId {
        if !matches!(
            self.instructions.get(id.0 as usize),
            Some(
                IR::ConstructImage { .. }
                    | IR::ConstructBuffer { .. }
                    | IR::BeginRendering { .. }
                    | IR::BeginCompute { .. }
            )
        ) {
            tracing::warn!(%id, "value cannot carry a name; the name is dropped");
            return id;
        }

        let name = self.lower_string(name.into());
        match self.instructions.get_mut(id.0 as usize) {
            Some(
                IR::ConstructImage { name: slot, .. }
                | IR::ConstructBuffer { name: slot, .. }
                | IR::BeginRendering { name: slot, .. }
                | IR::BeginCompute { name: slot, .. },
            ) => *slot = name,
            _ => unreachable!("the name target was checked above"),
        }

        id
    }

    pub fn declare_swapchain_var(&mut self, name: &str) -> ValueId {
        self.declare_var(
            ir::VariableKind::Swapchain,
            name,
            Value::Swapchain(AcquiredSwapchain::default()),
        )
    }

    pub fn acquire_next_image_from(
        &mut self, swapchain: ValueId, format: vk::Format, samples: vk::SampleCountFlags,
    ) -> ValueId {
        self.lower_type(ir::Type::Image { format, samples });
        let acquire = self.emit(IR::AcquireNextImage { swapchain });
        self.emit(IR::SwapchainImage {
            swapchain,
            acquire,
            format,
            samples,
        })
    }

    pub fn acquire_next_image(&mut self, swapchain: &SwapChain) -> ValueId {
        let variable = self.declare_swapchain_var("swapchain");
        if let Some(declared) = self.variables.last_mut() {
            declared.value = Value::Swapchain(AcquiredSwapchain::from(swapchain));
        }

        self.acquire_next_image_from(variable, swapchain.format(), swapchain.samples())
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
        self.release(attachment, Access::Present, DomainFlag::Present)
    }

    pub fn blit(&mut self, src: ValueId, dst: ValueId) -> ValueId { self.blit_filtered(src, dst, vk::Filter::LINEAR) }

    pub fn blit_filtered(&mut self, src: ValueId, dst: ValueId, filter: vk::Filter) -> ValueId {
        self.emit(IR::Blit { src, dst, filter })
    }

    pub fn copy_buffer_to_image(&mut self, buffer: ValueId, image: ValueId) -> ValueId {
        // an image bound to a variable has no extent to read here, so the whole of it is left
        // for the interpreter to measure off whatever was bound
        let Some(extent) = self.resolve_image(image).and_then(|image| image.extent) else {
            if !self.is_image(image) {
                tracing::warn!(%image, "cannot infer the extent to copy into; the copy is dropped");
                return image;
            }

            return self.emit(IR::CopyBufferToImage {
                buffer,
                image,
                region: None,
            });
        };

        self.copy_buffer_to_image_region(buffer, image, BufferImageCopy::whole(extent))
    }

    pub fn copy_buffer_to_image_region(&mut self, buffer: ValueId, image: ValueId, region: BufferImageCopy) -> ValueId {
        if region.is_empty() {
            tracing::warn!(%image, "copy region is empty; the copy is dropped");
            return image;
        }

        self.emit(IR::CopyBufferToImage {
            buffer,
            image,
            region: Some(region),
        })
    }

    pub fn clear(&mut self, attachment: ValueId, color: ClearValue) -> ValueId {
        let color = self.lower_constant(ir::Constant::ClearValue(color));
        self.clear_from(attachment, color)
    }

    pub fn clear_from(&mut self, attachment: ValueId, color: ValueId) -> ValueId {
        self.emit(IR::Clear { attachment, color })
    }

    pub fn begin_rendering(&mut self, color_attachments: &[ValueId]) -> &mut Self {
        self.begin_rendering_with(color_attachments, None, None)
    }

    pub fn begin_rendering_depth(&mut self, color_attachments: &[ValueId], depth: ValueId) -> &mut Self {
        self.begin_rendering_with(color_attachments, Some(depth), None)
    }

    pub fn begin_rendering_area(&mut self, color_attachments: &[ValueId], render_area: vk::Extent2D) -> &mut Self {
        let render_area = self.lower_constant(ir::Constant::Extent2D(render_area));
        self.begin_rendering_with(color_attachments, None, Some(render_area))
    }

    pub fn begin_rendering_depth_area(
        &mut self, color_attachments: &[ValueId], depth: ValueId, render_area: vk::Extent2D,
    ) -> &mut Self {
        let render_area = self.lower_constant(ir::Constant::Extent2D(render_area));
        self.begin_rendering_with(color_attachments, Some(depth), Some(render_area))
    }

    fn begin_rendering_with(
        &mut self, color_attachments: &[ValueId], depth_attachment: Option<ValueId>, render_area: Option<ValueId>,
    ) -> &mut Self {
        let depth_attachment = depth_attachment.unwrap_or(ValueId::INVALID);
        let render_area = render_area.unwrap_or(ValueId::INVALID);
        let id = self.emit(IR::BeginRendering {
            color_attachments: color_attachments.to_vec(),
            depth_attachment,
            render_area,
            declared_access: Vec::new(),
            name: ValueId::INVALID,
        });
        self.open_pass(PassKind::Rendering, id)
    }

    pub fn begin_compute(&mut self) -> &mut Self {
        let id = self.emit(IR::BeginCompute {
            declared_access: Vec::new(),
            name: ValueId::INVALID,
        });
        self.open_pass(PassKind::Compute, id)
    }

    pub fn compile(&self, pipelines: &impl PipelineBindings, id: ValueId) -> Result<Program, vk::Result> {
        self.compile_all(pipelines, &[id])
    }

    pub fn compile_all(&self, pipelines: &impl PipelineBindings, ids: &[ValueId]) -> Result<Program, vk::Result> {
        let program = self.lower(pipelines, ids);
        analyze_descriptors(&program, pipelines)?;
        Ok(program)
    }

    fn lower(&self, pipelines: &impl PipelineBindings, ids: &[ValueId]) -> Program {
        let mut roots = ids.to_vec();
        roots.extend(self.branch_conditions());
        roots.extend(
            self.instructions
                .iter()
                .enumerate()
                .filter_map(|(index, ir)| matches!(ir, IR::Type(_)).then_some(ValueId(index as u32))),
        );

        let mut next_id = self.instructions.len() as u32;
        let nodes = self.topo_sort(&roots);
        let nodes = self.layout_blocks(nodes, &mut next_id);
        // the barriers are what the resolved accesses are for, so this comes before them
        let nodes = resolve_descriptor_access(nodes, pipelines, &mut next_id);
        let nodes = self.sync(nodes, &mut next_id);
        let nodes = self.simplify_cfg(nodes);
        let nodes = self.fold_barriers(nodes, &mut next_id);
        let mut nodes = globals_first(nodes);
        self.infer_usage(&mut nodes);

        let slots: HashMap<ValueId, u32> = self
            .variable_ids
            .iter()
            .enumerate()
            .map(|(slot, id)| (*id, slot as u32))
            .collect();
        let nodes = self.infer(nodes);

        Program::new(nodes, self.variables.clone(), slots)
    }

    fn topo_sort(&self, roots: &[ValueId]) -> Vec<ir::Instr> {
        let mut nodes = Vec::new();
        let mut stack = roots.iter().rev().copied().collect::<Vec<_>>();
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
                    IR::Variable { name, .. } => {
                        if name.is_valid() {
                            stack.push(*name);
                        }
                    },
                    IR::Array { ty, elements } => {
                        stack.push(*ty);
                        elements.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::ConstructBuffer {
                        size,
                        initial_access,
                        name,
                        ..
                    } => {
                        if name.is_valid() {
                            stack.push(*name);
                        }
                        stack.push(*initial_access);
                        if size.is_valid() {
                            stack.push(*size);
                        }
                    },
                    IR::ConstructImage {
                        extent,
                        base_level,
                        level_count,
                        base_layer,
                        layer_count,
                        name,
                        ..
                    } => {
                        if name.is_valid() {
                            stack.push(*name);
                        }
                        stack.push(*layer_count);
                        stack.push(*base_layer);
                        stack.push(*level_count);
                        stack.push(*base_level);
                        if extent.is_valid() {
                            stack.push(*extent);
                        }
                    },
                    IR::AcquireNextImage { swapchain } => {
                        stack.push(*swapchain);
                    },
                    IR::SwapchainImage { swapchain, acquire, .. } => {
                        stack.push(*acquire);
                        stack.push(*swapchain);
                    },
                    IR::Index { array, index } => {
                        stack.push(*index);
                        stack.push(*array);
                    },
                    IR::Acquire { access, resource, .. } => {
                        stack.push(*access);
                        stack.push(*resource);
                    },
                    IR::Release { access, resource, .. } => {
                        stack.push(*access);
                        stack.push(*resource);
                    },
                    IR::Clear { color, attachment } => {
                        stack.push(*color);
                        stack.push(*attachment);
                    },
                    IR::Blit { src, dst, .. } => {
                        stack.push(*dst);
                        stack.push(*src);
                    },
                    IR::CopyBufferToImage { buffer, image, .. } => {
                        stack.push(*image);
                        stack.push(*buffer);
                    },
                    IR::BeginRendering {
                        color_attachments,
                        depth_attachment,
                        render_area,
                        declared_access,
                        name,
                        ..
                    } => {
                        if name.is_valid() {
                            stack.push(*name);
                        }
                        declared_access.iter().rev().for_each(|(resource, access)| {
                            stack.push(*access);
                            stack.push(*resource);
                        });
                        if render_area.is_valid() {
                            stack.push(*render_area);
                        }
                        if depth_attachment.is_valid() {
                            stack.push(*depth_attachment);
                        }
                        color_attachments.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::BindPipeline { pass, .. } => {
                        stack.push(*pass);
                    },
                    IR::SetState { pass, change } => {
                        stack.push(*pass);
                        match change {
                            StateChange::PushConstantsFrom { source, .. } => stack.push(*source),
                            // a transient reached only through its address still has to be built
                            StateChange::PushConstantAddress { buffer, .. } => stack.push(*buffer),
                            _ => {},
                        }
                    },
                    IR::BindVertexBuffers { pass, buffers, .. } => {
                        stack.push(*pass);
                        buffers.iter().rev().for_each(|v| stack.push(*v));
                    },
                    IR::BindIndexBuffer { pass, buffer, .. } => {
                        stack.push(*pass);
                        stack.push(*buffer);
                    },
                    IR::WriteDescriptor {
                        pass,
                        descriptor,
                        access,
                        ..
                    } => {
                        stack.push(*access);
                        stack.push(*pass);
                        stack.push(descriptor.image());
                    },
                    IR::Draw {
                        pass,
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                        ..
                    } => {
                        stack.push(*first_instance);
                        stack.push(*first_vertex);
                        stack.push(*instance_count);
                        stack.push(*vertex_count);
                        stack.push(*pass);
                    },
                    IR::DrawIndexed {
                        pass,
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                        ..
                    } => {
                        stack.push(*first_instance);
                        stack.push(*vertex_offset);
                        stack.push(*first_index);
                        stack.push(*instance_count);
                        stack.push(*index_count);
                        stack.push(*pass);
                    },
                    IR::CallOpaque { pass, body, .. } => {
                        stack.push(*body);
                        stack.push(*pass);
                    },
                    IR::EndRendering { pass } => {
                        stack.push(*pass);
                    },
                    IR::BeginCompute { declared_access, name } => {
                        if name.is_valid() {
                            stack.push(*name);
                        }
                        declared_access.iter().rev().for_each(|(resource, access)| {
                            stack.push(*access);
                            stack.push(*resource);
                        });
                    },
                    IR::Dispatch { pass, size, .. } => {
                        match size {
                            ir::DispatchSize::Groups { x, y, z } | ir::DispatchSize::Invocations { x, y, z } => {
                                stack.push(*z);
                                stack.push(*y);
                                stack.push(*x);
                            },
                            ir::DispatchSize::InvocationsPerPixel { image, .. } => stack.push(*image),
                            ir::DispatchSize::InvocationsPerElement { buffer, .. }
                            | ir::DispatchSize::Indirect { buffer, .. } => stack.push(*buffer),
                        }
                        stack.push(*pass);
                    },
                    IR::EndCompute { pass } => {
                        stack.push(*pass);
                    },
                    IR::Label { .. }
                    | IR::SelectionMerge { .. }
                    | IR::Branch { .. }
                    | IR::BranchConditional { .. }
                    | IR::Return => {},
                    IR::Phi { incoming } => {
                        incoming.iter().rev().for_each(|(value, _)| stack.push(*value));
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

    fn layout_blocks(&self, nodes: Vec<ir::Instr>, next_id: &mut u32) -> Vec<ir::Instr> {
        let mut globals = Vec::with_capacity(nodes.len());
        let mut recorded: HashMap<LabelId, Vec<ir::Instr>> = HashMap::new();

        for (id, ir) in nodes {
            match self.block_of(id) {
                _ if is_global(&ir) => globals.push((id, ir)),
                Some(label) => recorded.entry(label).or_default().push((id, ir)),
                None => globals.push((id, ir)),
            }
        }

        let mut result = globals;
        for block in &self.blocks {
            result.push((alloc_id(next_id), IR::Label { label: block.label }));
            result.extend(recorded.remove(&block.label).unwrap_or_default());

            match block.terminator {
                Terminator::Return => {
                    result.push((alloc_id(next_id), IR::Return));
                },
                Terminator::Branch { target } => {
                    result.push((alloc_id(next_id), IR::Branch { target }));
                },
                Terminator::BranchConditional {
                    condition,
                    merge,
                    true_label,
                    false_label,
                } => {
                    result.push((alloc_id(next_id), IR::SelectionMerge { merge }));
                    result.push((
                        alloc_id(next_id),
                        IR::BranchConditional {
                            condition,
                            true_label,
                            false_label,
                        },
                    ));
                },
            }
        }

        result
    }

    fn branch_conditions(&self) -> Vec<ValueId> {
        self.blocks
            .iter()
            .filter_map(|block| match block.terminator {
                Terminator::BranchConditional { condition, .. } => Some(condition),
                _ => None,
            })
            .collect()
    }

    fn join_arms(
        &self, construct: Construct, predecessors: usize, result: &mut Vec<ir::Instr>, next_id: &mut u32,
        undefined: ImageState, live: &HashSet<ValueId>,
    ) -> (HashMap<ValueId, ImageState>, HashMap<ValueId, BufferState>) {
        let Construct {
            entry_images,
            entry_buffers,
            mut arms,
            ..
        } = construct;

        let join_from_arm = !arms.is_empty() && arms.len() == predecessors;
        let (join_images, join_buffers) = match join_from_arm {
            true => (arms[0].images.clone(), arms[0].buffers.clone()),
            false => (entry_images.clone(), entry_buffers.clone()),
        };

        // query images/buffers that live in current arm so we don't get dead barriers
        let mut images = join_images
            .iter()
            .filter(|(resource, _)| live.contains(*resource))
            .collect::<Vec<_>>();
        images.sort_by_key(|(resource, _)| resource.0);
        let mut buffers = join_buffers
            .iter()
            .filter(|(buffer, _)| live.contains(*buffer))
            .collect::<Vec<_>>();
        buffers.sort_by_key(|(buffer, _)| buffer.0);

        // an insertion shifts everything after it, so the arms are patched back to front
        let first_to_patch = usize::from(join_from_arm);
        for arm in arms.drain(first_to_patch..).rev() {
            let mut barriers = Vec::new();

            for (resource, target) in &images {
                let current = arm
                    .images
                    .get(*resource)
                    .or_else(|| entry_images.get(*resource))
                    .copied()
                    .unwrap_or(undefined);
                if current.layout == target.layout && current.access == target.access {
                    continue;
                }

                let id = alloc_id(next_id);
                barriers.push((
                    id,
                    IR::ImageBarrier {
                        src_access: current.last_access,
                        dst_access: target.last_access,
                        old_layout: current.layout,
                        new_layout: target.layout,
                        value: **resource,
                    },
                ));
            }

            let mut emitted: Vec<(Access, Access)> = Vec::new();
            for (buffer, target) in &buffers {
                let current = arm.buffers.get(*buffer).copied().unwrap_or(BufferState {
                    last_access: undefined.last_access,
                    access: Access::None,
                });

                if current.access == target.access || current.access == Access::None {
                    continue;
                }
                if emitted.contains(&(current.access, target.access)) {
                    continue;
                }
                emitted.push((current.access, target.access));

                let id = alloc_id(next_id);
                barriers.push((
                    id,
                    IR::MemoryBarrier {
                        src_access: current.last_access,
                        dst_access: target.last_access,
                    },
                ));
            }

            result.splice(arm.at..arm.at, barriers);
        }

        (join_images, join_buffers)
    }

    fn sync(&self, nodes: Vec<ir::Instr>, next_id: &mut u32) -> Vec<ir::Instr> {
        let access_values = nodes
            .iter()
            .filter_map(|(id, ir)| match ir {
                IR::Constant(ir::Constant::Access(access)) => Some((*id, *access)),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let resolve_access = |id: &ValueId| {
            access_values
                .get(id)
                .copied()
                .unwrap_or_else(|| panic!("{id} is not an Access constant"))
        };
        let mut result = Vec::with_capacity(nodes.len() * 2);
        let mut image_states = HashMap::new();
        let mut buffer_states = HashMap::new();

        // how many ways there are into each block, which decides whether the state one arm
        // ends in can stand as the join or whether control can also fall straight through
        let mut predecessors: HashMap<LabelId, usize> = HashMap::new();
        for (_, ir) in &nodes {
            let targets = match ir {
                IR::Branch { target } => vec![*target],
                IR::BranchConditional {
                    true_label,
                    false_label,
                    ..
                } => vec![*true_label, *false_label],
                _ => continue,
            };
            for target in targets {
                *predecessors.entry(target).or_default() += 1;
            }
        }

        let mut region_buffers: HashMap<ValueId, Vec<(ValueId, Access)>> = HashMap::new();
        let mut region_images: HashMap<ValueId, Vec<(ValueId, Access)>> = HashMap::new();
        let mut open_region: Option<ValueId> = None;
        for (value_id, ir) in &nodes {
            let (buffers, access) = match ir {
                IR::BeginRendering { .. } | IR::BeginCompute { .. } => {
                    open_region = Some(*value_id);
                    continue;
                },
                IR::EndRendering { .. } | IR::EndCompute { .. } => {
                    open_region = None;
                    continue;
                },
                IR::WriteDescriptor { descriptor, access, .. } => {
                    let access = resolve_access(access);
                    if let Some(region) = open_region {
                        let root = self.resource_root(descriptor.image());
                        let sampled = region_images.entry(region).or_default();
                        match sampled.iter_mut().find(|(seen, _)| *seen == root) {
                            Some((_, merged)) => *merged |= access,
                            None => sampled.push((root, access)),
                        }
                    }
                    continue;
                },
                IR::BindVertexBuffers { buffers, .. } => (buffers.as_slice(), Access::AttributeRead),
                IR::BindIndexBuffer { buffer, .. } => (std::slice::from_ref(buffer), Access::IndexRead),
                _ => continue,
            };

            let Some(region) = open_region else {
                continue;
            };
            let bound = region_buffers.entry(region).or_default();
            for buffer in buffers {
                let root = self.resource_root(*buffer);
                if !bound.iter().any(|(bound, _)| *bound == root) {
                    bound.push((root, access));
                }
            }
        }

        // which declared_access are still named past each label, walked backwards so every label
        // sees what comes after it. a selection is the only construct there is, so program
        // order is the whole of what can follow a merge
        let mut live_after: HashMap<LabelId, HashSet<ValueId>> = HashMap::new();
        let mut live: HashSet<ValueId> = HashSet::new();
        let mut uses = Vec::new();
        for (_, ir) in nodes.iter().rev() {
            if let IR::Label { label } = ir {
                live_after.insert(*label, live.clone());
            }

            uses.clear();
            self.resource_uses(ir, &mut uses);
            live.extend(uses.iter().copied());
        }

        macro_rules! alloc {
            () => {{ alloc_id(next_id) }};
        }
        macro_rules! emit_access {
            ($access:expr) => {{
                let id = alloc!();
                result.push((id, IR::Constant(ir::Constant::Access($access))));
                id
            }};
        }
        let mut emitted_memory: Vec<(Access, Access)> = Vec::new();
        macro_rules! emit_memory_barrier {
            ($src:expr, $dst:expr) => {{
                let (src, dst): (BufferState, BufferState) = ($src, $dst);
                if !emitted_memory.contains(&(src.access, dst.access)) {
                    emitted_memory.push((src.access, dst.access));
                    let id = alloc!();
                    result.push((
                        id,
                        IR::MemoryBarrier {
                            src_access: src.last_access,
                            dst_access: dst.last_access,
                        },
                    ));
                }
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
        let mut host_write_id: Option<ValueId> = None;
        let mut read_access_ids: HashMap<Access, ValueId> = HashMap::new();

        // the constant for an access a resource is only ever read at is worth sharing, since
        // every barrier that waits on the same read names it
        macro_rules! read_access {
            ($access:expr) => {{
                let access: Access = $access;
                match read_access_ids.get(&access).copied() {
                    Some(id) => id,
                    None => {
                        let id = emit_access!(access);
                        read_access_ids.insert(access, id);
                        id
                    },
                }
            }};
        }

        // every buffer the host filled rests at the same point, so one constant names it
        macro_rules! host_write {
            () => {{
                match host_write_id {
                    Some(id) => id,
                    None => {
                        let id = emit_access!(Access::HostWrite);
                        host_write_id = Some(id);
                        id
                    },
                }
            }};
        }

        /// The access a buffer rests at until something in the module touches it, which is what
        /// its first barrier waits on. A buffer the graph allocates for this run rests at
        /// nothing, since there is no earlier use of it to wait for.
        macro_rules! resting_access {
            ($access:expr) => {{
                let access: Access = $access;
                match access {
                    Access::None => no_access_id,
                    Access::HostWrite => host_write!(),
                    // a write is never shared, so it stays a barrier point of its own
                    _ if access.writes() => emit_access!(access),
                    _ => read_access!(access),
                }
            }};
        }

        /// A buffer the host filled is visible to the GPU only after a barrier, and stays
        /// visible until something writes it again, which only a dispatch does today.
        macro_rules! buffer_barrier {
            ($buffer:expr, $access:expr) => {{
                let buffer = $buffer;
                let access: Access = $access;
                let last = match buffer_states.get(&buffer).copied() {
                    Some(state) => state,
                    None => BufferState {
                        last_access: host_write!(),
                        access: Access::HostWrite,
                    },
                };

                // a write is never shared, so it always ends up with a barrier of its own
                let next = BufferState {
                    last_access: match access.writes() {
                        true => emit_access!(access),
                        false => read_access!(access),
                    },
                    access,
                };
                if last.last_access != next.last_access {
                    if last.access != Access::None {
                        emit_memory_barrier!(last, next);
                    }
                    buffer_states.insert(buffer, next);
                }
            }};
        }

        let undefined = ImageState {
            layout: vk::ImageLayout::UNDEFINED,
            last_access: no_access_id,
            access: Access::None,
        };

        macro_rules! transition {
            ($resource:expr, $access_id:expr, $access:expr) => {{
                let resource = $resource;
                let access: Access = $access;
                let root = self.resource_root(resource);
                let state = image_states.get(&root).copied().unwrap_or(undefined);
                let new_layout: vk::ImageLayout = access.into();

                let new_state = if state.layout == new_layout && !state.access.writes() && !access.writes() {
                    let merged = state.access | access;
                    let last_access = if merged == state.access {
                        state.last_access
                    } else {
                        emit_access!(merged)
                    };
                    ImageState {
                        layout: new_layout,
                        last_access,
                        access: merged,
                    }
                } else {
                    emit_barrier!(state.last_access, $access_id, state.layout, new_layout, resource);
                    ImageState {
                        layout: new_layout,
                        last_access: $access_id,
                        access,
                    }
                };

                image_states.insert(root, new_state);
            }};
        }

        // a selection leaves a resource in one of two states, and what follows the merge
        // cannot know which; each construct is walked from the state it was entered with
        // and the arms are brought back into agreement before control joins
        let mut constructs: Vec<Construct> = Vec::new();

        for (value_id, ir) in nodes {
            emitted_memory.clear();

            match &ir {
                IR::SelectionMerge { merge, .. } => {
                    constructs.push(Construct {
                        merge: *merge,
                        entry_images: image_states.clone(),
                        entry_buffers: buffer_states.clone(),
                        arms: Vec::new(),
                    });
                },

                IR::Branch { target, .. } if constructs.last().is_some_and(|c| c.merge == *target) => {
                    let construct = constructs.last_mut().expect("just matched on the top construct");
                    construct.arms.push(Arm {
                        at: result.len(),
                        images: std::mem::replace(&mut image_states, construct.entry_images.clone()),
                        buffers: std::mem::replace(&mut buffer_states, construct.entry_buffers.clone()),
                    });
                },

                IR::Label { label } if constructs.last().is_some_and(|c| c.merge == *label) => {
                    let construct = constructs.pop().expect("just matched on the top construct");
                    let ways_in = predecessors.get(label).copied().unwrap_or_default();
                    let live = live_after.remove(label).unwrap_or_default();
                    let joined = self.join_arms(construct, ways_in, &mut result, next_id, undefined, &live);
                    image_states = joined.0;
                    buffer_states = joined.1;
                },

                IR::ConstructImage { initial_layout, .. } => {
                    image_states.insert(
                        value_id,
                        ImageState {
                            layout: *initial_layout,
                            last_access: no_access_id,
                            access: Access::None,
                        },
                    );
                },

                IR::ConstructBuffer { initial_access, .. } => {
                    let access = resolve_access(initial_access);
                    let last_access = resting_access!(access);
                    buffer_states.insert(value_id, BufferState { last_access, access });
                },

                IR::Acquire { resource, access } => {
                    transition!(*resource, *access, resolve_access(access));
                },

                IR::Clear { attachment, .. } => {
                    let access_id = emit_access!(Access::Clear);
                    transition!(*attachment, access_id, Access::Clear);
                },

                IR::Blit { src, dst, .. } => {
                    let read_id = emit_access!(Access::BlitRead);
                    transition!(*src, read_id, Access::BlitRead);
                    let write_id = emit_access!(Access::BlitWrite);
                    transition!(*dst, write_id, Access::BlitWrite);
                },

                IR::CopyBufferToImage { buffer, image, .. } => {
                    buffer_barrier!(self.resource_root(*buffer), Access::CopyRead);
                    let write_id = emit_access!(Access::CopyWrite);
                    transition!(*image, write_id, Access::CopyWrite);
                },

                IR::BeginRendering {
                    color_attachments,
                    depth_attachment,
                    declared_access,
                    ..
                } => {
                    for (image, access) in region_images.get(&value_id).cloned().into_iter().flatten() {
                        let sampled_id = read_access!(access);
                        transition!(image, sampled_id, access);
                    }

                    for (resource, access_id) in declared_access {
                        let access = resolve_access(access_id);
                        if self.is_buffer(*resource) {
                            buffer_barrier!(self.resource_root(*resource), access);
                            continue;
                        }

                        let access_id = match access.writes() {
                            true => emit_access!(access),
                            false => read_access!(access),
                        };
                        transition!(*resource, access_id, access);
                    }

                    let access_id = emit_access!(Access::ColorRW);
                    for attachment in color_attachments {
                        transition!(*attachment, access_id, Access::ColorRW);
                    }

                    if depth_attachment.is_valid() {
                        let depth_id = emit_access!(Access::DepthStencilRW);
                        transition!(*depth_attachment, depth_id, Access::DepthStencilRW);
                    }

                    for (buffer, access) in region_buffers.get(&value_id).cloned().into_iter().flatten() {
                        buffer_barrier!(buffer, access);
                    }
                },

                IR::BeginCompute { declared_access, .. } => {
                    for (image, access) in region_images.get(&value_id).cloned().into_iter().flatten() {
                        let sampled_id = read_access!(access);
                        transition!(image, sampled_id, access);
                    }

                    for (resource, access_id) in declared_access {
                        let access = resolve_access(access_id);
                        if self.is_buffer(*resource) {
                            buffer_barrier!(self.resource_root(*resource), access);
                            continue;
                        }

                        let access_id = match access.writes() {
                            true => emit_access!(access),
                            false => read_access!(access),
                        };
                        transition!(*resource, access_id, access);
                    }
                },

                IR::Dispatch {
                    size: ir::DispatchSize::Indirect { buffer, .. },
                    ..
                } => {
                    buffer_barrier!(self.resource_root(*buffer), Access::IndirectRead);
                },

                IR::Release { resource, access, .. } => {
                    transition!(*resource, *access, resolve_access(access));
                },

                _ => {},
            }

            result.push((value_id, ir));
        }

        result
    }

    fn is_buffer(&self, id: ValueId) -> bool {
        matches!(
            self.instructions.get(self.resource_root(id).0 as usize),
            Some(IR::ConstructBuffer { .. })
        )
    }

    /// The roots an instruction takes a barrier on, which is the whole of what reads the
    /// state a join records.
    fn resource_uses(&self, ir: &IR, uses: &mut Vec<ValueId>) {
        ir.visit_resource_side_effects(|effect| uses.push(self.resource_root(effect.resource)));
    }

    fn resource_root(&self, id: ValueId) -> ValueId {
        let mut id = id;

        for _ in 0..ir::MAX_RESOLVE_DEPTH {
            let Some(ir) = self.instructions.get(id.0 as usize) else {
                return id;
            };

            match ir::underlying_object(ir) {
                ir::UnderlyingObject::Forwards(next) => id = next,
                _ => return id,
            }
        }

        id
    }

    fn infer_usage(&self, nodes: &mut [ir::Instr]) {
        let access_of = |id: &ValueId| {
            nodes
                .iter()
                .find(|(instr_id, _)| instr_id == id)
                .map(|(_, ir)| ir)
                .or_else(|| self.instructions.get(id.0 as usize))
                .and_then(|ir| match ir {
                    IR::Constant(ir::Constant::Access(access)) => Some(*access),
                    _ => None,
                })
        };

        let mut usages: HashMap<ValueId, vk::ImageUsageFlags> = HashMap::new();
        for (_, ir) in nodes.iter() {
            let IR::ImageBarrier { dst_access, value, .. } = ir else {
                continue;
            };
            let Some(access) = access_of(dst_access) else {
                continue;
            };

            *usages.entry(self.resource_root(*value)).or_default() |= vk::ImageUsageFlags::from(access);
        }

        // a buffer's barriers are global, so unlike an image's there is nothing in them naming
        // the buffer: what a transient buffer is for has to be read off the uses themselves
        let mut buffer_usages: HashMap<ValueId, vk::BufferUsageFlags> = HashMap::new();
        let mut used = |buffer: &ValueId, usage: vk::BufferUsageFlags| {
            *buffer_usages.entry(self.resource_root(*buffer)).or_default() |= usage;
        };
        for (_, ir) in nodes.iter() {
            match ir {
                IR::CopyBufferToImage { buffer, .. } => used(buffer, vk::BufferUsageFlags::TRANSFER_SRC),
                IR::BindVertexBuffers { buffers, .. } => buffers
                    .iter()
                    .for_each(|b| used(b, vk::BufferUsageFlags::VERTEX_BUFFER)),
                IR::BindIndexBuffer { buffer, .. } => used(buffer, vk::BufferUsageFlags::INDEX_BUFFER),
                IR::BeginRendering { declared_access, .. } | IR::BeginCompute { declared_access, .. } => {
                    for (resource, access) in declared_access {
                        if self.is_buffer(*resource)
                            && let Some(access) = access_of(access)
                        {
                            used(resource, vk::BufferUsageFlags::from(access));
                        }
                    }
                },
                IR::Dispatch {
                    size: ir::DispatchSize::Indirect { buffer, .. },
                    ..
                } => used(buffer, vk::BufferUsageFlags::INDIRECT_BUFFER),
                // a pushed address is the only thing that says a shader reaches the buffer
                // through a pointer rather than through a binding. the state changes have not
                // been folded into the draws and dispatches yet, so this reads them where they
                // still are
                IR::SetState {
                    change: StateChange::PushConstantAddress { buffer, .. },
                    ..
                } => used(buffer, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                _ => {},
            }
        }

        // a bound resource is not created here, so what the graph works out it is used for is
        // recorded rather than acted on: it is what the caller has to have allocated it for
        for (value_id, ir) in nodes.iter_mut() {
            match ir {
                IR::ConstructImage { image, usage, .. } if image.is_null() => {
                    *usage |= usages.get(value_id).copied().unwrap_or_default();
                    if usage.is_empty() {
                        tracing::warn!(%value_id, "transient image is never used; it will be created with no usage");
                    }
                },
                IR::ConstructBuffer { buffer, usage, .. } if buffer.is_null() => {
                    *usage |= buffer_usages.get(value_id).copied().unwrap_or_default();
                    if usage.is_empty() {
                        tracing::warn!(%value_id, "transient buffer is never used; it will be created with no usage");
                    }
                },
                _ => {},
            }
        }
    }

    fn resolve_resource(&self, id: ValueId) -> Option<&IR> {
        let mut id = id;

        for _ in 0..ir::MAX_RESOLVE_DEPTH {
            let ir = self.instructions.get(id.0 as usize)?;
            id = match ir::underlying_object(ir) {
                ir::UnderlyingObject::Base => return Some(ir),
                ir::UnderlyingObject::Forwards(next) | ir::UnderlyingObject::Element(next) => next,
                ir::UnderlyingObject::None => return None,
            };
        }

        None
    }

    fn variable_bytes_size(&self, id: ValueId) -> Option<u32> {
        match self.instructions.get(id.0 as usize) {
            Some(IR::Variable {
                kind: ir::VariableKind::Bytes(size),
                ..
            }) => Some(*size),
            _ => None,
        }
    }

    fn is_image(&self, id: ValueId) -> bool {
        matches!(
            self.resolve_resource(id),
            Some(IR::ConstructImage { .. } | IR::SwapchainImage { .. })
        )
    }

    fn resolve_image(&self, id: ValueId) -> Option<ImageType> {
        match self.resolve_resource(id)? {
            IR::ConstructImage {
                format,
                samples,
                extent,
                ..
            } => Some(ImageType {
                format: *format,
                samples: *samples,
                extent: match self.instructions.get(extent.0 as usize) {
                    Some(IR::Constant(ir::Constant::Extent3D(extent))) => Some(*extent),
                    _ => None,
                },
            }),
            IR::SwapchainImage { format, samples, .. } => Some(ImageType {
                format: *format,
                samples: *samples,
                extent: None,
            }),
            _ => None,
        }
    }

    fn resolve_buffer_size(&self, id: ValueId) -> Option<u64> {
        match self.resolve_resource(id)? {
            // a transient buffer has no handle to ask, and one sized by a variable or bound to
            // one has no size to read here at all
            IR::ConstructBuffer { buffer, size, .. } if buffer.is_null() => {
                match self.instructions.get(size.0 as usize) {
                    Some(IR::Constant(ir::Constant::U32(size))) => Some(*size as u64),
                    _ => None,
                }
            },
            IR::ConstructBuffer { buffer, .. } => Some(buffer.size()),
            _ => None,
        }
    }

    fn resolve_extent_2d(&self, id: ValueId) -> Option<vk::Extent2D> {
        match self.instructions.get(id.0 as usize)? {
            IR::Constant(ir::Constant::Extent2D(extent)) => Some(*extent),
            _ => None,
        }
    }

    fn infer(&self, mut nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        #[derive(Clone)]
        struct InForce {
            state: PassState,
            area: vk::Rect2D,
            pipeline: Option<PipelineId>,
            late_area: bool,
        }

        let mut regions: HashMap<ValueId, InForce> = HashMap::new();

        for (value_id, ir) in nodes.iter_mut() {
            match ir {
                IR::BeginRendering {
                    color_attachments,
                    depth_attachment,
                    render_area,
                    ..
                } => {
                    let mut rendering = RenderingState::default();
                    let mut attachment_extent = None;
                    let mut samples_from_color = false;

                    for (index, attachment) in color_attachments.iter().enumerate() {
                        let Some(image) = self.resolve_image(*attachment) else {
                            tracing::warn!(%attachment, "cannot infer attachment type; pipelines may not match");
                            continue;
                        };

                        if index == 0 {
                            rendering.samples = image.samples;
                            samples_from_color = true;
                            attachment_extent = image.extent.map(|extent| vk::Extent2D {
                                width: extent.width,
                                height: extent.height,
                            });
                        } else if image.samples != rendering.samples {
                            tracing::warn!(
                                %attachment,
                                "attachment sample count differs from the region's first attachment"
                            );
                        }

                        rendering.color_formats.push(image.format);
                    }

                    if depth_attachment.is_valid() {
                        match self.resolve_image(*depth_attachment) {
                            Some(image) => {
                                rendering.depth_format = Some(image.format);

                                // a depth-only region has nothing else to take these from
                                if !samples_from_color {
                                    rendering.samples = image.samples;
                                    attachment_extent = image.extent.map(|extent| vk::Extent2D {
                                        width: extent.width,
                                        height: extent.height,
                                    });
                                } else if image.samples != rendering.samples {
                                    tracing::warn!(
                                        attachment = %depth_attachment,
                                        "depth attachment sample count differs from the color attachments"
                                    );
                                }
                            },
                            None => {
                                tracing::warn!(attachment = %depth_attachment, "cannot infer depth attachment type; pipelines may not match")
                            },
                        }
                    }

                    // framebuffer-relative state resolves against this, so it has to match the
                    // area the region is opened with
                    let extent = match render_area.is_valid() {
                        true => self.resolve_extent_2d(*render_area),
                        false => attachment_extent,
                    };
                    regions.insert(
                        *value_id,
                        InForce {
                            state: PassState::for_rendering(rendering),
                            area: vk::Rect2D::default().extent(extent.unwrap_or_default()),
                            pipeline: None,
                            late_area: extent.is_none(),
                        },
                    );
                },

                IR::BeginCompute { .. } => {
                    regions.insert(
                        *value_id,
                        InForce {
                            state: PassState::default(),
                            area: vk::Rect2D::default(),
                            pipeline: None,
                            late_area: false,
                        },
                    );
                },

                IR::BindPipeline { pass, pipeline, .. } => {
                    if let Some(mut in_force) = regions.get(pass).cloned() {
                        in_force.pipeline = Some(*pipeline);
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::SetState { pass, change } => {
                    if let Some(mut in_force) = regions.get(pass).cloned() {
                        in_force.state.apply(change.clone());
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::Draw {
                    pass,
                    pipeline,
                    state,
                    dynamic,
                    ..
                }
                | IR::DrawIndexed {
                    pass,
                    pipeline,
                    state,
                    dynamic,
                    ..
                }
                | IR::CallOpaque {
                    pass,
                    pipeline,
                    state,
                    dynamic,
                    ..
                } => {
                    if let Some(in_force) = regions.get(pass).cloned() {
                        if in_force.pipeline.is_none() {
                            tracing::warn!(%value_id, "draw with no pipeline bound");
                        }
                        *pipeline = in_force.pipeline.unwrap_or(PipelineId::INVALID);

                        // an area that is only known at execute time cannot be baked into a
                        // pipeline, so anything framebuffer-relative has to be recorded instead
                        let mut resolved = in_force.state.clone();
                        if in_force.late_area {
                            resolved.dynamic |= DynamicStateFlags::Viewport | DynamicStateFlags::Scissor;
                        }
                        (*state, *dynamic) = resolved.resolve(in_force.area);
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::Dispatch {
                    pass,
                    pipeline,
                    push_constants,
                    ..
                } => {
                    if let Some(in_force) = regions.get(pass).cloned() {
                        if in_force.pipeline.is_none() {
                            tracing::warn!(%value_id, "dispatch with no pipeline bound");
                        }
                        *pipeline = in_force.pipeline.unwrap_or(PipelineId::INVALID);
                        *push_constants = in_force.state.push_constants.clone();
                        regions.insert(*value_id, in_force);
                    }
                },

                IR::BindVertexBuffers { pass, .. }
                | IR::BindIndexBuffer { pass, .. }
                | IR::WriteDescriptor { pass, .. }
                | IR::EndRendering { pass }
                | IR::EndCompute { pass } => {
                    if let Some(in_force) = regions.get(pass).cloned() {
                        regions.insert(*value_id, in_force);
                    }
                },

                _ => {},
            }
        }

        nodes
    }
}

pub trait Count {
    fn lower(self, module: &mut Module) -> ValueId;
}

impl Count for u32 {
    fn lower(self, module: &mut Module) -> ValueId { module.lower_u32(self) }
}

impl Count for ValueId {
    fn lower(self, _: &mut Module) -> ValueId { self }
}

impl Module {
    fn open_pass(&mut self, kind: PassKind, begin: ValueId) -> &mut Self {
        self.expect_no_open_pass("another one opens");
        self.ongoing_pass = Some(OngoingPass {
            kind,
            begin,
            tip: begin,
        });
        self
    }

    fn close_pass(&mut self) -> Option<OngoingPass> { self.ongoing_pass.take() }

    fn expect_no_open_pass(&mut self, when: &str) {
        if let Some(open) = self.close_pass() {
            tracing::error!(
                begin = %open.begin,
                "a {} pass is still open where {when}; it is closed here",
                open.kind.name()
            );
        }
    }

    fn chain(&mut self, wanted: Option<PassKind>, what: &str, ir: impl FnOnce(ValueId) -> IR) -> &mut Self {
        let Some(open) = self.ongoing_pass else {
            tracing::error!(what, "recorded outside of a pass; it is dropped");
            return self;
        };

        if let Some(wanted) = wanted
            && open.kind != wanted
        {
            tracing::error!(
                what,
                "belongs to a {} pass but a {} pass is open; it is dropped",
                wanted.name(),
                open.kind.name()
            );
            return self;
        }

        let id = self.emit(ir(open.tip));
        if let Some(open) = self.ongoing_pass.as_mut() {
            open.tip = id;
        }
        self
    }

    fn set_state(&mut self, change: StateChange) -> &mut Self {
        self.chain(Some(PassKind::Rendering), "state", |pass| IR::SetState { pass, change })
    }

    pub fn with_name(&mut self, name: impl Into<Arc<str>>) -> &mut Self {
        match self.ongoing_pass {
            Some(open) => {
                self.set_name(open.begin, name);
            },
            None => tracing::error!("no pass is open to name; the name is dropped"),
        }
        self
    }

    pub fn push_constants<T: Copy>(&mut self, value: &T) -> &mut Self { self.push_constants_at(0, value) }

    pub fn push_constants_at<T: Copy>(&mut self, offset: u32, value: &T) -> &mut Self {
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.push_constant_bytes(offset, bytes)
    }

    pub fn push_constant_bytes(&mut self, offset: u32, data: &[u8]) -> &mut Self {
        self.pass_state(StateChange::PushConstants {
            offset,
            data: data.to_vec(),
        })
    }

    pub fn push_constants_from(&mut self, variable: ValueId) -> &mut Self { self.push_constants_from_at(0, variable) }

    pub fn push_constants_from_at(&mut self, offset: u32, variable: ValueId) -> &mut Self {
        let Some(size) = self.variable_bytes_size(variable) else {
            tracing::error!(%variable, "push constants can only come from a bytes variable");
            return self;
        };

        self.pass_state(StateChange::PushConstantsFrom {
            offset,
            size,
            source: variable,
        })
    }

    pub fn push_constant_address(&mut self, offset: u32, buffer: ValueId) -> &mut Self {
        self.pass_state(StateChange::PushConstantAddress { offset, buffer })
    }

    /// Push constants are the one piece of state both kinds of pass carry, so they are chained
    /// without asking which one is open.
    fn pass_state(&mut self, change: StateChange) -> &mut Self {
        self.chain(None, "push constants", |pass| IR::SetState { pass, change })
    }

    fn bind_pipeline(&mut self, kind: PassKind, pipeline: PipelineId, bind_point: vk::PipelineBindPoint) -> &mut Self {
        self.chain(Some(kind), "pipeline", |pass| IR::BindPipeline {
            pass,
            pipeline,
            bind_point,
        })
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: PipelineId) -> &mut Self {
        self.bind_pipeline(PassKind::Rendering, pipeline, vk::PipelineBindPoint::GRAPHICS)
    }

    pub fn bind_compute_pipeline(&mut self, pipeline: PipelineId) -> &mut Self {
        self.bind_pipeline(PassKind::Compute, pipeline, vk::PipelineBindPoint::COMPUTE)
    }

    /// Writes one descriptor into the open pass. Every write still standing when a draw or
    /// dispatch is reached is what that command's descriptor sets are built from, so rebinding
    /// the same set and binding only affects the commands after it.
    fn write_descriptor(&mut self, set: u32, binding: u32, descriptor: ir::Descriptor) -> &mut Self {
        if !self.is_image(descriptor.image()) {
            tracing::error!(
                image = %descriptor.image(),
                set,
                binding,
                "only an image value can be written to an image descriptor"
            );
            return self;
        }

        let access = self.lower_access(Access::None);
        self.chain(None, "descriptor", |pass| IR::WriteDescriptor {
            pass,
            set,
            binding,
            descriptor,
            access,
        })
    }

    pub fn bind_image(&mut self, set: u32, binding: u32, image: ValueId) -> &mut Self {
        self.write_descriptor(set, binding, ir::Descriptor::SampledImage { image })
    }

    pub fn bind_texture(&mut self, set: u32, binding: u32, image: ValueId, sampler: vk::Sampler) -> &mut Self {
        self.write_descriptor(set, binding, ir::Descriptor::CombinedImageSampler { image, sampler })
    }

    pub fn bind_vertex_buffer(&mut self, binding: u32, buffer: ValueId) -> &mut Self {
        self.bind_vertex_buffers(binding, &[buffer], &[0])
    }

    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers: &[ValueId], offsets: &[u64]) -> &mut Self {
        assert_eq!(
            buffers.len(),
            offsets.len(),
            "every bound vertex buffer needs an offset"
        );

        let buffers = buffers.to_vec();
        let offsets = offsets.to_vec();
        self.chain(Some(PassKind::Rendering), "vertex buffers", |pass| {
            IR::BindVertexBuffers {
                pass,
                first_binding,
                buffers,
                offsets,
            }
        })
    }

    pub fn bind_index_buffer(&mut self, buffer: ValueId, index_type: vk::IndexType) -> &mut Self {
        self.bind_index_buffer_at(buffer, 0, index_type)
    }

    pub fn bind_index_buffer_at(&mut self, buffer: ValueId, offset: u64, index_type: vk::IndexType) -> &mut Self {
        self.chain(Some(PassKind::Rendering), "index buffer", |pass| IR::BindIndexBuffer {
            pass,
            buffer,
            offset,
            index_type,
        })
    }

    pub fn set_primitive_topology(&mut self, topology: vk::PrimitiveTopology) -> &mut Self {
        self.set_state(StateChange::PrimitiveTopology(topology))
    }

    pub fn set_depth(&mut self, depth: DepthState) -> &mut Self { self.set_state(StateChange::Depth(depth)) }

    pub fn set_rasterization(&mut self, rasterization: RasterizationState) -> &mut Self {
        self.set_state(StateChange::Rasterization(rasterization))
    }

    pub fn set_dynamic_state(&mut self, dynamic: DynamicStateFlags) -> &mut Self {
        self.set_state(StateChange::DynamicState(dynamic))
    }

    pub fn set_viewport(&mut self, index: u32, viewport: impl Into<Viewport>) -> &mut Self {
        self.set_state(StateChange::Viewport {
            index,
            viewport: viewport.into(),
        })
    }

    pub fn set_scissor(&mut self, index: u32, rect: Rect2D) -> &mut Self {
        self.set_state(StateChange::Scissor { index, rect })
    }

    pub fn set_color_blend(&mut self, index: u32, blend: impl Into<ColorBlendAttachmentState>) -> &mut Self {
        self.set_state(StateChange::ColorBlend {
            index: Some(index),
            blend: blend.into(),
        })
    }

    pub fn broadcast_color_blend(&mut self, blend: impl Into<ColorBlendAttachmentState>) -> &mut Self {
        self.set_state(StateChange::ColorBlend {
            index: None,
            blend: blend.into(),
        })
    }

    pub fn draw(&mut self, vertex_count: impl Count, instance_count: impl Count) -> &mut Self {
        self.draw_range(vertex_count, instance_count, 0, 0)
    }

    pub fn draw_range(
        &mut self, vertex_count: impl Count, instance_count: impl Count, first_vertex: impl Count,
        first_instance: impl Count,
    ) -> &mut Self {
        let vertex_count = vertex_count.lower(self);
        let instance_count = instance_count.lower(self);
        let first_vertex = first_vertex.lower(self);
        let first_instance = first_instance.lower(self);
        self.chain(Some(PassKind::Rendering), "draw", |pass| IR::Draw {
            pass,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            pipeline: PipelineId::INVALID,
            state: Default::default(),
            dynamic: Default::default(),
        })
    }

    pub fn draw_indexed(&mut self, index_count: impl Count, instance_count: impl Count) -> &mut Self {
        self.draw_indexed_range(index_count, instance_count, 0, 0, 0)
    }

    pub fn draw_indexed_range(
        &mut self, index_count: impl Count, instance_count: impl Count, first_index: impl Count, vertex_offset: i32,
        first_instance: impl Count,
    ) -> &mut Self {
        let index_count = index_count.lower(self);
        let instance_count = instance_count.lower(self);
        let first_index = first_index.lower(self);
        let vertex_offset = self.lower_i32(vertex_offset);
        let first_instance = first_instance.lower(self);
        self.chain(Some(PassKind::Rendering), "indexed draw", |pass| IR::DrawIndexed {
            pass,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            pipeline: PipelineId::INVALID,
            state: Default::default(),
            dynamic: Default::default(),
        })
    }

    pub fn record_from(&mut self, body: ValueId) -> &mut Self {
        self.chain(Some(PassKind::Rendering), "callback", |pass| IR::CallOpaque {
            pass,
            body,
            pipeline: PipelineId::INVALID,
            state: Default::default(),
            dynamic: Default::default(),
        })
    }

    pub fn end_rendering(&mut self) -> ValueId { self.end_pass(PassKind::Rendering, |pass| IR::EndRendering { pass }) }

    pub fn access(&mut self, resource: ValueId, access: Access) -> &mut Self {
        let Some(open) = self.ongoing_pass else {
            tracing::error!(%resource, "an access was declared outside of a pass; it is dropped");
            return self;
        };

        let existing = match self.instructions.get(open.begin.0 as usize) {
            Some(IR::BeginRendering { declared_access, .. } | IR::BeginCompute { declared_access, .. }) => {
                declared_access
                    .iter()
                    .find_map(|(id, access)| (*id == resource).then_some(*access))
            },
            _ => return self,
        };
        let merged = existing.map_or(access, |existing| self.resolve_access(existing) | access);
        let access = self.lower_access(merged);

        let declared = match self.instructions.get_mut(open.begin.0 as usize) {
            Some(IR::BeginRendering { declared_access, .. } | IR::BeginCompute { declared_access, .. }) => {
                declared_access
            },
            _ => return self,
        };

        match declared.iter_mut().find(|(id, _)| *id == resource) {
            Some((_, slot)) => *slot = access,
            None => declared.push((resource, access)),
        }

        self
    }

    /// [`Self::access`] for the compute accesses, which only a compute pass reaches anything by.
    fn compute_access(&mut self, resource: ValueId, access: Access) -> &mut Self {
        if self.ongoing_pass.is_some_and(|open| open.kind != PassKind::Compute) {
            tracing::error!(%resource, "a compute access was declared in a rendering pass; it is dropped");
            return self;
        }

        self.access(resource, access)
    }

    pub fn read(&mut self, resource: ValueId) -> &mut Self { self.compute_access(resource, Access::ComputeRead) }

    pub fn write(&mut self, resource: ValueId) -> &mut Self { self.compute_access(resource, Access::ComputeWrite) }

    pub fn read_write(&mut self, resource: ValueId) -> &mut Self { self.compute_access(resource, Access::ComputeRW) }

    pub fn sample_image(&mut self, image: ValueId) -> &mut Self { self.compute_access(image, Access::ComputeSampled) }

    pub fn read_uniform(&mut self, buffer: ValueId) -> &mut Self {
        self.compute_access(buffer, Access::ComputeUniformRead)
    }

    fn dispatch_size(&mut self, size: ir::DispatchSize) -> &mut Self {
        self.chain(Some(PassKind::Compute), "dispatch", |pass| IR::Dispatch {
            pass,
            size,
            pipeline: PipelineId::INVALID,
            push_constants: Default::default(),
        })
    }

    /// Dispatches these group counts. Each axis is a count known now or a value holding one when
    /// the module runs, in any mix.
    pub fn dispatch(&mut self, groups_x: impl Count, groups_y: impl Count, groups_z: impl Count) -> &mut Self {
        let x = groups_x.lower(self);
        let y = groups_y.lower(self);
        let z = groups_z.lower(self);
        self.dispatch_size(ir::DispatchSize::Groups { x, y, z })
    }

    /// Dispatches at least this many invocations per axis, rounded up to whole workgroups of
    /// whatever the bound pipeline declares. An axis given as a value is counted and rounded up
    /// when the module runs rather than now.
    pub fn dispatch_invocations(
        &mut self, invocations_x: impl Count, invocations_y: impl Count, invocations_z: impl Count,
    ) -> &mut Self {
        let x = invocations_x.lower(self);
        let y = invocations_y.lower(self);
        let z = invocations_z.lower(self);
        self.dispatch_size(ir::DispatchSize::Invocations { x, y, z })
    }

    /// Dispatches one invocation per pixel of `image`, width along x, height along y and depth
    /// along z.
    pub fn dispatch_invocations_per_pixel(&mut self, image: ValueId) -> &mut Self {
        self.dispatch_invocations_per_pixel_scaled(image, [1.0; 3])
    }

    /// Dispatches [`Self::dispatch_invocations_per_pixel`] scaled per axis: above one is more
    /// invocations than pixels, below one is fewer. Rounding up to whole workgroups happens after
    /// the scaling.
    pub fn dispatch_invocations_per_pixel_scaled(&mut self, image: ValueId, scale: [f32; 3]) -> &mut Self {
        // an image sized by a variable has no extent to read here, so the count is left for
        // the interpreter to work out from whatever the image was allocated at
        let Some(extent) = self.resolve_image(image).and_then(|image| image.extent) else {
            if !self.is_image(image) {
                tracing::warn!(%image, "cannot infer the extent to dispatch over; the dispatch is dropped");
                return self;
            }

            return self.dispatch_size(ir::DispatchSize::per_pixel(image, scale));
        };

        self.dispatch_invocations(
            scaled(extent.width as u64, scale[0]),
            scaled(extent.height as u64, scale[1]),
            scaled(extent.depth as u64, scale[2]),
        )
    }

    /// Dispatches one invocation per `element_size` bytes of `buffer`, along the x-axis only.
    pub fn dispatch_invocations_per_element(&mut self, buffer: ValueId, element_size: u64) -> &mut Self {
        self.dispatch_invocations_per_element_scaled(buffer, element_size, 1.0)
    }

    /// Dispatches [`Self::dispatch_invocations_per_element`] scaled: above one is more invocations
    /// than elements, below one is fewer.
    pub fn dispatch_invocations_per_element_scaled(
        &mut self, buffer: ValueId, element_size: u64, scale: f32,
    ) -> &mut Self {
        if element_size == 0 {
            tracing::warn!(%buffer, "elements of no size have no count to dispatch over; the dispatch is dropped");
            return self;
        }

        // a buffer bound to a variable has no size to read here, so the count is left for the
        // interpreter to work out from whatever was bound, the same way a per-pixel dispatch
        // over an image sized at run time is
        let Some(size) = self.resolve_buffer_size(buffer) else {
            if !self.is_buffer(buffer) {
                tracing::warn!(%buffer, "cannot infer the element count to dispatch over; the dispatch is dropped");
                return self;
            }

            return self.dispatch_size(ir::DispatchSize::per_element(buffer, element_size, scale));
        };

        self.dispatch_invocations(scaled(size / element_size, scale), 1, 1)
    }

    /// Dispatches the group counts the device reads out of `buffer` itself.
    pub fn dispatch_indirect(&mut self, buffer: ValueId) -> &mut Self { self.dispatch_indirect_at(buffer, 0) }

    pub fn dispatch_indirect_at(&mut self, buffer: ValueId, offset: u64) -> &mut Self {
        self.dispatch_size(ir::DispatchSize::Indirect { buffer, offset })
    }

    pub fn end_compute(&mut self) -> ValueId { self.end_pass(PassKind::Compute, |pass| IR::EndCompute { pass }) }

    fn end_pass(&mut self, kind: PassKind, end: impl FnOnce(ValueId) -> IR) -> ValueId {
        let Some(open) = self.ongoing_pass.filter(|open| open.kind == kind) else {
            tracing::error!("no {} pass is open to close", kind.name());
            return self.ongoing_pass.map_or(ValueId(0), |open| open.tip);
        };

        self.close_pass();
        self.emit(end(open.tip))
    }
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;
    use crate::{BlendPreset, Image, PipelineState, ResolvedViewport, Unchecked, clear, graph::analysis::Declared};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
    const LAYOUT: vk::ImageLayout = vk::ImageLayout::READ_ONLY_OPTIMAL;
    const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;

    fn module_with_attachment() -> (Module, ValueId) {
        let mut module = Module::default();
        let attachment = ImageAttachment::new(
            Image::imported(
                vk::Image::null(),
                FORMAT,
                vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
                vk::SampleCountFlags::TYPE_1,
            ),
            FORMAT,
            vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            vk::SampleCountFlags::TYPE_1,
            vk::ImageLayout::UNDEFINED,
        );
        let (_, construct) = module.lower_image_attachment(&attachment, Some("target".into()));
        (module, construct)
    }

    fn depth_info() -> ImageInfo { ImageInfo::depth_target(extent_2d(), DEPTH_FORMAT) }

    fn access_constant(module: &Module, program: &Program, id: &ValueId) -> Access {
        program
            .instructions()
            .iter()
            .find(|(instr_id, _)| instr_id == id)
            .map(|(_, ir)| ir)
            .or_else(|| module.instructions.get(id.0 as usize))
            .and_then(|ir| match ir {
                IR::Constant(ir::Constant::Access(a)) => Some(*a),
                _ => None,
            })
            .expect("barrier operand should be an access constant")
    }

    fn memory_barriers(module: &Module, program: &Program) -> Vec<(Access, Access)> {
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::MemoryBarrier { src_access, dst_access } => Some((
                    access_constant(module, program, src_access),
                    access_constant(module, program, dst_access),
                )),
                _ => None,
            })
            .collect()
    }

    struct ImageBarrier {
        src: Access,
        dst: Access,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        resource: ValueId,
    }

    fn image_barriers(module: &Module, program: &Program) -> Vec<ImageBarrier> {
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::ImageBarrier {
                    src_access,
                    dst_access,
                    old_layout,
                    new_layout,
                    value,
                } => Some(ImageBarrier {
                    src: access_constant(module, program, src_access),
                    dst: access_constant(module, program, dst_access),
                    old_layout: *old_layout,
                    new_layout: *new_layout,
                    resource: module.resource_root(*value),
                }),
                _ => None,
            })
            .collect()
    }

    fn extent_2d() -> vk::Extent2D {
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        }
    }

    fn transient_info() -> ImageInfo { ImageInfo::color_target(extent_2d(), FORMAT) }

    /// An image that declares no usage at all, so what it ends up created with is entirely
    /// what the graph inferred.
    fn untyped_info() -> ImageInfo {
        ImageInfo {
            extent: vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            format: FORMAT,
            ..Default::default()
        }
    }

    /// A sampler handle that stands for one, since a combined image sampler is only well formed
    /// with a real one behind it.
    fn a_sampler() -> vk::Sampler { vk::Sampler::from_raw(1) }

    fn image_usage(program: &Program, id: ValueId) -> vk::ImageUsageFlags {
        program
            .instructions()
            .iter()
            .find_map(|(instr_id, ir)| match ir {
                IR::ConstructImage { usage, .. } if *instr_id == id => Some(*usage),
                _ => None,
            })
            .expect("the image should still be declared after compiling")
    }

    fn draws(program: &Program) -> Vec<(PipelineId, PipelineState)> {
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { pipeline, state, .. } | IR::DrawIndexed { pipeline, state, .. } => {
                    Some((*pipeline, state.clone()))
                },
                _ => None,
            })
            .collect()
    }

    fn u32_constant(module: &Module, id: &ValueId) -> u32 {
        match module.instructions.get(id.0 as usize) {
            Some(IR::Constant(ir::Constant::U32(v))) => *v,
            _ => panic!("{id} is not a u32 constant"),
        }
    }

    fn i32_constant(module: &Module, id: &ValueId) -> i32 {
        match module.instructions.get(id.0 as usize) {
            Some(IR::Constant(ir::Constant::I32(v))) => *v,
            _ => panic!("{id} is not an i32 constant"),
        }
    }

    /// The state in force where the body is called is what its pipeline is compiled with, the
    /// same as a draw's, since the callback records into a region that is already open and
    /// cannot change any of it.
    #[test]
    fn an_opaque_body_is_compiled_with_the_state_in_force_where_it_is_called() {
        let mut module = Module::default();
        let target = module.transient_image(&transient_info());
        let body = module.declare_callback_var("draws");

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(3))
            .set_primitive_topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .broadcast_color_blend(BlendPreset::AlphaBlend)
            .record_from(body)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let calls = compiled
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::CallOpaque { pipeline, state, .. } => Some((*pipeline, state.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(calls.len(), 1, "{}", compiled.dump());
        assert_eq!(calls[0].0, PipelineId(3));
        assert_eq!(calls[0].1.topology, vk::PrimitiveTopology::TRIANGLE_STRIP);
        assert_eq!(calls[0].1.rendering.color_formats, vec![FORMAT]);
    }

    /// A body records commands and nothing else, so what the region reads has to be declared
    /// around it. The barrier that puts a sampled image in the layout the body reads it in is
    /// the whole reason an opaque pass can be trusted.
    #[test]
    fn a_region_with_an_opaque_body_still_barriers_what_it_declared() {
        let (mut module, swapchain) = module_with_attachment();
        let sampled = module.transient_image(&transient_info());
        let body = module.declare_callback_var("draws");

        let cleared = module.clear(sampled, clear::f32::BLACK);
        let drawn = module
            .begin_rendering(&[swapchain])
            .bind_graphics_pipeline(PipelineId(0))
            .access(cleared, Access::FragmentSampled)
            .record_from(body)
            .end_rendering();
        let end = module.present(drawn);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let transitions = image_barriers(&module, &compiled)
            .into_iter()
            .filter(|barrier| module.resource_root(barrier.resource) == sampled)
            .map(|barrier| (barrier.old_layout, barrier.new_layout, barrier.dst))
            .collect::<Vec<_>>();

        assert_eq!(
            transitions,
            vec![
                (
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    Access::Clear
                ),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    Access::FragmentSampled
                ),
            ],
            "{}",
            compiled.dump()
        );
    }

    /// The body is a variable, which is what makes the compiled program worth keeping: binding
    /// a different one is a write into it rather than a reason to compile it again.
    #[test]
    fn the_body_of_an_opaque_pass_is_a_variable_of_the_program() {
        let mut module = Module::default();
        let target = module.transient_image(&transient_info());
        let body = module.declare_callback_var("draws");

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .record_from(body)
            .end_rendering();

        let mut compiled = module.compile(&Unchecked, end).unwrap();
        assert!(
            compiled
                .instructions()
                .iter()
                .any(|(id, ir)| *id == body && matches!(ir, IR::Variable { .. })),
            "the callback variable should survive into the program\n{}",
            compiled.dump()
        );

        let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = called.clone();
        compiled.set(
            body,
            PassCallback::new(move |_| flag.store(true, std::sync::atomic::Ordering::Relaxed)),
        );
        assert!(
            !called.load(std::sync::atomic::Ordering::Relaxed),
            "binding is not calling"
        );
    }

    #[test]
    fn a_transient_image_is_created_with_the_usage_the_graph_implies() {
        let (mut module, swapchain) = module_with_attachment();
        let target = module.transient_image(&transient_info());

        let rendered = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();
        let swapchain = module.blit(rendered, swapchain);
        let end = module.present(swapchain);

        assert_eq!(
            image_usage(&module.compile(&Unchecked, end).unwrap(), target),
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC
        );
    }

    #[test]
    fn a_blit_puts_each_side_in_its_own_transfer_layout() {
        let (mut module, destination) = module_with_attachment();
        let source = module.transient_image(&transient_info());
        let end = module.blit(source, destination);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.len(), 2);

        assert_eq!(barriers[0].resource, source);
        assert_eq!(barriers[0].src, Access::None);
        assert_eq!(barriers[0].dst, Access::BlitRead);
        assert_eq!(barriers[0].old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barriers[0].new_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        assert_eq!(barriers[1].resource, destination);
        assert_eq!(barriers[1].dst, Access::BlitWrite);
        assert_eq!(barriers[1].new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    #[test]
    fn an_imported_image_starts_in_the_layout_its_owner_left_it_in() {
        let mut module = Module::default();
        let image = Image::imported(
            vk::Image::null(),
            FORMAT,
            vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            vk::SampleCountFlags::TYPE_1,
        );

        let imported = module.import_image(&image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let end = module.clear(imported, crate::clear::f32::BLACK);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].old_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(barriers[0].new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    #[test]
    fn a_second_read_at_the_same_layout_does_not_repeat_the_barrier() {
        let (mut module, first) = module_with_attachment();
        let source = module.transient_image(&transient_info());

        // the second blit writes what the first one produced, so both are in the graph and the
        // source is read twice without ever leaving TRANSFER_SRC_OPTIMAL
        let first = module.blit(source, first);
        let end = module.blit(source, first);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let barriers = image_barriers(&module, &compiled);
        let reads = barriers.iter().filter(|b| b.resource == source).count();
        assert_eq!(reads, 1);

        // the destination is written twice, and a write always waits
        assert_eq!(barriers.len(), 3);
    }

    #[test]
    fn a_release_to_the_layout_an_image_already_rests_in_costs_nothing() {
        let (mut module, destination) = module_with_attachment();
        let source = module.transient_image(&transient_info());
        let blit = module.blit(source, destination);

        let present = module.present(blit);
        let resting = module.release(source, Access::BlitRead, DomainFlag::Graphics);
        let compiled = module.compile_all(&Unchecked, &[present, resting]).unwrap();

        // nothing consumes the release, so only naming it as a root keeps it
        let position = |id: ValueId| compiled.instructions().iter().position(|(instr_id, _)| *instr_id == id);
        assert!(position(resting) > position(blit));
        assert!(
            module
                .compile(&Unchecked, present)
                .unwrap()
                .instructions()
                .iter()
                .all(|(id, _)| *id != resting)
        );

        // and it asks for the layout the blit already left it in, so it emits no barrier
        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.iter().filter(|b| b.resource == source).count(), 1);
    }

    #[test]
    fn a_release_that_does_change_the_layout_still_emits_its_barrier() {
        let (mut module, destination) = module_with_attachment();
        let source = module.transient_image(&transient_info());
        let blit = module.blit(source, destination);

        let present = module.present(blit);
        let resting = module.release(source, Access::ColorRW, DomainFlag::Graphics);
        let compiled = module.compile_all(&Unchecked, &[present, resting]).unwrap();

        let barriers = image_barriers(&module, &compiled);
        let last = barriers
            .iter()
            .rfind(|b| b.resource == source)
            .expect("the source should have been transitioned");
        assert_eq!(last.old_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(last.new_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    #[test]
    fn binding_a_vertex_buffer_makes_the_host_write_visible_to_vertex_input() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let instructions = compiled.instructions();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::AttributeRead)]
        );

        let binds = instructions
            .iter()
            .filter(|(_, ir)| matches!(ir, IR::BindVertexBuffers { .. }))
            .count();
        assert_eq!(binds, 1);

        let position = |predicate: fn(&IR) -> bool| {
            instructions
                .iter()
                .position(|(_, ir)| predicate(ir))
                .expect("instruction should be present")
        };
        assert!(
            position(|ir| matches!(ir, IR::MemoryBarrier { .. }))
                < position(|ir| matches!(ir, IR::BeginRendering { .. }))
        );

        let draws = draws(&compiled);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].0, PipelineId(0));
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
    }

    /// Nothing declares what a transient buffer is for, so what the graph does with it is what
    /// it has to be created for.
    #[test]
    fn a_transient_buffer_is_created_with_the_usage_the_graph_implies() {
        let (mut module, attachment) = module_with_attachment();
        let scratch = module.transient_buffer(&BufferInfo::new(
            256,
            vk::BufferUsageFlags::empty(),
            MemoryLocation::GpuOnly,
        ));

        let written = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(scratch)
            .push_constant_address(0, scratch)
            .dispatch(1, 1, 1)
            .end_compute();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, written)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let usage = compiled
            .instructions()
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::ConstructBuffer { buffer, usage, .. } if buffer.is_null() => Some(*usage),
                _ => None,
            })
            .expect("the transient should be constructed");

        assert_eq!(
            usage,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::VERTEX_BUFFER
        );
    }

    /// A buffer the graph allocates for this run has no earlier use to wait on, so its first
    /// write is not a barrier the way an imported buffer's would be.
    #[test]
    fn the_first_write_to_a_transient_buffer_waits_on_nothing() {
        let mut module = Module::default();
        let scratch = module.transient_buffer(&BufferInfo::new(
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        ));

        let written = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(scratch)
            .dispatch(1, 1, 1)
            .end_compute();
        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .read(written)
            .dispatch(1, 1, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::ComputeWrite, Access::ComputeRead)]
        );
    }

    /// A transient reached only through a pushed address is still a resource the run has to
    /// allocate, so it has to survive the walk that decides what the program contains.
    #[test]
    fn a_transient_reached_only_by_address_is_still_built() {
        let mut module = Module::default();
        let scratch = module.transient_buffer(&BufferInfo::new(
            256,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        ));

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .push_constant_address(0, scratch)
            .dispatch(1, 1, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let constructed = compiled
            .instructions()
            .iter()
            .any(|(id, ir)| *id == scratch && matches!(ir, IR::ConstructBuffer { .. }));
        assert!(constructed);
    }

    /// A slot names a resource the graph does not own, so what it stands for is a construct
    /// like any other: one identity, which everything reaching the buffer resolves back to.
    #[test]
    fn a_bound_buffer_is_one_resource_the_whole_module_shares() {
        let mut module = Module::default();
        let instances = module.declare_buffer_var("instances", Access::HostWrite);

        let written = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(instances)
            .dispatch(1, 1, 1)
            .end_compute();
        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .read(written)
            .dispatch(1, 1, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(module.resource_root(written), instances);
        assert_eq!(module.resource_root(end), instances);
        assert_eq!(
            compiled
                .instructions()
                .iter()
                .filter(|(_, ir)| matches!(ir, IR::ConstructBuffer { .. }))
                .count(),
            1,
            "{}",
            compiled.dump()
        );
    }

    /// Nothing about a bound buffer is read off a handle, so the ordering the graph works out
    /// is the same as for one it owns. Only the first barrier differs: a bound buffer rests
    /// wherever the caller promised, and a host write has to be made visible.
    #[test]
    fn a_bound_buffer_waits_at_the_access_it_was_declared_to_rest_at() {
        let mut module = Module::default();
        let instances = module.declare_buffer_var("instances", Access::HostWrite);

        let written = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(instances)
            .dispatch(1, 1, 1)
            .end_compute();
        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .read(written)
            .dispatch(1, 1, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![
                (Access::HostWrite, Access::ComputeWrite),
                (Access::ComputeWrite, Access::ComputeRead),
            ]
        );
    }

    /// The caller allocates a bound buffer, so the usage the graph works out is not acted on
    /// here. It is still worked out, since it is what the caller had to have allocated for.
    #[test]
    fn a_bound_buffer_records_the_usage_the_graph_implies() {
        let (mut module, attachment) = module_with_attachment();
        let vertices = module.declare_buffer_var("vertices", Access::HostWrite);

        let written = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(vertices)
            .push_constant_address(0, vertices)
            .dispatch(1, 1, 1)
            .end_compute();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, written)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let usage = compiled
            .instructions()
            .iter()
            .find_map(|(id, ir)| match ir {
                IR::ConstructBuffer { usage, .. } if *id == vertices => Some(*usage),
                _ => None,
            })
            .expect("the bound buffer should be constructed");

        assert_eq!(
            usage,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::VERTEX_BUFFER
        );
    }

    /// A pipeline outlives the run it was compiled in, and a bound image is not measured until
    /// one is bound, so framebuffer-relative state cannot be folded into a draw.
    #[test]
    fn a_bound_image_leaves_the_viewport_to_be_recorded() {
        let mut module = Module::default();
        let target = module.declare_image_var("target", FORMAT, vk::SampleCountFlags::TYPE_1, LAYOUT);

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .set_viewport(0, Rect2D::framebuffer())
            .set_dynamic_state(DynamicStateFlags::None)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 1);
        assert!(draws[0].1.viewports.is_empty(), "{:?}", draws[0].1.viewports);
        assert!(draws[0].1.dynamic.contains(DynamicStateFlags::Viewport));
        assert!(draws[0].1.dynamic.contains(DynamicStateFlags::Scissor));
    }

    /// What a bound image is measured at can wait for the binding, but what it is made of
    /// cannot: the pipelines a region compiles to are built against its format.
    #[test]
    fn a_bound_image_carries_the_format_its_pipelines_are_built_against() {
        let mut module = Module::default();
        let target = module.declare_image_var("target", FORMAT, vk::SampleCountFlags::TYPE_1, LAYOUT);

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let draws = draws(&compiled);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);

        // the layout the caller promised is what the region's first transition leaves
        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].resource, target);
        assert_eq!(barriers[0].old_layout, LAYOUT);
        assert_eq!(barriers[0].new_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    /// Neither a bound image's extent nor a bound buffer's size is known until one is bound, so
    /// a dispatch measured off either is left to be counted when the program runs.
    #[test]
    fn a_dispatch_measured_off_a_bound_resource_is_sized_when_it_runs() {
        let mut module = Module::default();
        let target = module.declare_image_var("target", FORMAT, vk::SampleCountFlags::TYPE_1, LAYOUT);
        let elements = module.declare_buffer_var("elements", Access::HostWrite);

        let painted = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(target)
            .dispatch_invocations_per_pixel(target)
            .end_compute();
        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(elements)
            .read(painted)
            .dispatch_invocations_per_element(elements, 16)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        match dispatch_sizes(&compiled).as_slice() {
            [
                ir::DispatchSize::InvocationsPerPixel { image, .. },
                ir::DispatchSize::InvocationsPerElement {
                    buffer, element_size, ..
                },
            ] => {
                assert_eq!(module.resource_root(*image), target);
                assert_eq!(module.resource_root(*buffer), elements);
                assert_eq!(*element_size, 16);
            },
            sizes => panic!("expected both dispatches to keep their resource, got {sizes:?}"),
        }
    }

    /// A bound resource is neither allocated by the graph nor handed a handle when it is
    /// declared, so it has to read as neither in a dump.
    #[test]
    fn a_bound_resource_reads_as_bound() {
        let mut module = Module::default();
        let instances = module.declare_buffer_var("instances", Access::HostWrite);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(instances)
            .dispatch(1, 1, 1)
            .end_compute();

        let dump = module.compile(&Unchecked, end).unwrap().dump();
        assert!(dump.contains("= const \"instances\""), "{dump}");
        assert!(dump.contains("buffer %") && dump.contains("(\"instances\")"), "{dump}");
        assert!(dump.contains(" bound"), "{dump}");
        assert!(!dump.contains("transient"), "{dump}");
    }

    /// The slot is the construct, so what binds a handle to it is the same call that writes
    /// every other variable.
    #[test]
    fn a_handle_reaches_the_slot_that_named_the_resource() {
        let mut module = Module::default();
        let instances = module.declare_buffer_var("instances", Access::HostWrite);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(instances)
            .dispatch(1, 1, 1)
            .end_compute();

        let mut compiled = module.compile(&Unchecked, end).unwrap();
        assert!(matches!(compiled.variables()[0].value, Value::None));

        compiled.set(instances, Buffer::default());
        assert!(matches!(compiled.variables()[0].value, Value::Buffer(_)));
        assert_eq!(compiled.variables()[0].resource, Some(instances));
    }

    #[test]
    #[should_panic(expected = "was given")]
    fn binding_the_wrong_kind_to_a_resource_slot_is_rejected() {
        let mut module = Module::default();
        let instances = module.declare_buffer_var("instances", Access::HostWrite);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(instances)
            .dispatch(1, 1, 1)
            .end_compute();

        module.compile(&Unchecked, end).unwrap().set(instances, 3u32);
    }

    /// The formats the pipelines were built for are the graph's to keep, so an image that does
    /// not match them is refused rather than left to mismatch when the region is recorded.
    #[test]
    #[should_panic(expected = "was declared")]
    fn binding_an_image_of_another_format_is_rejected() {
        let mut module = Module::default();
        let target = module.declare_image_var("target", FORMAT, vk::SampleCountFlags::TYPE_1, LAYOUT);

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();

        let extent = vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1);
        let other = ImageAttachment::new(
            Image::imported(vk::Image::null(), DEPTH_FORMAT, extent, vk::SampleCountFlags::TYPE_1),
            DEPTH_FORMAT,
            extent,
            vk::SampleCountFlags::TYPE_1,
            LAYOUT,
        );

        module.compile(&Unchecked, end).unwrap().set(target, other);
    }

    /// A module re-run over its own buffers has to wait on what the previous run left them to,
    /// which the host write an untouched buffer rests at would not cover.
    #[test]
    fn an_imported_buffer_waits_at_the_access_it_was_imported_at() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::default(), Access::AttributeRead);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(buffer)
            .dispatch(1, 1, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::AttributeRead, Access::ComputeWrite)]
        );
    }

    /// A memory barrier names no buffer, so two buffers waiting on the same pair are both
    /// covered by the one the first of them asks for. Emitting it at all is the part worth
    /// pinning: the two share their resting constant, and the barrier the write needs would
    /// be the thing lost to their states comparing equal.
    #[test]
    fn buffers_waiting_on_the_same_pair_share_one_barrier() {
        let mut module = Module::default();
        let first = module.import_buffer(&Buffer::default(), Access::ComputeRead);
        let second = module.import_buffer(&Buffer::default(), Access::ComputeRead);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(first)
            .write(second)
            .dispatch(1, 1, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::ComputeRead, Access::ComputeWrite)]
        );
    }

    /// The run a barrier is shared across is one instruction's worth. A second pass reaching
    /// the same buffer at the same access is a use in its own right, and has to wait again.
    #[test]
    fn a_later_pass_waiting_on_the_same_pair_gets_its_own_barrier() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::default(), Access::ComputeRead);

        let written = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(buffer)
            .dispatch(1, 1, 1)
            .end_compute();
        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(1))
            .read(written)
            .dispatch(1, 1, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![
                (Access::ComputeRead, Access::ComputeWrite),
                (Access::ComputeWrite, Access::ComputeRead),
            ]
        );
    }

    /// The counts a compiled program runs at are the variables, not what they held while it was
    /// being recorded.
    #[test]
    fn a_dispatch_and_a_draw_can_read_their_counts_from_variables() {
        let (mut module, attachment) = module_with_attachment();
        let invocations = module.declare_u32_var("invocations", 0);
        let vertices = module.declare_u32_var("vertices", 0);
        let one = module.constant_u32(1);

        let dispatched = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .dispatch_invocations(invocations, one, one)
            .end_compute();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(vertices, one)
            .end_rendering();

        let compiled = module.compile_all(&Unchecked, &[dispatched, end]).unwrap();
        let sizes = compiled
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Dispatch { size, .. } => Some(*size),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sizes,
            vec![ir::DispatchSize::Invocations {
                x: invocations,
                y: one,
                z: one,
            }]
        );

        let counts = compiled
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw {
                    vertex_count,
                    instance_count,
                    ..
                } => Some((*vertex_count, *instance_count)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(counts, vec![(vertices, one)]);
    }

    /// A depth attachment reaches the pipeline through the region's rendering state, which is
    /// what makes the created pipeline agree with the attachment it renders into.
    #[test]
    fn a_region_opened_with_depth_infers_the_depth_format() {
        let (mut module, attachment) = module_with_attachment();
        let depth = module.transient_image(&depth_info());

        let end = module
            .begin_rendering_depth(&[attachment], depth)
            .bind_graphics_pipeline(PipelineId(0))
            .set_depth(DepthState::less())
            .draw(3, 1)
            .end_rendering();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let draws = draws(&compiled);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
        assert_eq!(draws[0].1.rendering.depth_format, Some(DEPTH_FORMAT));
        assert_eq!(draws[0].1.depth, DepthState::less());
    }

    /// Depth state a region without a depth attachment cannot honour is dropped, so the draw
    /// does not ask for a pipeline that would fail validation.
    #[test]
    fn depth_state_without_a_depth_attachment_is_dropped() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_depth(DepthState::less())
            .draw(3, 1)
            .end_rendering();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let draws = draws(&compiled);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.rendering.depth_format, None);
        assert_eq!(draws[0].1.depth, DepthState::default());
    }

    /// The region writes the depth attachment, so the graph has to move it into the layout an
    /// attachment write wants before the region opens.
    #[test]
    fn the_depth_attachment_is_transitioned_for_the_region() {
        let (mut module, attachment) = module_with_attachment();
        let depth = module.transient_image(&depth_info());

        let end = module
            .begin_rendering_depth(&[attachment], depth)
            .bind_graphics_pipeline(PipelineId(0))
            .set_depth(DepthState::less())
            .draw(3, 1)
            .end_rendering();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let barrier = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == module.resource_root(depth))
            .expect("the depth attachment should be transitioned");

        assert_eq!(barrier.dst, Access::DepthStencilRW);
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    /// Nothing declares the usage of a transient depth target, so rendering into it is what has
    /// to put the attachment bit on.
    #[test]
    fn rendering_into_a_transient_depth_target_infers_its_usage() {
        let (mut module, attachment) = module_with_attachment();
        let depth = module.transient_image(&ImageInfo {
            extent: vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1),
            format: DEPTH_FORMAT,
            ..Default::default()
        });

        let end = module
            .begin_rendering_depth(&[attachment], depth)
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();
        let compiled = module.compile(&Unchecked, end).unwrap();

        assert!(
            image_usage(&compiled, depth).contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            "the depth attachment should be created as one"
        );
    }

    #[test]
    fn rebinding_the_same_vertex_buffer_does_not_repeat_the_barrier() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(memory_barriers(&module, &compiled).len(), 1);
    }

    #[test]
    fn binding_an_index_buffer_makes_the_host_write_visible_to_index_input() {
        let (mut module, attachment) = module_with_attachment();
        let indices = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let instructions = compiled.instructions();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::IndexRead)]
        );

        let position = |predicate: fn(&IR) -> bool| {
            instructions
                .iter()
                .position(|(_, ir)| predicate(ir))
                .expect("instruction should be present")
        };
        assert!(
            position(|ir| matches!(ir, IR::MemoryBarrier { .. }))
                < position(|ir| matches!(ir, IR::BeginRendering { .. }))
        );
    }

    /// Both buffers wait on the same host writes with nothing recorded in between, so the two
    /// waits fold into one that names every stage reading them.
    #[test]
    fn vertex_and_index_buffers_share_the_barrier_their_stages_need() {
        let (mut module, attachment) = module_with_attachment();
        let vertices = module.import_buffer(&Buffer::default(), Access::HostWrite);
        let indices = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, vertices)
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::AttributeRead | Access::IndexRead)],
            "{}",
            compiled.dump()
        );
    }

    #[test]
    fn rebinding_the_same_index_buffer_does_not_repeat_the_barrier() {
        let (mut module, attachment) = module_with_attachment();
        let indices = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(3, 1)
            .end_rendering();

        assert_eq!(
            memory_barriers(&module, &module.compile(&Unchecked, end).unwrap()).len(),
            1
        );
    }

    /// An indexed draw goes through the same state inference as a plain one; a variant missing
    /// from that pass would silently come out with no pipeline and default state.
    #[test]
    fn an_indexed_draw_inherits_the_pipeline_and_state_in_force() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);
        let indices = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .set_primitive_topology(vk::PrimitiveTopology::LINE_STRIP)
            .broadcast_color_blend(BlendPreset::PremultipliedAlphaBlend)
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw_indexed(6, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].0, pipeline);
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
        assert_eq!(draws[0].1.topology, vk::PrimitiveTopology::LINE_STRIP);
        assert!(draws[0].1.blend.iter().all(|blend| blend.blend_enable));
    }

    #[test]
    fn an_indexed_draw_lowers_its_range_with_a_signed_vertex_offset() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .draw_indexed_range(9, 2, 3, -4, 5)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let instructions = compiled.instructions();
        let (_, ir) = instructions
            .iter()
            .find(|(_, ir)| matches!(ir, IR::DrawIndexed { .. }))
            .expect("the indexed draw should survive compilation");

        let IR::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            ..
        } = ir
        else {
            unreachable!()
        };

        assert_eq!(u32_constant(&module, index_count), 9);
        assert_eq!(u32_constant(&module, instance_count), 2);
        assert_eq!(u32_constant(&module, first_index), 3);
        assert_eq!(i32_constant(&module, vertex_offset), -4);
        assert_eq!(u32_constant(&module, first_instance), 5);
    }

    #[test]
    fn a_draw_inherits_the_regions_formats_and_the_default_state() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].0, pipeline);
        assert_eq!(draws[0].1.rendering.color_formats, vec![FORMAT]);
        assert_eq!(draws[0].1.rasterization, RasterizationState::default());
        assert_eq!(draws[0].1.topology, vk::PrimitiveTopology::TRIANGLE_LIST);
    }

    #[test]
    fn state_set_after_bind_reaches_the_draw() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::BACK,
                polygon_mode: vk::PolygonMode::LINE,
                ..Default::default()
            })
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].1.rasterization.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(draws[0].1.rasterization.polygon_mode, vk::PolygonMode::LINE);
    }

    #[test]
    fn a_state_change_between_draws_splits_them_into_two_permutations() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .draw(3, 1)
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::FRONT,
                ..Default::default()
            })
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].1.rasterization.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(draws[1].1.rasterization.cull_mode, vk::CullModeFlags::FRONT);
        assert_eq!(draws[0].0, draws[1].0);
        assert_ne!(draws[0].1, draws[1].1);
    }

    #[test]
    fn draws_that_agree_on_state_share_one_permutation() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_primitive_topology(vk::PrimitiveTopology::LINE_STRIP)
            .draw(3, 1)
            .draw(6, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0], draws[1]);
        assert_eq!(draws[0].1.topology, vk::PrimitiveTopology::LINE_STRIP);
    }

    #[test]
    fn rebinding_a_declaration_keeps_the_state_accumulated_so_far() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .set_rasterization(RasterizationState {
                cull_mode: vk::CullModeFlags::BACK,
                ..Default::default()
            })
            .bind_graphics_pipeline(PipelineId(1))
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws[0].0, PipelineId(1));
        assert_eq!(draws[0].1.rasterization.cull_mode, vk::CullModeFlags::BACK);
    }

    #[test]
    fn a_draw_with_no_bound_pipeline_infers_invalid() {
        let (mut module, attachment) = module_with_attachment();

        let end = module.begin_rendering(&[attachment]).draw(3, 1).end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws[0].0, PipelineId::INVALID);
    }

    #[test]
    fn a_second_region_over_the_same_attachment_keeps_its_own_state() {
        let (mut module, attachment) = module_with_attachment();
        let pipeline = PipelineId(0);

        let attachment = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .set_rasterization(RasterizationState {
                polygon_mode: vk::PolygonMode::LINE,
                ..Default::default()
            })
            .draw(3, 1)
            .end_rendering();
        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(pipeline)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].1.rasterization.polygon_mode, vk::PolygonMode::LINE);
        assert_eq!(draws[1].1.rasterization.polygon_mode, vk::PolygonMode::FILL);
        assert_eq!(draws[1].1.rendering.color_formats, vec![FORMAT]);
    }

    #[test]
    fn a_dynamic_viewport_travels_with_the_draw_instead_of_the_pipeline() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_viewport(0, Rect2D::relative(0.0, 0.0, 0.5, 0.5))
            .draw(3, 1)
            .set_viewport(0, Rect2D::framebuffer())
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dynamic = compiled
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        // both draws share a permutation, only the recorded viewport differs
        let draws = draws(&compiled);
        assert_eq!(draws[0].1, draws[1].1);
        assert!(draws[0].1.viewports.is_empty());
        assert_eq!(dynamic[0].viewports, vec![Rect2D::relative(0.0, 0.0, 0.5, 0.5).into()]);
        assert_eq!(dynamic[1].viewports, vec![Viewport::framebuffer()]);
    }

    #[test]
    fn a_static_viewport_is_resolved_against_the_render_area_and_keys_the_pipeline() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .set_dynamic_state(DynamicStateFlags::None)
            .set_viewport(0, Rect2D::relative(0.0, 0.0, 0.5, 1.0))
            .set_scissor(0, Rect2D::framebuffer())
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(
            draws[0].1.viewports,
            vec![ResolvedViewport {
                x: 0.0,
                y: 0.0,
                width: (WIDTH / 2) as f32,
                height: HEIGHT as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }]
        );
        assert_eq!(
            draws[0].1.scissors,
            vec![vk::Rect2D::default().extent(vk::Extent2D {
                width: WIDTH,
                height: HEIGHT
            })]
        );
    }

    #[test]
    fn a_static_viewport_follows_the_render_area_a_region_was_opened_with() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering_area(
                &[attachment],
                vk::Extent2D {
                    width: WIDTH / 2,
                    height: HEIGHT / 2,
                },
            )
            .bind_graphics_pipeline(PipelineId(0))
            .set_dynamic_state(DynamicStateFlags::None)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws[0].1.viewports[0].width, (WIDTH / 2) as f32);
        assert_eq!(draws[0].1.viewports[0].height, (HEIGHT / 2) as f32);
    }

    #[test]
    fn push_constants_reach_the_draws_that_follow_them() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .push_constants(&[1.0f32, 2.0])
            .draw(3, 1)
            .push_constants_at(4, &3.0f32)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let pushed = compiled
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.push_constants.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        // the second push only touches the tail, so the first four bytes carry over
        assert_eq!(pushed[0].offset, 0);
        assert_eq!(pushed[0].data, [1.0f32, 2.0].map(f32::to_ne_bytes).concat());
        assert_eq!(pushed[1].data, [1.0f32, 3.0].map(f32::to_ne_bytes).concat());

        // and none of it splits the pipeline permutation
        let draws = draws(&compiled);
        assert_eq!(draws[0], draws[1]);
    }

    #[test]
    fn a_draw_recorded_before_a_push_does_not_see_it() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .push_constants(&7u32)
            .draw(3, 1)
            .end_rendering();

        let pushed = module
            .compile(&Unchecked, end)
            .unwrap()
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.push_constants.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(pushed[0].is_empty());
        assert_eq!(pushed[1].data, 7u32.to_ne_bytes());
    }

    #[test]
    fn a_push_before_the_pipeline_bind_still_reaches_the_draw() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .push_constants(&7u32)
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let (_, IR::Draw { dynamic, .. }) = compiled
            .instructions()
            .iter()
            .find(|(_, ir)| matches!(ir, IR::Draw { .. }))
            .expect("draw should be present")
        else {
            unreachable!()
        };

        assert_eq!(dynamic.push_constants.data, 7u32.to_ne_bytes());
        assert_eq!(draws(&compiled)[0].0, PipelineId(0));
    }

    #[test]
    fn a_dispatch_picks_up_the_compute_pipeline_and_the_pushes_in_force() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(1))
            .write(attachment)
            .push_constants(&7u32)
            .dispatch(8, 4, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let (
            _,
            IR::Dispatch {
                pipeline,
                push_constants,
                size: ir::DispatchSize::Groups { x, y, z },
                ..
            },
        ) = compiled
            .instructions()
            .iter()
            .find(|(_, ir)| matches!(ir, IR::Dispatch { .. }))
            .expect("dispatch should be present")
        else {
            unreachable!()
        };

        assert_eq!(*pipeline, PipelineId(1));
        assert_eq!(push_constants.data, 7u32.to_ne_bytes());
        assert_eq!(
            [
                u32_constant(&module, x),
                u32_constant(&module, y),
                u32_constant(&module, z)
            ],
            [8, 4, 1]
        );
    }

    fn dispatch_sizes(program: &Program) -> Vec<ir::DispatchSize> {
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Dispatch { size, .. } => Some(*size),
                _ => None,
            })
            .collect()
    }

    fn invocations(module: &Module, program: &Program) -> [u32; 3] {
        match dispatch_sizes(program).as_slice() {
            [ir::DispatchSize::Invocations { x, y, z }] => [
                u32_constant(module, x),
                u32_constant(module, y),
                u32_constant(module, z),
            ],
            sizes => panic!("expected one dispatch sized in invocations, got {sizes:?}"),
        }
    }

    /// The workgroup only enters the picture once a pipeline is bound, which is why the count
    /// reaches the compiled module as the invocations the caller asked for.
    #[test]
    fn a_dispatch_sized_in_invocations_keeps_the_count_it_was_given() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(attachment)
            .dispatch_invocations(1920, 1080, 1)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(invocations(&module, &compiled), [1920, 1080, 1]);
    }

    #[test]
    fn a_dispatch_per_pixel_covers_the_image_it_was_sized_from() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(attachment)
            .dispatch_invocations_per_pixel(attachment)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(invocations(&module, &compiled), [WIDTH, HEIGHT, 1]);
    }

    /// Scaling rounds up, so an axis that scales to a fraction of an invocation still gets one.
    #[test]
    fn a_scaled_dispatch_per_pixel_rounds_every_axis_up() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(attachment)
            .dispatch_invocations_per_pixel_scaled(attachment, [0.5, 2.0, 0.25])
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(invocations(&module, &compiled), [WIDTH / 2, HEIGHT * 2, 1]);
    }

    #[test]
    fn a_dispatch_per_element_sizes_itself_on_the_x_axis_alone() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::new(vk::Buffer::null(), 800, 0, None), Access::HostWrite);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(buffer)
            .dispatch_invocations_per_element(buffer, 20)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(invocations(&module, &compiled), [40, 1, 1]);
    }

    #[test]
    fn a_scaled_dispatch_per_element_scales_the_element_count() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::new(vk::Buffer::null(), 800, 0, None), Access::HostWrite);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(buffer)
            .dispatch_invocations_per_element_scaled(buffer, 20, 3.0)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(invocations(&module, &compiled), [120, 1, 1]);
    }

    /// A buffer of counts is no use to the device until the write that filled it is visible, and
    /// nothing but the dispatch itself says that it will be read as commands.
    #[test]
    fn an_indirect_dispatch_waits_for_the_buffer_of_counts() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());
        let commands = module.import_buffer(&Buffer::new(vk::Buffer::null(), 12, 0, None), Access::HostWrite);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(image)
            .dispatch_indirect(commands)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::IndirectRead)]
        );
        assert_eq!(
            dispatch_sizes(&compiled),
            vec![ir::DispatchSize::Indirect {
                buffer: commands,
                offset: 0
            }]
        );
    }

    /// The counts are filled once, so the second dispatch reading the same buffer waits on
    /// nothing further.
    #[test]
    fn indirect_dispatches_out_of_one_buffer_wait_on_it_once() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());
        let commands = module.import_buffer(&Buffer::new(vk::Buffer::null(), 24, 0, None), Access::HostWrite);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(image)
            .dispatch_indirect(commands)
            .dispatch_indirect_at(commands, 12)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            dispatch_sizes(&compiled),
            vec![
                ir::DispatchSize::Indirect {
                    buffer: commands,
                    offset: 0
                },
                ir::DispatchSize::Indirect {
                    buffer: commands,
                    offset: 12
                },
            ]
        );
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::IndirectRead)]
        );
    }

    #[test]
    fn a_dispatch_that_writes_an_image_puts_it_in_general_with_storage_usage() {
        let (mut module, swapchain) = module_with_attachment();
        let target = module.transient_image(&untyped_info());

        let computed = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(target)
            .dispatch(8, 8, 1)
            .end_compute();
        // the region's value stands in for what it wrote, so the blit reads the dispatch
        let swapchain = module.blit(computed, swapchain);
        let end = module.present(swapchain);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let written = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == target)
            .expect("the written image should have been transitioned");

        assert_eq!(written.dst, Access::ComputeWrite);
        assert_eq!(written.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(written.new_layout, vk::ImageLayout::GENERAL);
        assert_eq!(
            image_usage(&compiled, target),
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC
        );
    }

    #[test]
    fn a_dispatch_that_reads_what_an_earlier_one_wrote_waits_for_it() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());

        let written = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(image)
            .dispatch(1, 1, 1)
            .end_compute();
        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(1))
            .read(written)
            .dispatch(1, 1, 1)
            .end_compute();

        let barriers = image_barriers(&module, &module.compile(&Unchecked, end).unwrap());
        assert_eq!(barriers.len(), 2);
        assert_eq!(barriers[1].src, Access::ComputeWrite);
        assert_eq!(barriers[1].dst, Access::ComputeRead);
        assert_eq!(barriers[1].old_layout, vk::ImageLayout::GENERAL);
        assert_eq!(barriers[1].new_layout, vk::ImageLayout::GENERAL);
    }

    #[test]
    fn a_resource_declared_twice_carries_both_accesses() {
        let mut module = Module::default();
        let image = module.transient_image(&untyped_info());

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .read(image)
            .write(image)
            .dispatch(1, 1, 1)
            .end_compute();

        let barriers = image_barriers(&module, &module.compile(&Unchecked, end).unwrap());
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].dst, Access::ComputeRW);
    }

    #[test]
    fn recorded_accesses_are_lowered_to_constants() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::default(), Access::HostWrite);
        let image = module.transient_image(&untyped_info());

        let _end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .read(image)
            .bind_image(0, 0, image)
            .dispatch(1, 1, 1)
            .end_compute();

        let IR::ConstructBuffer { initial_access, .. } = module.get(buffer) else {
            panic!("the imported buffer should be constructed directly");
        };
        assert_eq!(module.resolve_access(*initial_access), Access::HostWrite);

        let declared = module.instructions.iter().find_map(|ir| match ir {
            IR::BeginCompute { declared_access, .. } => declared_access
                .iter()
                .find_map(|(resource, access)| (*resource == image).then_some(*access)),
            _ => None,
        });
        assert_eq!(declared.map(|id| module.resolve_access(id)), Some(Access::ComputeRead));

        let descriptor = module.instructions.iter().find_map(|ir| match ir {
            IR::WriteDescriptor { access, .. } => Some(*access),
            _ => None,
        });
        assert_eq!(descriptor.map(|id| module.resolve_access(id)), Some(Access::None));
    }

    #[test]
    fn a_buffer_a_dispatch_wrote_is_made_visible_to_the_draw_that_reads_it() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let computed = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(1))
            .write(buffer)
            .dispatch(1, 1, 1)
            .end_compute();
        let rendered = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, buffer)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile_all(&Unchecked, &[computed, rendered]).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![
                (Access::HostWrite, Access::ComputeWrite),
                (Access::ComputeWrite, Access::AttributeRead)
            ]
        );
    }

    /// A dispatch cannot be recorded inside a rendering region, so a buffer the draw binds has
    /// to bring its producer out ahead of the region rather than into it, even when the draw is
    /// the only root and the dispatch is reachable through the bound buffer alone.
    #[test]
    fn a_dispatch_the_draw_binds_the_result_of_stays_outside_the_region() {
        let (mut module, attachment) = module_with_attachment();
        let buffer = module.import_buffer(&Buffer::default(), Access::HostWrite);

        let computed = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(1))
            .write(buffer)
            .dispatch(1, 1, 1)
            .end_compute();
        let rendered = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, computed)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, rendered).unwrap();
        let position = |predicate: fn(&IR) -> bool| {
            compiled
                .instructions()
                .iter()
                .position(|(_, ir)| predicate(ir))
                .expect("instruction should be present")
        };

        let end_compute = position(|ir| matches!(ir, IR::EndCompute { .. }));
        let begin_rendering = position(|ir| matches!(ir, IR::BeginRendering { .. }));
        assert!(
            end_compute < begin_rendering,
            "the compute region closed at {end_compute} but the rendering region opened at {begin_rendering}"
        );

        // and the barrier that hands the buffer over is hoisted out with it
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![
                (Access::HostWrite, Access::ComputeWrite),
                (Access::ComputeWrite, Access::AttributeRead)
            ]
        );
    }

    #[test]
    fn a_copy_waits_for_the_host_write_and_puts_the_image_in_the_transfer_layout() {
        let mut module = Module::default();
        let staging = module.import_buffer(&Buffer::default(), Access::HostWrite);
        let texture = module.transient_image(&untyped_info());
        let end = module.copy_buffer_to_image(staging, texture);

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![(Access::HostWrite, Access::CopyRead)]
        );

        let barriers = image_barriers(&module, &compiled);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].resource, texture);
        assert_eq!(barriers[0].dst, Access::CopyWrite);
        assert_eq!(barriers[0].old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barriers[0].new_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    /// The whole point of naming a sampled image: a layout transition cannot be recorded
    /// inside a rendering region, so it has to be hoisted out the way an attachment's is.
    #[test]
    fn a_sampled_image_reaches_the_shader_read_layout_before_the_region_opens() {
        let (mut module, attachment) = module_with_attachment();
        let texture = module.transient_image(&untyped_info());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .access(texture, Access::FragmentSampled)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let sampled = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == texture)
            .expect("the sampled image should have been transitioned");
        assert_eq!(sampled.dst, Access::FragmentSampled);
        assert_eq!(sampled.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let last_barrier = compiled
            .instructions()
            .iter()
            .rposition(|(_, ir)| matches!(ir, IR::ImageBarrier { .. }))
            .expect("barriers should be present");
        let begin = compiled
            .instructions()
            .iter()
            .position(|(_, ir)| matches!(ir, IR::BeginRendering { .. }))
            .expect("the region should be present");
        assert!(last_barrier < begin);
    }

    fn descriptor_writes(program: &Program) -> Vec<(u32, u32, ir::Descriptor)> {
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::WriteDescriptor {
                    set,
                    binding,
                    descriptor,
                    ..
                } => Some((*set, *binding, *descriptor)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_rendering_descriptor_is_written_by_an_instruction_of_its_own() {
        let (mut module, attachment) = module_with_attachment();
        let first = module.transient_image(&untyped_info());
        let second = module.transient_image(&untyped_info());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_image(2, 4, first)
            .bind_texture(2, 4, second, a_sampler())
            .draw(3, 1)
            .end_rendering();

        let bindings = Declared::new(&[(2, 4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let compiled = module.compile(&bindings, end).unwrap();
        // both writes survive: what the draw sees is the last one, but the first one remains in
        // the recorded command stream
        assert_eq!(
            descriptor_writes(&compiled),
            [
                (2, 4, ir::Descriptor::SampledImage { image: first }),
                (
                    2,
                    4,
                    ir::Descriptor::CombinedImageSampler {
                        image: second,
                        sampler: a_sampler(),
                    }
                ),
            ]
        );

        let sampled = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == second)
            .expect("the bound image should be transitioned");
        assert_eq!(sampled.dst, Access::FragmentSampled);
    }

    #[test]
    fn a_compute_descriptor_is_written_by_an_instruction_of_its_own() {
        let mut module = Module::default();
        let texture = module.transient_image(&untyped_info());
        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .bind_image(1, 3, texture)
            .dispatch(1, 1, 1)
            .end_compute();

        let bindings = Declared::in_stages(
            &[(1, 3, vk::DescriptorType::SAMPLED_IMAGE)],
            vk::ShaderStageFlags::COMPUTE,
        );
        let compiled = module.compile(&bindings, end).unwrap();
        assert_eq!(
            descriptor_writes(&compiled),
            [(1, 3, ir::Descriptor::SampledImage { image: texture })]
        );

        let sampled = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == texture)
            .expect("the bound image should be transitioned");
        assert_eq!(sampled.dst, Access::ComputeSampled);
    }

    /// A layout transition cannot be recorded inside a region, so the barrier for an image a
    /// descriptor names has to be emitted before the region it is written in opens.
    #[test]
    fn a_written_descriptor_is_transitioned_before_its_region_opens() {
        let (mut module, attachment) = module_with_attachment();
        let texture = module.transient_image(&untyped_info());
        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_texture(0, 0, texture, a_sampler())
            .draw(3, 1)
            .end_rendering();

        let bindings = Declared::new(&[(0, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let compiled = module.compile(&bindings, end).unwrap();
        let instructions = compiled.instructions();
        let barrier = instructions
            .iter()
            .position(|(_, ir)| matches!(ir, IR::ImageBarrier { value, .. } if module.resource_root(*value) == texture))
            .expect("the written image should be transitioned");
        let begin = instructions
            .iter()
            .position(|(_, ir)| matches!(ir, IR::BeginRendering { .. }))
            .expect("the region should survive");

        assert!(barrier < begin);
    }

    #[test]
    fn sampling_the_same_image_twice_in_a_region_does_not_repeat_the_barrier() {
        let (mut module, attachment) = module_with_attachment();
        let texture = module.transient_image(&untyped_info());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .access(texture, Access::FragmentSampled)
            .draw(3, 1)
            .access(texture, Access::FragmentSampled)
            .draw(3, 1)
            .end_rendering();

        let barriers = image_barriers(&module, &module.compile(&Unchecked, end).unwrap());
        assert_eq!(barriers.iter().filter(|b| b.resource == texture).count(), 1);
    }

    #[test]
    fn an_image_uploaded_and_then_sampled_moves_through_both_layouts_in_order() {
        let (mut module, attachment) = module_with_attachment();
        let staging = module.import_buffer(&Buffer::default(), Access::HostWrite);
        let texture = module.transient_image(&untyped_info());
        let uploaded = module.copy_buffer_to_image(staging, texture);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .access(uploaded, Access::FragmentSampled)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let layouts = image_barriers(&module, &compiled)
            .iter()
            .filter(|barrier| barrier.resource == texture)
            .map(|barrier| (barrier.old_layout, barrier.new_layout))
            .collect::<Vec<_>>();

        assert_eq!(
            layouts,
            vec![
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                ),
            ]
        );
    }

    #[test]
    fn an_uploaded_and_sampled_image_is_created_with_both_usages() {
        let (mut module, attachment) = module_with_attachment();
        let staging = module.import_buffer(&Buffer::default(), Access::HostWrite);
        let texture = module.transient_image(&untyped_info());
        let uploaded = module.copy_buffer_to_image(staging, texture);

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .access(uploaded, Access::FragmentSampled)
            .draw(3, 1)
            .end_rendering();

        assert_eq!(
            image_usage(&module.compile(&Unchecked, end).unwrap(), texture),
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED
        );
    }

    /// A copy takes the extent off the image it is filling, so a region that names its own
    /// has to be the one that survives.
    #[test]
    fn a_copy_region_overrides_the_extent_the_whole_image_would_have_used() {
        let mut module = Module::default();
        let staging = module.import_buffer(&Buffer::default(), Access::HostWrite);
        let texture = module.transient_image(&untyped_info());

        let whole = module.copy_buffer_to_image(staging, texture);
        let patch = module.copy_buffer_to_image_region(
            staging,
            whole,
            BufferImageCopy::region(vk::Offset2D { x: 4, y: 8 }, vk::Extent2D { width: 2, height: 3 }),
        );

        let regions = module
            .compile(&Unchecked, patch)
            .unwrap()
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::CopyBufferToImage {
                    region: Some(region), ..
                } => Some((region.image_offset, region.image_extent)),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            regions,
            vec![
                (
                    vk::Offset3D::default(),
                    vk::Extent3D::default().width(WIDTH).height(HEIGHT).depth(1)
                ),
                (
                    vk::Offset3D { x: 4, y: 8, z: 0 },
                    vk::Extent3D::default().width(2).height(3).depth(1)
                ),
            ]
        );
    }

    #[test]
    fn blend_state_is_per_attachment_and_keys_the_pipeline() {
        let (mut module, attachment) = module_with_attachment();

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .broadcast_color_blend(BlendPreset::AlphaBlend)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws[0].1.blend, vec![ColorBlendAttachmentState::default()]);
        assert_eq!(draws[1].1.blend, vec![BlendPreset::AlphaBlend.into()]);
        assert_ne!(draws[0].1, draws[1].1);
    }

    /// Constants are interned so the same number is one value, but two variables are two
    /// things the caller sets independently, however alike they start out.
    #[test]
    fn variables_that_start_the_same_are_still_two_separate_slots() {
        let mut module = Module::default();
        assert_eq!(module.lower_u32(7), module.lower_u32(7));

        let first = module.declare_u32_var("first", 7);
        let second = module.declare_u32_var("second", 7);
        assert_ne!(first, second);
    }

    #[test]
    fn setting_a_variable_leaves_the_compiled_instructions_alone() {
        let (mut module, target) = module_with_attachment();
        let color = module.declare_clear_var("clear color", clear::f32::BLACK);
        let cleared = module.clear_from(target, color);
        let end = module.present(cleared);

        let before = module.compile(&Unchecked, end).unwrap();
        let mut after = module.compile(&Unchecked, end).unwrap();
        after.set(color, clear::f32::WHITE);

        assert_eq!(before.instructions().len(), after.instructions().len());
        assert!(
            before
                .instructions()
                .iter()
                .zip(after.instructions())
                .all(|((a, _), (b, _))| a == b)
        );
    }

    #[test]
    #[should_panic(expected = "was given")]
    fn setting_a_variable_to_the_wrong_kind_is_rejected() {
        let mut module = Module::default();
        let color = module.declare_clear_var("clear color", clear::f32::BLACK);
        let target = module.transient_image(&transient_info());
        let end = module.clear_from(target, color);

        module.compile(&Unchecked, end).unwrap().set(color, 3u32);
    }

    /// A pipeline outlives the frame it was compiled in, so an extent that is only known at
    /// execute time cannot be folded into one.
    #[test]
    fn an_image_sized_by_a_variable_leaves_the_viewport_to_be_recorded() {
        let mut module = Module::default();
        let extent = vk::Extent3D::default().width(8).height(8).depth(1);
        let extent = module.declare_extent_3d_var("extent", extent);
        let target = module.transient_image_sized(&transient_info(), extent);

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .set_viewport(0, Rect2D::framebuffer())
            .set_dynamic_state(DynamicStateFlags::None)
            .draw(3, 1)
            .end_rendering();

        let draws = draws(&module.compile(&Unchecked, end).unwrap());
        assert_eq!(draws.len(), 1);
        assert!(draws[0].1.viewports.is_empty(), "{:?}", draws[0].1.viewports);
        assert!(draws[0].1.scissors.is_empty(), "{:?}", draws[0].1.scissors);
        assert!(draws[0].1.dynamic.contains(DynamicStateFlags::Viewport));
        assert!(draws[0].1.dynamic.contains(DynamicStateFlags::Scissor));
    }

    /// An extent that is not known until the program runs cannot be turned into a count
    /// here, so the dispatch keeps the image and is sized when it executes.
    #[test]
    fn a_dispatch_per_pixel_of_a_variable_sized_image_is_sized_when_it_runs() {
        let mut module = Module::default();
        let extent = vk::Extent3D::default().width(8).height(8).depth(1);
        let extent = module.declare_extent_3d_var("extent", extent);
        let target = module.transient_image_sized(&transient_info(), extent);

        let end = module
            .begin_compute()
            .bind_compute_pipeline(PipelineId(0))
            .write(target)
            .dispatch_invocations_per_pixel(target)
            .end_compute();

        let compiled = module.compile(&Unchecked, end).unwrap();
        match dispatch_sizes(&compiled).as_slice() {
            [ir::DispatchSize::InvocationsPerPixel { image, scale }] => {
                assert_eq!(module.resource_root(*image), target);
                assert_eq!(ir::DispatchSize::scale(*scale), [1.0; 3]);
            },
            sizes => panic!("expected one dispatch sized per pixel, got {sizes:?}"),
        }
    }

    /// A conditional clear followed by a blit: the arm that clears leaves the image in the
    /// transfer-destination layout, the arm that does not leaves it untouched, and the two
    /// have to agree before the blit that follows the merge can name one layout.
    fn module_with_a_conditional_clear() -> (Module, ValueId, ValueId) {
        let (mut module, destination) = module_with_attachment();
        let source = module.transient_image(&transient_info());
        let animate = module.declare_bool_var("animate", true);

        let cleared = module.set_condition(animate, |m| m.clear(source, clear::f32::WHITE), |_| source);
        let end = module.blit(cleared, destination);

        (module, source, end)
    }

    fn block_at(compiled: &Program, label: LabelId) -> usize {
        compiled
            .instructions()
            .iter()
            .position(|(_, ir)| matches!(ir, IR::Label { label: l, .. } if *l == label))
            .expect("every declared label is placed")
    }

    /// The labels of the one selection the module records, read back off the branch rather
    /// than assumed, since which number a label gets is not part of the contract.
    fn selection_labels(compiled: &Program) -> (LabelId, LabelId, LabelId) {
        let merge = compiled
            .instructions()
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::SelectionMerge { merge } => Some(*merge),
                _ => None,
            })
            .expect("the module records one selection");

        let (taken, skipped) = compiled
            .instructions()
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::BranchConditional {
                    true_label,
                    false_label,
                    ..
                } => Some((*true_label, *false_label)),
                _ => None,
            })
            .expect("the selection branches");

        (merge, taken, skipped)
    }

    /// Every block ends in a terminator, whether or not it was given one: a block the builder
    /// never branched out of falls back to a return rather than running into the next label.
    #[test]
    fn every_block_ends_in_a_terminator() {
        let (module, _, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();
        let instructions = compiled.instructions();

        let terminated = |index: usize| {
            matches!(
                instructions.get(index),
                Some((_, IR::Branch { .. } | IR::BranchConditional { .. } | IR::Return))
            )
        };

        // the globals sit ahead of the first label, so only a label that follows another one
        // closes a block
        let mut blocks = 0;
        for (index, (_, ir)) in instructions.iter().enumerate() {
            if !matches!(ir, IR::Label { .. }) {
                continue;
            }
            if blocks > 0 {
                assert!(
                    terminated(index - 1),
                    "the block before {index} runs on\n{}",
                    compiled.dump()
                );
            }
            blocks += 1;
        }

        assert!(blocks > 0, "{}", compiled.dump());
        assert!(
            terminated(instructions.len() - 1),
            "the last block runs off the end\n{}",
            compiled.dump()
        );
    }

    /// A block ends in a branch, and nothing after that terminator runs. Resources and
    /// variables are global, so they have to land ahead of the first branch rather than
    /// merely inside the entry block.
    #[test]
    fn every_global_lands_before_the_first_branch() {
        let (module, _, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let first_branch = compiled
            .instructions()
            .iter()
            .position(|(_, ir)| matches!(ir, IR::Branch { .. } | IR::BranchConditional { .. }))
            .expect("the selection branches");

        for (index, (id, ir)) in compiled.instructions().iter().enumerate() {
            assert!(
                !is_global(ir) || index < first_branch,
                "{id} is global but sits past the branch at {first_branch}\n{}",
                compiled.dump()
            );
        }
    }

    /// The barrier pass shares one access constant between every barrier waiting on that
    /// access. If it were left inside an arm, the other arm's barrier would name a value the
    /// path it ran on never produced.
    #[test]
    fn a_constant_a_barrier_names_is_never_defined_inside_a_block() {
        let (module, _, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();
        let instructions = compiled.instructions();

        let first_label = instructions
            .iter()
            .position(|(_, ir)| matches!(ir, IR::Label { .. }))
            .expect("the selection opens a block");

        let defined_before = instructions[..first_label]
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();

        for (_, ir) in instructions {
            let operands = match ir {
                IR::MemoryBarrier { src_access, dst_access } => vec![*src_access, *dst_access],
                IR::ImageBarrier {
                    src_access, dst_access, ..
                } => vec![*src_access, *dst_access],
                _ => continue,
            };

            for operand in operands {
                assert!(
                    defined_before.contains(&operand),
                    "{operand} is only defined on some paths\n{}",
                    compiled.dump()
                );
            }
        }
    }

    /// A branch jumps to the label, so anything the sort left ahead of it inside its own
    /// block would be stepped straight over.
    #[test]
    fn a_label_is_the_first_instruction_of_its_block() {
        let (module, _, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let mut open = None;
        for (id, ir) in compiled.instructions() {
            match ir {
                IR::Label { label, .. } => open = Some(*label),
                _ => assert!(
                    is_global(ir) || module.block_of(*id).is_none_or(|block| Some(block) == open),
                    "{id} comes before the label of the block it was recorded in\n{}",
                    compiled.dump()
                ),
            }
        }
    }

    #[test]
    fn a_selection_lays_each_block_out_in_one_piece() {
        let (module, _, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let mut seen = Vec::new();
        let mut current = None;
        for (id, ir) in compiled.instructions() {
            if let IR::Label { label, .. } = ir {
                assert!(!seen.contains(label), "block {label} is laid out in two pieces");
                seen.push(*label);
                current = Some(*label);
                continue;
            }

            // an instruction the module recorded in a block never drifts into another one
            if let Some(block) = module.block_of(*id)
                && !is_global(ir)
            {
                assert_eq!(Some(block), current, "{id} left the block it was recorded in");
            }
        }

        assert_eq!(seen.len(), 4, "an entry block plus the selection's three");
    }

    /// Nothing after the merge knows which way control went, so it can only name one layout
    /// for the image. The arm that skipped the clear has to be brought to the layout the arm
    /// that cleared ended in, however little it did itself.
    #[test]
    fn both_arms_of_a_selection_leave_a_resource_in_the_same_layout() {
        let (module, source, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let (merge, taken, skipped) = selection_labels(&compiled);
        let (merge, taken, skipped) = (
            block_at(&compiled, merge),
            block_at(&compiled, taken),
            block_at(&compiled, skipped),
        );

        let layout_leaving = |range: std::ops::Range<usize>| {
            compiled.instructions()[range]
                .iter()
                .filter_map(|(_, ir)| match ir {
                    IR::ImageBarrier { new_layout, value, .. } if module.resource_root(*value) == source => {
                        Some(*new_layout)
                    },
                    _ => None,
                })
                .next_back()
        };

        let cleared = layout_leaving(taken..skipped);
        assert_eq!(
            cleared,
            Some(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            "{}",
            compiled.dump()
        );
        assert_eq!(layout_leaving(skipped..merge), cleared, "{}", compiled.dump());
    }

    /// The arm that does nothing still has to end where the other one did, so reconciling it
    /// is the only reason it holds a barrier at all.
    #[test]
    fn an_arm_that_touches_nothing_is_still_brought_to_the_join() {
        let (module, source, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();

        let (merge, _, skipped) = selection_labels(&compiled);
        let barriers = compiled.instructions()[block_at(&compiled, skipped)..block_at(&compiled, merge)]
            .iter()
            .filter(|(_, ir)| matches!(ir, IR::ImageBarrier { value, .. } if module.resource_root(*value) == source))
            .count();
        assert_eq!(barriers, 1, "{}", compiled.dump());
    }

    /// An arm that never reached a buffer has nothing to make available, so bringing it to a
    /// join the other arm read at is a barrier with no work in it.
    #[test]
    fn an_arm_that_never_reached_a_buffer_waits_on_nothing_at_the_join() {
        let mut module = Module::default();
        let buffer = module.transient_buffer(&BufferInfo::new(
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        ));
        let target = module.transient_image(&transient_info());
        let enabled = module.declare_bool_var("enabled", true);

        let end = module.set_condition(
            enabled,
            |m| {
                m.begin_compute()
                    .bind_compute_pipeline(PipelineId(0))
                    .write(target)
                    .read(buffer)
                    .dispatch(1, 1, 1)
                    .end_compute()
            },
            |_| target,
        );

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(memory_barriers(&module, &compiled), vec![], "{}", compiled.dump());
    }

    /// Both arms leave the buffer written, so whichever ran it is written all the same and
    /// there is nothing for the join to say. The two writes hold constants of their own, which
    /// is what a join comparing them by name rather than by access would trip over.
    #[test]
    fn arms_that_write_a_buffer_the_same_way_need_no_barrier_at_the_join() {
        let mut module = Module::default();
        let buffer = module.import_buffer(&Buffer::default(), Access::ComputeRead);
        let enabled = module.declare_bool_var("enabled", true);

        let dispatch = |m: &mut Module, pipeline: u32| {
            m.begin_compute()
                .bind_compute_pipeline(PipelineId(pipeline))
                .write(buffer)
                .dispatch(1, 1, 1)
                .end_compute()
        };
        let end = module.set_condition(enabled, |m| dispatch(m, 0), |m| dispatch(m, 1));

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&module, &compiled),
            vec![
                (Access::ComputeRead, Access::ComputeWrite),
                (Access::ComputeRead, Access::ComputeWrite),
            ],
            "{}",
            compiled.dump()
        );
    }

    /// Two buffers an arm has to catch up on ask for the same pair, and a memory barrier names
    /// no buffer, so the one covers both.
    #[test]
    fn buffers_caught_up_at_the_same_pair_share_one_barrier() {
        let mut module = Module::default();
        let first = module.import_buffer(&Buffer::default(), Access::ComputeRead);
        let second = module.import_buffer(&Buffer::default(), Access::ComputeRead);
        let target = module.transient_image(&transient_info());
        let enabled = module.declare_bool_var("enabled", true);

        let read_both = |m: &mut Module, target| {
            m.begin_compute()
                .bind_compute_pipeline(PipelineId(0))
                .write(target)
                .read_uniform(first)
                .read_uniform(second)
                .dispatch(1, 1, 1)
                .end_compute()
        };

        let drawn = module.set_condition(enabled, |m| read_both(m, target), |_| target);
        let end = read_both(&mut module, drawn);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let (merge, _, skipped) = selection_labels(&compiled);
        let caught_up = compiled.instructions()[block_at(&compiled, skipped)..block_at(&compiled, merge)]
            .iter()
            .filter(|(_, ir)| matches!(ir, IR::MemoryBarrier { .. }))
            .count();
        assert_eq!(caught_up, 1, "{}", compiled.dump());
    }

    /// Buffers an arm binds and nothing past the merge names again are never asked what state
    /// they are in, so the arm that skipped the draw has nothing to catch up on.
    #[test]
    fn an_arm_skips_catching_up_a_resource_the_merge_never_reaches() {
        let (mut module, target) = module_with_attachment();
        let enabled = module.declare_bool_var("enabled", true);
        let vertices = module.declare_buffer_var("vertices", Access::HostWrite);
        let indices = module.declare_buffer_var("indices", Access::HostWrite);

        let drawn = module.set_condition(
            enabled,
            |m| {
                m.begin_rendering(&[target])
                    .bind_graphics_pipeline(PipelineId(0))
                    .bind_vertex_buffer(0, vertices)
                    .bind_index_buffer(indices, vk::IndexType::UINT32)
                    .draw(3, 1)
                    .end_rendering()
            },
            |_| target,
        );

        let end = module.release(drawn, Access::Present, DomainFlag::Present);
        let compiled = module.compile(&Unchecked, end).unwrap();
        let (merge, _, skipped) = selection_labels(&compiled);
        let caught_up = compiled.instructions()[block_at(&compiled, skipped)..block_at(&compiled, merge)]
            .iter()
            .filter(|(_, ir)| matches!(ir, IR::MemoryBarrier { .. }))
            .count();
        assert_eq!(caught_up, 0, "{}", compiled.dump());
    }

    /// Bytes pushed from a variable are only sized when compiled; what they hold is read
    /// when the draw is recorded, so the block has to reach the draw naming its variable.
    #[test]
    fn push_constants_from_a_variable_reach_the_draw_as_a_variable() {
        let (mut module, target) = module_with_attachment();
        let block = module.declare_bytes_var("push block", 16);

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .push_constants_from(block)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let pushed = compiled
            .instructions()
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.push_constants.clone()),
                _ => None,
            })
            .expect("the draw carries the block in force");

        assert_eq!(pushed.source, Some(block));
        assert_eq!(pushed.size(), 16);
        assert_eq!(pushed.offset, 0);
    }

    /// Nothing else in the graph names the block, so the state change pushing it is the only
    /// thing keeping its variable alive; without it the program runs with an empty slot.
    #[test]
    fn a_pushed_block_keeps_its_variable_in_the_program() {
        let (mut module, target) = module_with_attachment();
        let block = module.declare_bytes_var("push block", 16);

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .push_constants_from(block)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let declared = compiled
            .instructions()
            .iter()
            .any(|(id, ir)| *id == block && matches!(ir, IR::Variable { .. }));

        assert!(declared, "{}", compiled.dump());
    }

    /// A block cannot be half literal and half variable, so whichever was asked for last is
    /// the whole of it.
    #[test]
    fn writing_bytes_over_a_variable_block_replaces_it() {
        let (mut module, target) = module_with_attachment();
        let block = module.declare_bytes_var("push block", 16);

        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .push_constants_from(block)
            .push_constants(&7u32)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let pushed = compiled
            .instructions()
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::Draw { dynamic, .. } => Some(dynamic.push_constants.clone()),
                _ => None,
            })
            .expect("the draw carries the block in force");

        assert_eq!(pushed.source, None);
        assert_eq!(pushed.data, 7u32.to_ne_bytes());
    }

    #[test]
    fn a_branch_target_resolves_to_the_block_it_names() {
        let (module, _, end) = module_with_a_conditional_clear();
        let compiled = module.compile(&Unchecked, end).unwrap();

        for (_, ir) in compiled.instructions() {
            let targets = match ir {
                IR::Branch { target, .. } => vec![*target],
                IR::BranchConditional {
                    true_label,
                    false_label,
                    ..
                } => vec![*true_label, *false_label],
                _ => continue,
            };

            for target in targets {
                let index = compiled.label_index(target).expect("a branch names a placed block");
                assert!(matches!(compiled.instructions()[index], (_, IR::Label { label, .. }) if label == target));
            }
        }
    }
    /// What a pass reaches with no binding the graph can see, ordered by the access the caller
    /// states rather than one inferred from the kind of pass it is.
    #[test]
    fn a_declared_rendering_access_transitions_before_the_region_opens() {
        let (mut module, attachment) = module_with_attachment();
        let texture = module.transient_image(&untyped_info());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .access(texture, Access::VertexSampled)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let sampled = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == texture)
            .expect("the declared image should be transitioned");
        assert_eq!(sampled.dst, Access::VertexSampled);
        assert_eq!(sampled.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let instructions = compiled.instructions();
        let at = |find: fn(&IR) -> bool| instructions.iter().position(|(_, ir)| find(ir)).expect("present");
        assert!(
            at(|ir| matches!(ir, IR::ImageBarrier { .. })) < at(|ir| matches!(ir, IR::BeginRendering { .. })),
            "{}",
            compiled.dump()
        );
    }
    /// A fragment stage writing a storage image, an overdraw counter being the usual reason. It
    /// is not an attachment and has no binding the graph can see, so the region declares it: the
    /// layout, the usage and the barrier all come off that one access.
    #[test]
    fn a_fragment_write_declared_on_a_region_reaches_general_as_a_storage_image() {
        let (mut module, attachment) = module_with_attachment();
        let overdraw = module.transient_image(&untyped_info());

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .access(overdraw, Access::FragmentWrite)
            .draw(3, 1)
            .end_rendering();

        let compiled = module.compile(&Unchecked, end).unwrap();
        let barrier = image_barriers(&module, &compiled)
            .into_iter()
            .find(|barrier| barrier.resource == overdraw)
            .expect("the declared image should be transitioned");

        assert_eq!(barrier.dst, Access::FragmentWrite);
        assert_eq!(barrier.new_layout, vk::ImageLayout::GENERAL);
        assert_eq!(image_usage(&compiled, overdraw), vk::ImageUsageFlags::STORAGE);

        // and it still has to land before the region opens, since no transition can be recorded
        // inside one
        let instructions = compiled.instructions();
        let at = |find: fn(&IR) -> bool| instructions.iter().position(|(_, ir)| find(ir)).expect("present");
        assert!(
            at(|ir| matches!(ir, IR::ImageBarrier { .. })) < at(|ir| matches!(ir, IR::BeginRendering { .. })),
            "{}",
            compiled.dump()
        );
    }
}
