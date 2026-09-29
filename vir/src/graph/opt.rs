use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

use ash::vk;

use crate::{Access, IR, LabelId, Module, ValueId, graph::ir, resource::aspect_mask};

struct BlockAddress {
    label: LabelId,
    start: usize,
    body: Range<usize>,
    terminator: usize,
}

impl BlockAddress {
    fn span(&self) -> Range<usize> { self.start..self.terminator + 1 }
}

fn blocks(nodes: &[ir::Instr]) -> Vec<BlockAddress> {
    let mut blocks = Vec::new();
    let mut open: Option<(LabelId, usize)> = None;

    for (index, (_, ir)) in nodes.iter().enumerate() {
        match ir {
            IR::Label { label } => open = Some((*label, index)),
            _ if ir.is_terminator() => {
                if let Some((label, start)) = open.take() {
                    blocks.push(BlockAddress {
                        label,
                        start,
                        body: start + 1..index,
                        terminator: index,
                    });
                }
            },
            _ => {},
        }
    }

    blocks
}

fn used_values(nodes: &[ir::Instr]) -> HashSet<ValueId> {
    let mut used = HashSet::new();
    for (_, ir) in nodes {
        ir.visit_operands(|id| {
            used.insert(id);
        });
    }
    used
}

fn pinned(nodes: &[ir::Instr]) -> HashSet<LabelId> {
    nodes
        .iter()
        .filter_map(|(_, ir)| match ir {
            IR::SelectionMerge { merge } => Some(*merge),
            _ => None,
        })
        .collect()
}

fn is_empty(nodes: &[ir::Instr], block: &BlockAddress, used: &HashSet<ValueId>) -> bool {
    nodes[block.body.clone()]
        .iter()
        .all(|(id, ir)| !ir.side_effects().is_observable() && !used.contains(id))
}

fn predecessors_of(blocks: &[BlockAddress], nodes: &[ir::Instr], label: LabelId) -> Vec<LabelId> {
    blocks
        .iter()
        .filter(|block| match &nodes[block.terminator].1 {
            IR::Branch { target } => *target == label,
            IR::BranchConditional {
                true_label,
                false_label,
                ..
            } => *true_label == label || *false_label == label,
            _ => false,
        })
        .map(|block| block.label)
        .collect()
}

type RewrittenPhi = (usize, Vec<(ValueId, LabelId)>);

fn wire_phis(nodes: &[ir::Instr], from: LabelId, to: LabelId) -> Option<Vec<RewrittenPhi>> {
    let mut rewritten = Vec::new();

    for (index, (_, ir)) in nodes.iter().enumerate() {
        let IR::Phi { incoming } = ir else {
            continue;
        };
        if !incoming.iter().any(|(_, label)| *label == from) {
            continue;
        }

        let mut merged: Vec<(ValueId, LabelId)> = Vec::with_capacity(incoming.len());
        for (value, label) in incoming {
            let label = match *label == from {
                true => to,
                false => *label,
            };

            match merged.iter().find(|(_, seen)| *seen == label) {
                Some((seen, _)) if seen != value => return None,
                Some(_) => {},
                None => merged.push((*value, label)),
            }
        }

        rewritten.push((index, merged));
    }

    Some(rewritten)
}

fn wire_empty_block(nodes: &mut Vec<ir::Instr>) -> bool {
    let blocks = blocks(nodes);
    let used = used_values(nodes);
    let pinned = pinned(nodes);

    for block in blocks.iter().skip(1) {
        let IR::Branch { target } = nodes[block.terminator].1 else {
            continue;
        };
        if target == block.label || pinned.contains(&block.label) || !is_empty(nodes, block, &used) {
            continue;
        }

        // more than one way in means the phis of the successor grow an incoming per predecessor,
        // which is worth doing only once a construct builds such a shape
        let [predecessor] = predecessors_of(&blocks, nodes, block.label)[..] else {
            continue;
        };
        let Some(phis) = wire_phis(nodes, block.label, predecessor) else {
            continue;
        };
        let Some(from) = blocks.iter().find(|block| block.label == predecessor) else {
            continue;
        };

        // the edge moves first, so a predecessor that turns out not to branch here leaves the
        // phis as they were
        match &mut nodes[from.terminator].1 {
            IR::Branch { target: edge } => *edge = target,
            IR::BranchConditional {
                true_label,
                false_label,
                ..
            } => {
                if *true_label == block.label {
                    *true_label = target;
                }
                if *false_label == block.label {
                    *false_label = target;
                }
            },
            _ => continue,
        }

        for (index, incoming) in phis {
            nodes[index].1 = IR::Phi { incoming };
        }

        nodes.drain(block.span());
        return true;
    }

    false
}

fn fold_equal_targets(nodes: &mut Vec<ir::Instr>) -> bool {
    let found = nodes.iter().position(|(_, ir)| match ir {
        IR::BranchConditional {
            true_label,
            false_label,
            ..
        } => true_label == false_label,
        _ => false,
    });
    let Some(index) = found else {
        return false;
    };

    let IR::BranchConditional { true_label, .. } = nodes[index].1 else {
        return false;
    };
    nodes[index].1 = IR::Branch { target: true_label };

    if index > 0 && matches!(nodes[index - 1].1, IR::SelectionMerge { .. }) {
        nodes.remove(index - 1);
    }

    true
}

fn access_constants(nodes: &[ir::Instr]) -> HashMap<ValueId, Access> {
    nodes
        .iter()
        .filter_map(|(id, ir)| match ir {
            IR::Constant(ir::Constant::Access(access)) => Some((*id, *access)),
            _ => None,
        })
        .collect()
}

fn barrier_runs(nodes: &[ir::Instr]) -> Vec<Vec<usize>> {
    let mut runs = Vec::new();
    let mut run: Vec<usize> = Vec::new();
    let mut close = |run: &mut Vec<usize>| match run.len() > 1 {
        true => runs.push(std::mem::take(run)),
        false => run.clear(),
    };

    for (index, (_, ir)) in nodes.iter().enumerate() {
        match ir {
            IR::MemoryBarrier { .. } => run.push(index),
            IR::ImageBarrier { .. } => {},
            _ if !ir.side_effects().is_observable() => {},
            _ => close(&mut run),
        }
    }

    close(&mut run);

    runs
}

struct Folded {
    at: usize,
    src: Access,
    dst: Access,
}

fn fold_run(
    nodes: &[ir::Instr], accesses: &HashMap<ValueId, Access>, run: &[usize], kept: &mut Vec<Folded>,
    dropped: &mut HashSet<usize>,
) {
    let first = kept.len();

    for &at in run {
        let IR::MemoryBarrier { src_access, dst_access } = nodes[at].1 else {
            continue;
        };
        let (Some(src), Some(dst)) = (accesses.get(&src_access), accesses.get(&dst_access)) else {
            continue;
        };
        let (src, dst) = (*src, *dst);

        match kept[first..]
            .iter_mut()
            .find(|folded| folded.src == src || folded.dst == dst)
        {
            Some(folded) => {
                folded.src |= src;
                folded.dst |= dst;
                dropped.insert(at);
            },
            None => kept.push(Folded { at, src, dst }),
        }
    }
}

fn constant_for(
    nodes: &mut Vec<ir::Instr>, pool: &mut HashMap<Access, ValueId>, next_id: &mut u32, access: Access,
) -> ValueId {
    if let Some(id) = pool.get(&access) {
        return *id;
    }

    let id = ValueId(*next_id);
    *next_id += 1;
    pool.insert(access, id);
    nodes.push((id, IR::Constant(ir::Constant::Access(access))));
    id
}

enum BarrierScope {
    Image(ValueId),
    Memory,
}

fn is_barrier(ir: &IR) -> bool { matches!(ir, IR::MemoryBarrier { .. } | IR::ImageBarrier { .. }) }

fn is_fixed(ir: &IR) -> bool {
    // moving a barrier above a pass would make the pass wait on the barrier's source
    let fixed = ir::SideEffect::Control | ir::SideEffect::External | ir::SideEffect::Host;
    ir.side_effects().intersects(fixed) || ir.opens_region() || ir.closes_region()
}

impl Module {
    /// The attachment the executor sizes a rendering pass from when the pass names no area.
    fn sizing_attachment(&self, attachments: &[(ValueId, ValueId)]) -> Option<ValueId> {
        let elements_with = |range: Access| {
            attachments
                .iter()
                .filter(move |(_, access)| self.resolve_access(*access).intersects(range))
                .flat_map(|(resource, _)| self.resource_elements(*resource))
        };
        elements_with(Access::ColorRW)
            .next()
            .or_else(|| elements_with(Access::DepthStencilRW).next())
    }

    /// A load op clears only the render area, so the area has to be the whole image.
    fn pass_covers(&self, pass: &IR, cleared: ValueId, source: ValueId) -> bool {
        let IR::BeginRendering {
            attachments,
            render_area,
            ..
        } = pass
        else {
            return false;
        };

        let sizing = self.sizing_attachment(attachments);
        if !render_area.is_valid() && sizing == Some(cleared) {
            return true;
        }

        let extent_of = |image: ValueId| {
            self.resolve_image(image)
                .and_then(|image| image.extent)
                .map(|extent| vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                })
        };
        let area = match render_area.is_valid() {
            true => self.resolve_extent_2d(*render_area),
            false => sizing.and_then(extent_of),
        };
        matches!((area, extent_of(source)), (Some(area), Some(image)) if area == image)
    }

    /// A load op clears the one mip and layer the view renders to, and only the depth aspect.
    fn is_single_image(&self, source: ValueId) -> bool {
        if self.resource_elements(source) != [source] {
            return false;
        }

        let one = |id: &ValueId| matches!(self.get(*id), IR::Constant(ir::Constant::U32(1)));
        match self.resolve_resource(source) {
            Some(IR::ConstructImage {
                format,
                level_count,
                layer_count,
                ..
            }) => one(level_count) && one(layer_count) && !aspect_mask(*format).contains(vk::ImageAspectFlags::STENCIL),
            Some(IR::SwapchainImage { .. }) => true,
            _ => false,
        }
    }

    /// A clear whose result only ever opens a rendering pass as a written attachment becomes that
    /// attachment's load op, which clears every such attachment of the pass in one command and
    /// leaves no transfer to synchronize.
    pub(crate) fn fold_clears(&self, mut nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        let mut uses: HashMap<ValueId, Vec<usize>> = HashMap::new();
        for (index, (_, ir)) in nodes.iter().enumerate() {
            ir.visit_operands(|operand| uses.entry(operand).or_default().push(index));
        }

        let mut folded: HashMap<ValueId, (ValueId, ValueId)> = HashMap::new();
        for (id, ir) in &nodes {
            let IR::Clear {
                attachment: source,
                color,
            } = ir
            else {
                continue;
            };
            let Some(users) = uses.get(id) else {
                continue;
            };
            if !self.is_single_image(*source) {
                continue;
            }

            let mut renders = false;
            let foldable = users.iter().all(|&index| match &nodes[index].1 {
                pass @ IR::BeginRendering { attachments, .. } => {
                    renders = true;
                    let written = attachments.iter().all(|(resource, access)| {
                        let access = self.resolve_access(*access);
                        resource != id
                            || (access.writes() && access.intersects(Access::ColorRW | Access::DepthStencilRW))
                    });
                    written && self.pass_covers(pass, *id, *source)
                },
                IR::PassResult { resource, .. } => resource == id,
                _ => false,
            });
            if foldable && renders {
                folded.insert(*id, (*source, *color));
            }
        }

        if folded.is_empty() {
            return nodes;
        }

        nodes.retain(|(id, _)| !folded.contains_key(id));
        for (_, ir) in &mut nodes {
            match ir {
                IR::BeginRendering {
                    attachments, clears, ..
                } => {
                    for (resource, _) in attachments.iter_mut() {
                        if let Some((source, color)) = folded.get(resource) {
                            *resource = *source;
                            clears.push((*source, *color));
                        }
                    }
                },
                IR::PassResult { resource, .. } => {
                    if let Some((source, _)) = folded.get(resource) {
                        *resource = *source;
                    }
                },
                _ => {},
            }
        }

        nodes
    }

    fn barrier_scope(&self, ir: &IR) -> Option<BarrierScope> {
        match ir {
            IR::ImageBarrier { value, .. } => Some(BarrierScope::Image(self.resource_root(*value))),
            IR::MemoryBarrier { .. } => Some(BarrierScope::Memory),
            _ => None,
        }
    }

    fn holds_back(
        &self, (id, ir): &ir::Instr, scope: &BarrierScope, operands: &[ValueId], uses: &mut Vec<ValueId>,
    ) -> bool {
        operands.contains(id) || is_fixed(ir) || self.touches(scope, ir, uses)
    }

    fn touches(&self, scope: &BarrierScope, ir: &IR, uses: &mut Vec<ValueId>) -> bool {
        match (scope, self.barrier_scope(ir)) {
            (BarrierScope::Image(root), Some(BarrierScope::Image(other))) => return *root == other,
            (_, Some(_)) => return false,
            _ => {},
        }

        uses.clear();
        self.resource_uses(ir, uses);
        match scope {
            BarrierScope::Image(root) => uses.contains(root),
            // a memory barrier does not say which buffer it is for
            BarrierScope::Memory => uses.iter().any(|resource| self.is_buffer(*resource)),
        }
    }

    pub(crate) fn schedule_barriers(&self, nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        let mut result: Vec<ir::Instr> = Vec::with_capacity(nodes.len());
        let mut operands = Vec::new();
        let mut uses = Vec::new();

        for (id, ir) in nodes {
            let Some(scope) = self.barrier_scope(&ir) else {
                // a command no pending barrier is for goes ahead of them, so the barriers it
                // lets through can join the ones the next command needs
                let run = result.iter().rev().take_while(|(_, ir)| is_barrier(ir)).count();
                let start = result.len() - run;
                let free = run > 0
                    && !is_fixed(&ir)
                    && result[start..].iter().all(|(_, barrier)| {
                        self.barrier_scope(barrier)
                            .is_some_and(|scope| !self.touches(&scope, &ir, &mut uses))
                    });
                match free {
                    true => result.insert(start, (id, ir)),
                    false => result.push((id, ir)),
                }
                continue;
            };

            operands.clear();
            ir.visit_operands(|operand| operands.push(operand));

            // a barrier moves only to join an earlier run, since moving alone batches nothing
            let after = result
                .iter()
                .rposition(|node| self.holds_back(node, &scope, &operands, &mut uses))
                .map_or(0, |index| index + 1);
            let at = match result[after..].iter().position(|(_, ir)| is_barrier(ir)) {
                Some(offset) => {
                    let mut at = after + offset;
                    while result.get(at).is_some_and(|(_, ir)| is_barrier(ir)) {
                        at += 1;
                    }
                    at
                },
                None => result.len(),
            };
            result.insert(at, (id, ir));
        }

        result
    }

    pub(crate) fn simplify_cfg(&self, mut nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        loop {
            let mut changed = wire_empty_block(&mut nodes);
            changed |= fold_equal_targets(&mut nodes);

            if !changed {
                return nodes;
            }
        }
    }

    pub(crate) fn fold_barriers(&self, nodes: Vec<ir::Instr>, next_id: &mut u32) -> Vec<ir::Instr> {
        let accesses = access_constants(&nodes);

        let mut kept: Vec<Folded> = Vec::new();
        let mut dropped = HashSet::new();
        for run in barrier_runs(&nodes) {
            fold_run(&nodes, &accesses, &run, &mut kept, &mut dropped);
        }
        if dropped.is_empty() {
            return nodes;
        }

        let mut pool: HashMap<Access, ValueId> = HashMap::new();
        for (id, ir) in &nodes {
            if let IR::Constant(ir::Constant::Access(access)) = ir {
                pool.entry(*access).or_insert(*id);
            }
        }

        let mut result = Vec::with_capacity(nodes.len());
        for (index, (id, ir)) in nodes.into_iter().enumerate() {
            if dropped.contains(&index) {
                continue;
            }

            let Some(folded) = kept.iter().find(|folded| folded.at == index) else {
                result.push((id, ir));
                continue;
            };

            let src_access = constant_for(&mut result, &mut pool, next_id, folded.src);
            let dst_access = constant_for(&mut result, &mut pool, next_id, folded.dst);
            result.push((id, IR::MemoryBarrier { src_access, dst_access }));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::*;
    use crate::{ClearValue, DomainFlag, Image, ImageInfo, PipelineId, Program, Unchecked};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

    fn transient_target(module: &mut Module) -> ValueId {
        let extent = vk::Extent2D { width: 64, height: 64 };
        module.transient_image(&ImageInfo::color_target(extent, FORMAT))
    }

    fn draw_into(module: &mut Module, target: ValueId) -> ValueId {
        module
            .begin_rendering([(target, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering::<1>()[0]
    }

    fn labels(program: &Program) -> Vec<LabelId> {
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::Label { label } => Some(*label),
                _ => None,
            })
            .collect()
    }

    fn conditional(program: &Program) -> Option<(LabelId, LabelId)> {
        program.instructions().iter().find_map(|(_, ir)| match ir {
            IR::BranchConditional {
                true_label,
                false_label,
                ..
            } => Some((*true_label, *false_label)),
            _ => None,
        })
    }

    fn phi(program: &Program) -> Vec<(ValueId, LabelId)> {
        program
            .instructions()
            .iter()
            .find_map(|(_, ir)| match ir {
                IR::Phi { incoming } => Some(incoming.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn memory_barriers(program: &Program) -> Vec<(Access, Access)> {
        let accesses = access_constants(program.instructions());
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::MemoryBarrier { src_access, dst_access } => Some((accesses[src_access], accesses[dst_access])),
                _ => None,
            })
            .collect()
    }

    fn positions(program: &Program, found: impl Fn(&IR) -> bool) -> Vec<usize> {
        program
            .instructions()
            .iter()
            .enumerate()
            .filter(|(_, (_, ir))| found(ir))
            .map(|(index, _)| index)
            .collect()
    }

    /// No clear touches another clear's target, so the transitions into all three are asked for
    /// together ahead of the first clear.
    #[test]
    fn barriers_in_front_of_unrelated_clears_gather_before_them() {
        let mut module = Module::default();
        let output = transient_target(&mut module);
        let targets = [(); 3].map(|_| transient_target(&mut module));
        let cleared = targets.map(|target| module.clear(target, crate::clear::f32::BLACK));
        // sampling keeps the clears from folding into the pass
        let sampled = cleared.map(|cleared| (cleared, Access::FragmentSampled));
        let drawn = module
            .begin_rendering(std::iter::once((output, Access::ColorRW)).chain(sampled))
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering::<4>()[0];
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        let clears = positions(&compiled, |ir| matches!(ir, IR::Clear { .. }));
        let barriers = positions(&compiled, |ir| matches!(ir, IR::ImageBarrier { .. }));
        assert_eq!(clears, (clears[0]..clears[0] + 3).collect::<Vec<_>>(), "{dump}");
        // the pass's own target touches no clear, so its transition joins the run too
        assert_eq!(barriers[..4], (clears[0] - 4..clears[0]).collect::<Vec<_>>(), "{dump}");
        // the pass still waits for the clears it samples
        assert!(barriers[4] > clears[2], "{dump}");
    }

    /// Hoisting the second pass's transition above the first pass would make the first pass wait
    /// on it, so it stays after the first pass ends.
    #[test]
    fn a_barrier_does_not_climb_above_a_pass() {
        let mut module = Module::default();
        let first = transient_target(&mut module);
        let second = transient_target(&mut module);

        let drawn = draw_into(&mut module, first);
        let both = module
            .begin_rendering([(second, Access::ColorRW), (drawn, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering::<2>()[1];
        let end = module.export(both, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        let first_end = positions(&compiled, |ir| matches!(ir, IR::EndRendering { .. }))[0];
        let into_second = positions(
            &compiled,
            |ir| matches!(ir, IR::ImageBarrier { value, .. } if *value == second),
        );
        assert_eq!(into_second.len(), 1, "{dump}");
        assert!(into_second[0] > first_end, "{dump}");
    }

    /// A memory barrier does not name the buffer it is for, so it stays behind anything that
    /// touches a buffer.
    #[test]
    fn a_memory_barrier_does_not_climb_above_a_buffer_use() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let staging = module.declare_buffer_var("staging", Access::HostWrite);
        let vertices = module.declare_buffer_var("vertices", Access::HostWrite);

        let copied = module.copy_buffer_to_image(staging, target);
        let drawn = module
            .begin_rendering([(copied, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, vertices)
            .draw(3, 1)
            .end_rendering::<1>()[0];
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        let copy = positions(&compiled, |ir| matches!(ir, IR::CopyBufferToImage { .. }))[0];
        let waits = positions(&compiled, |ir| matches!(ir, IR::MemoryBarrier { .. }));
        assert_eq!(
            memory_barriers(&compiled),
            vec![
                (Access::HostWrite, Access::CopyRead),
                (Access::HostWrite, Access::AttributeRead),
            ],
            "{dump}"
        );
        assert!(waits[0] < copy && copy < waits[1], "{dump}");
    }

    fn image_barriers(program: &Program) -> Vec<(ValueId, Access, vk::ImageLayout)> {
        let accesses = access_constants(program.instructions());
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::ImageBarrier {
                    src_access,
                    old_layout,
                    value,
                    ..
                } => Some((*value, accesses[src_access], *old_layout)),
                _ => None,
            })
            .collect()
    }

    fn pass_clears(program: &Program) -> Vec<Vec<ValueId>> {
        program
            .instructions()
            .iter()
            .filter_map(|(_, ir)| match ir {
                IR::BeginRendering { clears, .. } => Some(clears.iter().map(|(cleared, _)| *cleared).collect()),
                _ => None,
            })
            .collect()
    }

    fn clear_count(program: &Program) -> usize { positions(program, |ir| matches!(ir, IR::Clear { .. })).len() }

    fn imported_depth(module: &mut Module, format: vk::Format) -> ValueId {
        let extent = vk::Extent3D::default().width(64).height(64).depth(1);
        let image = Image::imported(vk::Image::null(), format, extent, vk::SampleCountFlags::TYPE_1);
        module.import_image(&image, vk::ImageLayout::UNDEFINED, Access::DepthStencilRW)
    }

    /// The pass overwrites the whole target, so the clear becomes its load op and the target
    /// goes straight into the pass without keeping what was in it.
    #[test]
    fn a_clear_that_opens_a_pass_becomes_its_load_op() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let cleared = module.clear(target, crate::clear::f32::BLACK);
        let drawn = draw_into(&mut module, cleared);
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        assert_eq!(clear_count(&compiled), 0, "{dump}");
        assert_eq!(pass_clears(&compiled), vec![vec![target]], "{dump}");
        let barriers = image_barriers(&compiled);
        assert_eq!(barriers.len(), 2, "one into the pass, one for the export\n{dump}");
        assert_eq!(barriers[0].2, vk::ImageLayout::UNDEFINED, "{dump}");
    }

    /// Clearing through the load op still waits on whoever wrote the depth last.
    #[test]
    fn a_folded_depth_clear_waits_on_the_last_writer() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let depth = imported_depth(&mut module, vk::Format::D32_SFLOAT);
        let cleared = module.clear(depth, ClearValue::depth(1.0));
        let drawn = module
            .begin_rendering([(target, Access::ColorRW), (cleared, Access::DepthStencilRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering::<2>()[0];
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        assert_eq!(clear_count(&compiled), 0, "{dump}");
        assert_eq!(pass_clears(&compiled), vec![vec![depth]], "{dump}");
        let into_depth = image_barriers(&compiled)
            .into_iter()
            .find(|(value, ..)| *value == depth)
            .expect("the depth is transitioned into the pass");
        assert_eq!(
            (into_depth.1, into_depth.2),
            (Access::DepthStencilRW, vk::ImageLayout::UNDEFINED),
            "{dump}"
        );
    }

    /// Each arm renders into the cleared targets, so each pass clears them itself and the clear
    /// ahead of the selection is gone.
    #[test]
    fn a_clear_that_opens_a_pass_in_both_arms_folds_into_both() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let enabled = module.declare_bool_var("enabled", true);
        let cleared = module.clear(target, crate::clear::f32::BLACK);
        let drawn = module.set_condition(enabled, move |m| draw_into(m, cleared), move |m| draw_into(m, cleared));
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        assert_eq!(clear_count(&compiled), 0, "{dump}");
        assert_eq!(pass_clears(&compiled), vec![vec![target], vec![target]], "{dump}");
    }

    /// Past the merge nothing knows whether the clear ran, so the clear has to happen on its own.
    #[test]
    fn a_clear_chosen_by_a_selection_stays() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let enabled = module.declare_bool_var("enabled", true);
        let maybe = module.set_condition(
            enabled,
            move |m| m.clear(target, crate::clear::f32::BLACK),
            move |_| target,
        );
        let drawn = draw_into(&mut module, maybe);
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(clear_count(&compiled), 1, "{}", compiled.dump());
    }

    /// Only the depth aspect is attached, so a load op would leave the stencil as it was.
    #[test]
    fn a_depth_stencil_clear_stays() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let depth = imported_depth(&mut module, vk::Format::D24_UNORM_S8_UINT);
        let cleared = module.clear(depth, ClearValue::depth(1.0));
        let drawn = module
            .begin_rendering([(target, Access::ColorRW), (cleared, Access::DepthStencilRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering::<2>()[0];
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(clear_count(&compiled), 1, "{}", compiled.dump());
    }

    /// A load op clears only the render area, which here is a corner of the target.
    #[test]
    fn a_clear_wider_than_the_render_area_stays() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let cleared = module.clear(target, crate::clear::f32::BLACK);
        let drawn = module
            .begin_rendering_area([(cleared, Access::ColorRW)], vk::Extent2D { width: 32, height: 32 })
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering::<1>()[0];
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(clear_count(&compiled), 1, "{}", compiled.dump());
    }

    /// An arm that leaves every resource where the other arm leaves it has nothing to catch up
    /// on, so the block it was written as holds only its branch and the edge goes straight to
    /// the merge.
    #[test]
    fn an_arm_with_no_work_left_in_it_is_threaded_to_the_merge() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let enabled = module.declare_bool_var("enabled", true);

        // the first pass leaves the target where a second pass over it wants it, so the arm
        // that skips the second pass is not asked to bring it anywhere
        let drawn = draw_into(&mut module, target);
        let maybe = module.set_condition(enabled, move |m| draw_into(m, drawn), move |_| drawn);
        let end = module.export(maybe, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        let (taken, skipped) = conditional(&compiled).expect("the selection branches");
        let merge = *labels(&compiled).last().expect("the merge is laid out last");
        assert_eq!(
            skipped, merge,
            "the empty arm should branch straight to the merge\n{dump}"
        );
        assert_ne!(taken, merge, "the arm holding the draw should keep its block\n{dump}");
        assert_eq!(labels(&compiled).len(), 3, "the empty arm should be gone\n{dump}");

        // the value the merge picks for that edge now arrives from the block that branched
        let entry = labels(&compiled)[0];
        let incoming = phi(&compiled);
        assert!(
            incoming.iter().any(|(value, label)| *value == drawn && *label == entry),
            "{incoming:?}\n{dump}"
        );
    }

    /// An arm the join does ask something of is not empty, so the block stays and keeps the
    /// barrier that brings the resource up to the state the merge reads it in.
    #[test]
    fn an_arm_holding_a_barrier_keeps_its_block() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let enabled = module.declare_bool_var("enabled", true);

        let drawn = module.set_condition(enabled, move |m| draw_into(m, target), move |_| target);
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        let (taken, skipped) = conditional(&compiled).expect("the selection branches");
        let merge = *labels(&compiled).last().expect("the merge is laid out last");
        assert_ne!(skipped, merge, "the arm has a barrier to record\n{dump}");
        assert_ne!(taken, skipped);
        assert_eq!(labels(&compiled).len(), 4, "{dump}");
    }

    /// Neither arm does anything, so the two edges become one and there is no longer a choice
    /// for the branch to record.
    #[test]
    fn a_selection_whose_arms_both_do_nothing_folds_away() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let enabled = module.declare_bool_var("enabled", true);

        let drawn = draw_into(&mut module, target);
        let maybe = module.set_condition(enabled, move |_| drawn, move |_| drawn);
        let end = module.export(maybe, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        assert!(conditional(&compiled).is_none(), "the branch has one way to go\n{dump}");
        assert!(
            !compiled
                .instructions()
                .iter()
                .any(|(_, ir)| matches!(ir, IR::SelectionMerge { .. })),
            "the merge outlived the selection\n{dump}"
        );
        assert_eq!(labels(&compiled).len(), 2, "{dump}");
        assert_eq!(
            phi(&compiled).len(),
            1,
            "the arms agree, so one incoming says it\n{dump}"
        );
    }

    /// Both arms are empty but carry different values, so the merge still has to be told which
    /// one ran and the arms cannot both collapse onto the same edge.
    #[test]
    fn arms_that_disagree_on_what_they_carry_keep_an_edge_each() {
        let mut module = Module::default();
        let first = transient_target(&mut module);
        let second = transient_target(&mut module);
        let enabled = module.declare_bool_var("enabled", true);

        let chosen = module.set_condition(enabled, move |_| first, move |_| second);
        let end = module.export(chosen, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        let dump = compiled.dump();

        let (taken, skipped) = conditional(&compiled).expect("the selection branches");
        assert_ne!(taken, skipped, "the merge could no longer tell the arms apart\n{dump}");
        assert_eq!(phi(&compiled).len(), 2, "{dump}");
    }

    /// Two host writes made visible with nothing recorded in between are one wait, so the pair
    /// is asked for once and the reads it covers are named together.
    #[test]
    fn barriers_from_the_same_source_are_asked_for_once() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let vertices = module.declare_buffer_var("vertices", Access::HostWrite);
        let indices = module.declare_buffer_var("indices", Access::HostWrite);

        let drawn = module
            .begin_rendering([(target, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, vertices)
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw(3, 1)
            .end_rendering::<1>()[0];
        let end = module.export(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&compiled),
            vec![(Access::HostWrite, Access::AttributeRead | Access::IndexRead)],
            "{}",
            compiled.dump()
        );
    }

    /// The second pass reads what the first one leaves behind, so where its wait sits among the
    /// commands is observable and the two waits stay apart.
    #[test]
    fn barriers_a_pass_sits_between_stay_apart() {
        let mut module = Module::default();
        let target = transient_target(&mut module);
        let vertices = module.declare_buffer_var("vertices", Access::HostWrite);
        let indices = module.declare_buffer_var("indices", Access::HostWrite);

        let first = module
            .begin_rendering([(target, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_vertex_buffer(0, vertices)
            .draw(3, 1)
            .end_rendering::<1>()[0];
        let second = module
            .begin_rendering([(first, Access::ColorRW)])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_index_buffer(indices, vk::IndexType::UINT32)
            .draw(3, 1)
            .end_rendering::<1>()[0];
        let end = module.export(second, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(&Unchecked, end).unwrap();
        assert_eq!(
            memory_barriers(&compiled),
            vec![
                (Access::HostWrite, Access::AttributeRead),
                (Access::HostWrite, Access::IndexRead),
            ],
            "{}",
            compiled.dump()
        );
    }
}
