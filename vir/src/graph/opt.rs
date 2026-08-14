use std::{collections::HashSet, ops::Range};

use crate::{IR, LabelId, Module, ValueId, graph::ir};

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

impl Module {
    pub(crate) fn simplify_cfg(&self, mut nodes: Vec<ir::Instr>) -> Vec<ir::Instr> {
        loop {
            let mut changed = wire_empty_block(&mut nodes);
            changed |= fold_equal_targets(&mut nodes);

            if !changed {
                return nodes;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::*;
    use crate::{Access, DomainFlag, ImageInfo, PipelineId, Program};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

    fn transient_target(module: &mut Module) -> ValueId {
        let extent = vk::Extent2D { width: 64, height: 64 };
        module.transient_image(&ImageInfo::color_target(extent, FORMAT))
    }

    fn draw_into(module: &mut Module, target: ValueId) -> ValueId {
        module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .draw(3, 1)
            .end_rendering()
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
        let end = module.release(maybe, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(end);
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
        let end = module.release(drawn, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(end);
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
        let end = module.release(maybe, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(end);
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
        let end = module.release(chosen, Access::BlitRead, DomainFlag::Graphics);

        let compiled = module.compile(end);
        let dump = compiled.dump();

        let (taken, skipped) = conditional(&compiled).expect("the selection branches");
        assert_ne!(taken, skipped, "the merge could no longer tell the arms apart\n{dump}");
        assert_eq!(phi(&compiled).len(), 2, "{dump}");
    }
}
