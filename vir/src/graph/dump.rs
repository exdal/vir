use std::fmt::Write;

use crate::graph::ir::{IR, Instr, Symbols};

pub fn dump(instructions: &[Instr]) -> String {
    let program = Symbols::new(instructions);
    let width = instructions
        .iter()
        .map(|(id, _)| id.to_string().len())
        .max()
        .unwrap_or_default();

    let count = |predicate: fn(&IR) -> bool| instructions.iter().filter(|(_, ir)| predicate(ir)).count();
    let passes = count(|ir| matches!(ir, IR::BeginRendering { .. } | IR::BeginCompute { .. }));
    let barriers = count(|ir| matches!(ir, IR::MemoryBarrier { .. } | IR::ImageBarrier { .. }));
    let folded = count(|ir| matches!(ir, IR::Type(_) | IR::Constant(_)));
    let blocks = count(|ir| matches!(ir, IR::Label { .. }));

    let mut out = String::new();
    let _ = writeln!(
        out,
        "; render graph: {} instructions, {passes} passes, {barriers} barriers, {folded} types, {blocks} blocks",
        instructions.len()
    );

    // a block is a level of its own, so a pass inside one indents from the block rather than
    // from the margin the hoisted instructions sit at
    let mut block = 0usize;
    let mut region = 0usize;
    for (id, ir) in instructions {
        match ir {
            IR::Type(_) | IR::Constant(_) => continue,
            IR::BeginRendering { .. } | IR::BeginCompute { .. } => {
                blank_line(&mut out);
                let _ = writeln!(out, "{}; Pass {}", indent(block + region), pass_header(&program, ir));
            },
            IR::EndRendering { .. } | IR::EndCompute { .. } => region = region.saturating_sub(1),
            IR::Label { .. } => {
                blank_line(&mut out);
                block = 0;
                region = 0;
            },
            _ => {},
        }

        let indent = indent(block + region);
        let _ = writeln!(out, "{:>width$} = {indent}{}", id.to_string(), ir.display(&program));
        if let Some(state) = ir.draw_state() {
            let _ = writeln!(out, "{:width$}   {indent}  ; {state}", "");
        }

        match ir {
            IR::BeginRendering { .. } | IR::BeginCompute { .. } => region += 1,
            IR::EndRendering { .. } | IR::EndCompute { .. } => blank_line(&mut out),
            IR::Label { .. } => block = 1,
            _ => {},
        }
    }

    out
}

fn indent(depth: usize) -> String { "  ".repeat(depth) }

fn blank_line(out: &mut String) {
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push('\n');
    }
}

fn pass_header(program: &Symbols, ir: &IR) -> String {
    let (resources, name) = match ir {
        IR::BeginRendering {
            color_attachments,
            name,
            ..
        } => (color_attachments.clone(), name),
        IR::BeginCompute { resources, name } => (resources.iter().map(|(id, _)| *id).collect(), name),
        _ => return String::new(),
    };

    let targets = resources
        .iter()
        .map(|resource| {
            let mut target = match program.name(*resource) {
                Some(name) => format!("\"{name}\""),
                None => resource.to_string(),
            };
            if let Some(extent) = program.extent(*resource) {
                let _ = write!(target, " {}x{}", extent.width, extent.height);
            }
            target
        })
        .collect::<Vec<_>>()
        .join(", ");

    match (name, targets.is_empty()) {
        (Some(name), true) => format!("\"{name}\""),
        (Some(name), false) => format!("\"{name}\" -> {targets}"),
        (None, true) => String::new(),
        (None, false) => format!("-> {targets}"),
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use crate::{BlendPreset, Image, ImageAttachment, Module, PipelineId, Rect2D};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

    fn dump_of_one_pass() -> String {
        let extent = vk::Extent3D::default().width(64).height(32).depth(1);
        let attachment = ImageAttachment::new(
            Image::imported(vk::Image::null(), FORMAT, extent, vk::SampleCountFlags::TYPE_1),
            FORMAT,
            extent,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageLayout::UNDEFINED,
        );

        let mut module = Module::default();
        let target = module.import_attachment(&attachment);
        module.set_name(target, "target");

        let rendered = module
            .begin_rendering(&[target])
            .with_name("triangle")
            .bind_graphics_pipeline(PipelineId(0))
            .set_viewport(0, Rect2D::framebuffer())
            .broadcast_color_blend(BlendPreset::Off)
            .draw(3, 1)
            .end_rendering();

        module.compile(rendered).dump()
    }

    /// A program that never branches is still a program with a block: the entry one, which
    /// everything it records belongs to, ending in the return it was given by default.
    #[test]
    fn a_program_without_branches_is_one_entry_block() {
        let dump = dump_of_one_pass();
        assert!(dump.contains("1 blocks"), "{dump}");
        assert!(dump.contains("= label 0:"), "{dump}");
        assert_eq!(dump.matches("= label ").count(), 1, "{dump}");
        assert!(!dump.contains("branch"), "{dump}");
        assert!(dump.trim_end().ends_with("return"), "{dump}");
    }

    #[test]
    fn a_named_pass_heads_the_region_it_opened() {
        let dump = dump_of_one_pass();
        assert!(dump.contains("; Pass \"triangle\" -> \"target\" 64x32"), "{dump}");
    }

    #[test]
    fn a_resource_operand_carries_the_name_it_was_given() {
        let dump = dump_of_one_pass();
        assert!(dump.contains("image \"target\" 64x32"), "{dump}");
        assert!(dump.contains("color=[%"), "{dump}");
        assert!(dump.contains("(target)"), "{dump}");
    }

    #[test]
    fn a_compute_region_heads_and_indents_like_a_rendering_one() {
        let extent = vk::Extent3D::default().width(64).height(32).depth(1);
        let mut module = Module::default();
        let target = module.transient_image(&crate::ImageInfo {
            extent,
            format: FORMAT,
            ..Default::default()
        });
        module.set_name(target, "storage");

        let end = module
            .begin_compute()
            .with_name("blur")
            .bind_pipeline(PipelineId(0))
            .write(target)
            .push_constants(&1u32)
            .dispatch(8, 4, 1)
            .end_compute();

        let dump = module.compile(end).dump();
        assert!(dump.contains("; Pass \"blur\" -> \"storage\" 64x32"), "{dump}");
        assert!(
            dump.contains("  dispatch groups_x=8 groups_y=4 groups_z=1 pipeline=#0"),
            "{dump}"
        );
        assert!(dump.contains("; push_constants=[0..4]"), "{dump}");
        assert!(dump.contains("end_compute"), "{dump}");
    }

    #[test]
    fn constants_print_where_they_are_used() {
        let dump = dump_of_one_pass();
        assert!(dump.contains("draw verts=3 insts=1"), "{dump}");
        assert!(!dump.contains("= const"), "{dump}");
    }

    /// A variable has no value to fold in, so it prints as the name it was given wherever it
    /// is used, and the blocks around a branch are named rather than run together.
    #[test]
    fn a_selection_prints_its_blocks_and_the_variable_it_branches_on() {
        let extent = vk::Extent3D::default().width(64).height(32).depth(1);
        let mut module = Module::default();
        let target = module.transient_image(&crate::ImageInfo {
            extent,
            format: FORMAT,
            ..Default::default()
        });
        module.set_name(target, "target");

        let animate = module.variable_bool("animate", true);
        let color = module.variable_clear("hue", crate::clear::f32::BLACK);
        let cleared = module.if_else(animate, |m| m.clear_from(target, color), |_| target);
        let end = module.release(cleared, crate::Access::BlitRead, crate::DomainFlag::Graphics);

        let dump = module.compile(end).dump();
        assert!(dump.contains("4 blocks"), "{dump}");
        assert!(dump.contains("var \"animate\" Bool"), "{dump}");
        assert!(dump.contains("= label 0:"), "{dump}");
        assert!(dump.contains("selection_merge label 1"), "{dump}");
        assert!(dump.contains("branch_cond $animate -> label 2, label 3"), "{dump}");
        assert!(dump.contains("branch label 1"), "{dump}");
        assert!(dump.contains("color=$hue"), "{dump}");
        assert!(dump.contains("phi ["), "{dump}");
        assert!(dump.contains("\n%12 =   selection_merge"), "{dump}");
    }
}
