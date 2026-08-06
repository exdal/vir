use std::fmt::Write;

use crate::graph::ir::{IR, Instr, Program};

pub fn dump(instructions: &[Instr]) -> String {
    let program = Program::new(instructions);
    let width = instructions
        .iter()
        .map(|(id, _)| id.to_string().len())
        .max()
        .unwrap_or_default();

    let count = |predicate: fn(&IR) -> bool| instructions.iter().filter(|(_, ir)| predicate(ir)).count();
    let passes = count(|ir| matches!(ir, IR::BeginRendering { .. }));
    let barriers = count(|ir| matches!(ir, IR::MemoryBarrier { .. } | IR::ImageBarrier { .. }));
    let folded = count(|ir| matches!(ir, IR::Type(_) | IR::Constant(_)));

    let mut out = String::new();
    let _ = writeln!(
        out,
        "; render graph: {} instructions, {passes} passes, {barriers} barriers",
        instructions.len()
    );
    let _ = writeln!(out, "; {folded} types and constants are printed where they are used\n");

    let mut indent = "";
    for (id, ir) in instructions {
        match ir {
            IR::Type(_) | IR::Constant(_) => continue,
            IR::BeginRendering { .. } => {
                blank_line(&mut out);
                let _ = writeln!(out, "; Pass {}", pass_header(&program, ir));
            },
            IR::EndRendering { .. } => indent = "",
            _ => {},
        }

        let _ = writeln!(out, "{:>width$} = {indent}{}", id.to_string(), ir.display(&program));
        if let Some(state) = ir.draw_state() {
            let _ = writeln!(out, "{:width$}   {indent}  ; {state}", "");
        }

        match ir {
            IR::BeginRendering { .. } => indent = "  ",
            IR::EndRendering { .. } => blank_line(&mut out),
            _ => {},
        }
    }

    out
}

fn blank_line(out: &mut String) {
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push('\n');
    }
}

fn pass_header(program: &Program, ir: &IR) -> String {
    let IR::BeginRendering {
        color_attachments,
        name,
        ..
    } = ir
    else {
        return String::new();
    };

    let targets = color_attachments
        .iter()
        .map(|attachment| {
            let mut target = match program.name(*attachment) {
                Some(name) => format!("\"{name}\""),
                None => attachment.to_string(),
            };
            if let Some(extent) = program.extent(*attachment) {
                let _ = write!(target, " {}x{}", extent.width, extent.height);
            }
            target
        })
        .collect::<Vec<_>>()
        .join(", ");

    match name {
        Some(name) => format!("\"{name}\" -> {targets}"),
        None => format!("-> {targets}"),
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::*;
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

        dump(&module.compile(rendered))
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
    fn constants_print_where_they_are_used() {
        let dump = dump_of_one_pass();
        assert!(dump.contains("draw verts=3 insts=1"), "{dump}");
        assert!(!dump.contains("= const"), "{dump}");
    }
}
