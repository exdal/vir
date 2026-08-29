use std::fmt::Write;

use crate::{
    ValueId,
    graph::ir::{IR, Instr, Symbols},
};

pub fn dump(instructions: &[Instr], bound: impl IntoIterator<Item = ValueId>) -> String {
    dump_with(instructions, bound, false)
}

pub fn dump_with(
    instructions: &[Instr], bound: impl IntoIterator<Item = ValueId>, syntax_highlighting: bool,
) -> String {
    let program = Symbols::with_bound(instructions, bound);
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
    // from the margin the globals sit at
    let mut block = 0usize;
    let mut region = 0usize;
    for (id, ir) in instructions {
        match ir {
            IR::Type(_) | IR::Constant(_) => continue,
            IR::BeginRendering { .. } | IR::BeginCompute { .. } => {
                blank_line(&mut out);
                let _ = writeln!(out, ";{} Pass {}", indent(block + region), pass_header(&program, ir));
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
        let _ = writeln!(
            out,
            "{:>width$} = {indent}{}",
            id.to_string(),
            ir.display(&program, *id)
        );
        if let Some(state) = ir.draw_state() {
            let _ = writeln!(out, ";{:width$}   {indent}   {state}", "");
        }

        match ir {
            IR::BeginRendering { .. } | IR::BeginCompute { .. } => region += 1,
            IR::EndRendering { .. } | IR::EndCompute { .. } => blank_line(&mut out),
            IR::Label { .. } => block = 1,
            _ => {},
        }
    }

    match syntax_highlighting {
        true => highlight(&out),
        false => out,
    }
}

mod color {
    pub const RESET: &str = "\x1b[0m";
    pub const COMMENT: &str = "\x1b[90m";
    pub const OPCODE: &str = "\x1b[1;34m";
    pub const ID: &str = "\x1b[36m";
    pub const VARIABLE: &str = "\x1b[35m";
    pub const STRING: &str = "\x1b[32m";
    pub const NUMBER: &str = "\x1b[33m";
    pub const KEY: &str = "\x1b[37m";
}

fn highlight(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for line in text.lines() {
        highlight_line(&mut out, line);
        out.push('\n');
    }
    out
}

fn highlight_line(out: &mut String, line: &str) {
    if line.trim_start().starts_with(';') {
        out.push_str(color::COMMENT);
        highlight_operands(out, line, color::COMMENT);
        out.push_str(color::RESET);
        return;
    }

    let Some((id, body)) = line.split_once(" = ") else {
        highlight_operands(out, line, "");
        return;
    };

    let padding = id.len() - id.trim_start().len();
    out.push_str(&id[..padding]);
    token(out, color::ID, id.trim_start(), "");
    out.push_str(" = ");

    let indent = body.len() - body.trim_start().len();
    let opcode = body[indent..]
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .map_or(body.len(), |end| indent + end);
    out.push_str(&body[..indent]);
    token(out, color::OPCODE, &body[indent..opcode], "");
    highlight_operands(out, &body[opcode..], "");
}

fn highlight_operands(out: &mut String, text: &str, base: &str) {
    // a comment is one flat color, so a key inside one is not lifted out of it
    let key = match base {
        color::COMMENT => color::COMMENT,
        _ => color::KEY,
    };

    let mut rest = text;
    while let Some(head) = rest.chars().next() {
        let taken = match head {
            '"' => {
                let end = rest[1..].find('"').map_or(rest.len(), |end| end + 2);
                token(out, color::STRING, &rest[..end], base);
                end
            },
            '%' | '$' | '#' => {
                let end = word_end(&rest[1..]) + 1;
                let color = match head {
                    '$' => color::VARIABLE,
                    '#' => color::NUMBER,
                    _ => color::ID,
                };
                token(out, color, &rest[..end], base);
                end
            },
            _ if head.is_ascii_digit() => {
                let end = number_end(rest);
                token(out, color::NUMBER, &rest[..end], base);
                end
            },
            _ if head.is_alphabetic() || head == '_' => {
                let end = word_end(rest);
                match rest[end..].starts_with('=') {
                    true => token(out, key, &rest[..end], base),
                    false => out.push_str(&rest[..end]),
                }
                end
            },
            _ => {
                out.push(head);
                head.len_utf8()
            },
        };
        rest = &rest[taken..];
    }
}

fn word_end(text: &str) -> usize {
    text.find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(text.len())
}

fn number_end(text: &str) -> usize {
    let mut end = 0;
    for (index, c) in text.char_indices() {
        let digits = || text[index + 1..].starts_with(|next: char| next.is_ascii_digit());
        if !(c.is_ascii_digit() || c == '.' || (c == 'x' && digits())) {
            break;
        }
        end = index + c.len_utf8();
    }
    end
}

fn token(out: &mut String, color: &str, text: &str, base: &str) {
    if color == base {
        out.push_str(text);
        return;
    }

    out.push_str(color);
    out.push_str(text);
    out.push_str(color::RESET);
    out.push_str(base);
}

fn indent(depth: usize) -> String { "  ".repeat(depth) }

fn blank_line(out: &mut String) {
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push('\n');
    }
}

fn pass_header(program: &Symbols, ir: &IR) -> String {
    let (targeted, name) = match ir {
        IR::BeginRendering {
            color_attachments,
            name,
            ..
        } => (color_attachments.clone(), name),
        IR::BeginCompute { declared_access, name } => (declared_access.iter().map(|(id, _)| *id).collect(), name),
        _ => return String::new(),
    };

    let targets = targeted
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
    use ash::{vk, vk::Handle};

    use super::super::analysis::Declared;
    use crate::{BlendPreset, Image, ImageAttachment, ImageInfo, Module, PipelineId, Rect2D, Unchecked};

    const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

    fn strip_ansi(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find('\x1b') {
            out.push_str(&rest[..start]);
            let end = rest[start..].find('m').expect("an escape should end in m");
            rest = &rest[start + end + 1..];
        }
        out.push_str(rest);
        out
    }

    fn dump_of_one_pass() -> String { dump_of_one_pass_with(false) }

    fn dump_of_one_pass_with(syntax_highlighting: bool) -> String {
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

        module
            .compile(&Unchecked, rendered)
            .unwrap()
            .dump_with(syntax_highlighting)
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
        assert!(
            dump.lines()
                .any(|line| line.starts_with(';') && line.contains("Pass \"triangle\" -> \"target\" 64x32")),
            "{dump}"
        );
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
            .bind_compute_pipeline(PipelineId(0))
            .write(target)
            .push_constants(&1u32)
            .dispatch(8, 4, 1)
            .end_compute();

        let dump = module.compile(&Unchecked, end).unwrap().dump();
        assert!(dump.contains("Pass \"blur\" -> \"storage\" 64x32"), "{dump}");
        assert!(
            dump.contains("  dispatch groups_x=8 groups_y=4 groups_z=1 pipeline=#0"),
            "{dump}"
        );
        assert!(dump.contains("push_constants=[0..4]"), "{dump}");
        assert!(dump.contains("end_compute"), "{dump}");
    }

    #[test]
    fn constants_print_where_they_are_used() {
        let dump = dump_of_one_pass();
        assert!(dump.contains("draw verts=3 insts=1"), "{dump}");
        assert!(!dump.contains("= const"), "{dump}");
    }

    /// A callback is the one variable whose value the dump cannot show, so a debug build prints
    /// where it was declared instead. A release build carries no location at all.
    #[test]
    fn a_callback_variable_prints_where_it_was_declared() {
        let extent = vk::Extent3D::default().width(64).height(32).depth(1);
        let mut module = Module::default();
        let target = module.transient_image(&crate::ImageInfo {
            extent,
            format: FORMAT,
            ..Default::default()
        });

        let line = line!() + 1;
        let body = module.declare_callback_var("draws");
        let end = module
            .begin_rendering(&[target])
            .bind_graphics_pipeline(PipelineId(0))
            .record_from(body)
            .end_rendering();

        let dump = module.compile(&Unchecked, end).unwrap().dump();
        let expected = match cfg!(debug_assertions) {
            true => format!("var \"draws\" Callback({}:{line}) slot=", file!()),
            false => "var \"draws\" Callback slot=".to_string(),
        };
        assert!(dump.contains(&expected), "{expected}\n{dump}");
    }

    /// Highlighting only paints what the plain dump already says, so taking the escapes back
    /// out has to leave the two identical.
    #[test]
    fn highlighting_changes_nothing_but_the_color() {
        let colored = dump_of_one_pass_with(true);
        assert!(colored.contains('\x1b'), "{colored}");
        assert_eq!(strip_ansi(&colored), dump_of_one_pass());
    }

    #[test]
    fn highlighting_paints_ids_opcodes_and_operands() {
        let colored = dump_of_one_pass_with(true);
        assert!(colored.contains("\x1b[36m%3\x1b[0m = "), "{colored}");
        assert!(colored.contains("\x1b[33m64x32\x1b[0m"), "{colored}");
        assert!(colored.contains("\x1b[1;34mdraw\x1b[0m"), "{colored}");
        assert!(colored.contains("\x1b[32m\"target\"\x1b[0m"), "{colored}");
        assert!(colored.contains("\x1b[37mverts\x1b[0m=\x1b[33m3\x1b[0m"), "{colored}");
        assert!(colored.lines().any(|line| line.starts_with("\x1b[90m;")), "{colored}");
    }

    /// A comment is one color from end to end, so an operand painted inside it has to hand the
    /// comment's color back rather than reset to the terminal's default.
    #[test]
    fn a_highlighted_comment_stays_a_comment_after_an_operand() {
        let colored = dump_of_one_pass_with(true);
        let header = colored
            .lines()
            .find(|line| line.contains("Pass "))
            .expect("the pass header should be dumped");
        assert!(header.contains("\x1b[32m\"triangle\"\x1b[0m\x1b[90m"), "{header}");
        assert!(header.ends_with("\x1b[0m"), "{header}");
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

        let animate = module.declare_bool_var("animate", true);
        let color = module.declare_clear_var("hue", crate::clear::f32::BLACK);
        let cleared = module.set_condition(animate, |m| m.clear_from(target, color), |_| target);
        let end = module.release(cleared, crate::Access::BlitRead, crate::DomainFlag::Graphics);

        let dump = module.compile(&Unchecked, end).unwrap().dump();
        assert!(dump.contains("4 blocks"), "{dump}");
        assert!(dump.contains("var \"animate\" Bool"), "{dump}");
        assert!(dump.contains("= label 0:"), "{dump}");
        assert!(dump.contains("selection_merge label 1"), "{dump}");
        assert!(dump.contains("branch_cond %5(animate) -> label 2, label 3"), "{dump}");
        assert!(dump.contains("branch label 1"), "{dump}");
        assert!(dump.contains("color=%6(hue)"), "{dump}");
        assert!(dump.contains("phi ["), "{dump}");
        assert!(dump.contains("\n%12 =   selection_merge"), "{dump}");
    }
    /// A descriptor write is an instruction of its region, so it reads back as one of its lines.
    #[test]
    fn a_descriptor_write_is_a_line_of_its_region() {
        let target = vk::Extent2D::default().width(4).height(4);
        let mut module = Module::default();
        let attachment = module.transient_image(&ImageInfo::color_target(target, FORMAT));
        let texture = module.transient_image(&ImageInfo::color_target(target, FORMAT));
        module.set_name(texture, "source");

        let end = module
            .begin_rendering(&[attachment])
            .bind_graphics_pipeline(PipelineId(0))
            .bind_texture(1, 2, texture, vk::Sampler::from_raw(7))
            .draw(3, 1)
            .end_rendering();

        let bindings = Declared::new(&[(1, 2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]);
        let dump = module.compile(&bindings, end).unwrap().dump();
        assert!(dump.contains("write_descriptor set=1 binding=2"), "{dump}");
        assert!(dump.contains("combined_image_sampler %5(source)"), "{dump}");
        assert!(dump.contains("access=FragmentSampled"), "{dump}");
    }
}
