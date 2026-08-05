use std::{env, error::Error, path::PathBuf};

use shader_slang as slang;
use slang::Downcast;

const ENTRY_POINTS: [(&str, &str); 3] = [
    ("vs_main", "triangle.vert.spv"),
    ("fs_main", "triangle.frag.spv"),
    ("vs_buffer", "triangle_buffer.vert.spv"),
];

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=shaders/triangle.slang");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let shader_dir = manifest_dir.join("shaders");

    let global_session = slang::GlobalSession::new().ok_or("failed to create Slang global session")?;

    let options = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::High)
        .matrix_layout_row(true);
    let targets = [slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("glsl_450"))];

    // `load_module` resolves by name against the search paths, so point Slang at the
    // crate's shader directory instead of relying on the working directory.
    let search_path = std::ffi::CString::new(shader_dir.to_str().ok_or("non-UTF-8 shader path")?)?;
    let search_paths = [search_path.as_ptr()];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&options);
    let session = global_session
        .create_session(&session_desc)
        .ok_or("failed to create Slang session")?;

    let module = session.load_module("triangle.slang")?;

    // Link the module together with every entry point once, so they share a single
    // layout, then pull the per-stage SPIR-V back out.
    let mut components = vec![module.downcast().clone()];
    for (entry_point, _) in ENTRY_POINTS {
        let entry_point = module
            .find_entry_point_by_name(entry_point)
            .ok_or_else(|| format!("entry point `{entry_point}` not found in triangle.slang"))?;
        components.push(entry_point.downcast().clone());
    }

    let program = session.create_composite_component_type(&components)?;
    let linked = program.link()?;

    for (index, (_, file_name)) in ENTRY_POINTS.iter().enumerate() {
        let code = linked.entry_point_code(index as i64, 0)?;
        std::fs::write(out_dir.join(file_name), code.as_slice())?;
    }

    Ok(())
}
