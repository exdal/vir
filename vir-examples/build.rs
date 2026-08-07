use std::{env, error::Error, path::PathBuf};

use shader_slang as slang;
use slang::Downcast;

/// Every Slang module, with the entry points to pull out of it and the file each one is
/// written to. A module is linked on its own, so entry point names only have to be unique
/// within their module.
const MODULES: [(&str, &[(&str, &str)]); 7] = [
    (
        "deferred_geometry.slang",
        &[
            ("vs_main", "deferred_geometry.vert.spv"),
            ("fs_main", "deferred_geometry.frag.spv"),
        ],
    ),
    (
        "deferred_lighting.slang",
        &[
            ("vs_main", "deferred_lighting.vert.spv"),
            ("fs_main", "deferred_lighting.frag.spv"),
        ],
    ),
    (
        "compute.slang",
        &[
            ("cs_place", "compute.place.spv"),
            ("cs_expand", "compute.expand.spv"),
            ("vs_main", "compute.vert.spv"),
            ("fs_main", "compute.frag.spv"),
        ],
    ),
    (
        "triangle.slang",
        &[("vs_main", "triangle.vert.spv"), ("fs_main", "triangle.frag.spv")],
    ),
    (
        "vertex_buffer.slang",
        &[
            ("vs_main", "vertex_buffer.vert.spv"),
            ("fs_main", "vertex_buffer.frag.spv"),
        ],
    ),
    (
        "texture.slang",
        &[("vs_main", "texture.vert.spv"), ("fs_main", "texture.frag.spv")],
    ),
    (
        "egui.slang",
        &[("vs_main", "egui.vert.spv"), ("fs_main", "egui.frag.spv")],
    ),
];

fn main() -> Result<(), Box<dyn Error>> {
    for (module, _) in MODULES {
        println!("cargo:rerun-if-changed=shaders/{module}");
    }
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

    for (module_name, entry_points) in MODULES {
        let module = session.load_module(module_name)?;

        // Link the module together with every entry point once, so they share a single
        // layout, then pull the per-stage SPIR-V back out.
        let mut components = vec![module.downcast().clone()];
        for (entry_point, _) in entry_points {
            let entry_point = module
                .find_entry_point_by_name(entry_point)
                .ok_or_else(|| format!("entry point `{entry_point}` not found in {module_name}"))?;
            components.push(entry_point.downcast().clone());
        }

        let program = session.create_composite_component_type(&components)?;
        let linked = program.link()?;

        for (index, (_, file_name)) in entry_points.iter().enumerate() {
            let code = linked.entry_point_code(index as i64, 0)?;
            std::fs::write(out_dir.join(file_name), code.as_slice())?;
        }
    }

    Ok(())
}
