# vir-examples

One executable per example, each one built on the same harness so that what is left in the file
is only the thing the example is about.

```
cargo run --example triangle
cargo run --example vertex_buffer
cargo run --example offscreen
cargo run --example texture
cargo run --example compute
cargo run --example deferred
cargo run --example egui
```

| Example | What it shows |
| --- | --- |
| `triangle` | A draw with no buffers at all: geometry from `SV_VertexID`, push constants, blend presets, and the same pipeline once with baked and once with dynamic viewport and scissor. |
| `vertex_buffer` | A quad uploaded once into a persistent buffer and drawn indexed, next to a triangle rebuilt every frame out of the frame allocator. |
| `offscreen` | A frame as a chain: an offscreen target the graph owns for the run, a second one sized by a variable so the downscale costs no recompile, and three blits back onto the swapchain. |
| `texture` | A PNG staged and copied onto the GPU once at startup, then sampled through the graph's bindless table by pushing a slot index. |
| `compute` | Geometry that never exists on the CPU: two dispatches reaching their buffers through device addresses, chained so that the second reads what the first wrote and the draw's vertex input waits on both. |
| `deferred` | A simple deferred renderer that uses Damaged Helmet model. It's a great example to show how multiple attachments per pass is used. |
| `egui` | The overlay with nothing under it. |

Every example draws its controls through the same egui overlay, so the backend in
`src/egui_pass.rs` is exercised by all of them; between the font atlas and the meshes it is
where texture patches, indexed draws, per-draw scissors and bindless sampling all get run.

## Notes

- Examples are written with the help of LLMs to better document code. So it would be more understandable because I generally don't write comments myself.
- `cargo test` covers the examples too: each one asserts that reflection reads its own shaders
  back as the layouts and push constant blocks the Rust side assumes.
- `VIR_DUMP_IR=1` prints the compiled graph for the frame, which is the whole of what runs.
