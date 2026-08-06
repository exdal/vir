# vir-examples

One executable per example, each one built on the same harness so that what is left in the file
is only the thing the example is about.

```
cargo run --example triangle
cargo run --example vertex_buffer
cargo run --example offscreen
cargo run --example texture
cargo run --example compute
cargo run --example egui
```

| Example | What it shows |
| --- | --- |
| `triangle` | A draw with no buffers at all: geometry from `SV_VertexID`, push constants, blend presets, and the same pipeline once with baked and once with dynamic viewport and scissor. |
| `vertex_buffer` | A quad uploaded once into a persistent buffer and drawn indexed, next to a triangle rebuilt every frame out of the frame allocator. |
| `offscreen` | A frame as a chain: a persistent offscreen target, a transient image the graph owns for the length of the frame, two blits, and a second root that hands the target back in a known layout. |
| `texture` | A PNG staged and copied onto the GPU once at startup, then sampled through the graph's bindless table by pushing a slot index. |
| `compute` | Geometry that never exists on the CPU: two dispatches reaching their buffers through device addresses, chained so that the second reads what the first wrote and the draw's vertex input waits on both. |
| `egui` | The overlay with nothing under it. |

Every example draws its controls through the same egui overlay, so the backend in
`src/egui_pass.rs` is exercised by all of them; between the font atlas and the meshes it is
where texture patches, indexed draws, per-draw scissors and bindless sampling all get run.

## Notes

- `cargo test` covers the examples too: each one asserts that reflection reads its own shaders
  back as the layouts and push constant blocks the Rust side assumes.
- `VIR_DUMP_IR=1` prints the compiled graph for the first frame that has anything in it.
