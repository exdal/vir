# vir

Vulkan Intermediate Representation: a render graph for modern Vulkan, built on [`ash`](https://github.com/ash-rs/ash).

Record a frame as a list of passes. `vir` lowers it to an SSA-style instruction list, figures out
which passes depend on which, and inserts the pipeline barriers and layout transitions itself.

> Requires Vulkan 1.3 (dynamic rendering, buffer device address, synchronization2) and a nightly
> Rust toolchain (edition 2024).

## What it does

- **Render graph.** Passes are recorded into a `Module` as an IR of `ValueId` handles. Each pass
  declares the resources it touches and how (`Access`), and the graph derives the barriers between
  them. Reordering or dropping a pass never means rewriting a barrier by hand.
- **Automatic synchronization.** Read/write/read-write accesses on buffers and images are what the
  graph reads to insert the right barriers, layout transitions, and queue ownership transfers.
- **Reflection-driven pipelines.** Vertex layout, push constant ranges, and descriptor bindings
  come out of SPIR-V reflection, so a pipeline is declared from little more than its shader blobs.
- **Dynamic rendering.** No render pass or framebuffer objects; passes use `begin_rendering`
  directly, with viewport, scissor, blend, and rasterization as pipeline or dynamic state.
- **Compute and transfer.** Compute passes, blits, and buffer-to-image copies live in the same
  graph as graphics, so cross-domain dependencies are resolved the same way.
- **Allocators.** Frame and persistent allocators over
  [`gpu-allocator`](https://github.com/Traverse-Research/gpu-allocator), with buffer device address
  enabled.

## A pass, recorded

```rust
let target = frame.module.clear(frame.swapchain_image, BACKGROUND);

let target = frame
    .module
    .begin_rendering(&[target])
    .with_name("triangle")
    .bind_graphics_pipeline(pipeline)
    .set_viewport(0, Rect2D::framebuffer())
    .set_scissor(0, Rect2D::framebuffer())
    .broadcast_color_blend(BlendPreset::Off)
    .push_constants(&push)
    .draw(3, 1)
    .end_rendering();
```

A compute pass declares its accesses, and the graph barriers the dispatches against each other and
against any draw that later reads what they wrote:

```rust
frame
    .module
    .begin_compute()
    .bind_pipeline(place)
    .write(instances)
    .push_constants(&push)
    .dispatch_invocations(triangle_count, 1, 1)
    .end_compute();
```

## Layout

```
vir/               the library
  src/
    graph/         module, IR, render graph, passes, attachments
    resource/      buffers, images, pipelines, shaders, descriptors, swapchain
    context/       device context, command queues and buffers, access
    allocator/     frame and persistent allocators
vir-examples/      runnable examples and their Slang shaders
```

## Examples

The examples live in `vir-examples`. Their shaders are written in [Slang](https://shader-slang.org)
and compiled to SPIR-V at build time by `vir-examples/build.rs`.

```sh
cargo run -p vir-examples --example triangle
cargo run -p vir-examples --example vertex_buffer
cargo run -p vir-examples --example texture
cargo run -p vir-examples --example compute
cargo run -p vir-examples --example offscreen
cargo run -p vir-examples --example egui
```

Each windowed example draws its controls into an egui overlay that the harness owns; the example
itself only records its pipelines, resources, and passes.

## Using it in your own project

`vir` is not published to crates.io. Depend on it from git:

```toml
[dependencies]
vir = { git = "https://github.com/<you>/vir.git", package = "vir" }
```

It brings in `ash`, so you supply the instance, physical device, and logical device (see
`vir-examples/src/device_builder.rs` for one way to build them), then create a `Context` and a
`RenderGraph` on top.

## License

MIT. See [LICENSE](LICENSE).
