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

## A recorded pass

```rust
let target = frame.module.clear(frame.swapchain_image, BACKGROUND);

let target = frame
    .module
    .begin_rendering([(target, Access::ColorRW)])
    .with_name("triangle")
    .bind_graphics_pipeline(pipeline)
    .set_viewport(0, Rect2D::framebuffer())
    .set_scissor(0, Rect2D::framebuffer())
    .broadcast_color_blend(BlendPreset::Off)
    .push_constants(&push)
    .draw(3, 1)
    .end_rendering();
```

Every scalar descriptor type reflected from SPIR-V is ordinary pass state: samplers, sampled and
storage images, input attachments, uniform and storage buffers, texel buffers, and acceleration
structures. Binding methods describe the payload, while the pipeline active at each consuming draw
or dispatch supplies its exact Vulkan descriptor type and statically reachable access. For example,
`bind_image` covers sampled images, storage images, and input attachments; `bind_buffer` covers both
uniform and storage buffers:

```rust
let target = frame
    .module
    .begin_rendering([(target, Access::ColorRW)])
    .bind_graphics_pipeline(pipeline)
    .bind_texture(0, 3, texture, sampler)
    .bind_buffer(0, 4, material)
    .draw(3, 1)
    .end_rendering();
```

`bind_buffer_range` binds an explicit byte range, and `bind_texel_buffer` similarly covers both
uniform and storage texel buffers. Sampler-only, combined image/sampler, texel-buffer, and
acceleration-structure bindings remain separate because their Vulkan payloads differ. A descriptor
write may precede the pipeline bind, but a reflected pipeline must be active when a draw or dispatch
consumes it. If the same standing write is consumed by multiple pipelines, they must agree on its
descriptor type; rebind between pipelines when they do not.

Texel-buffer views and acceleration structures remain caller-owned; their binding calls also take
the backing buffer value so the graph can synchronize it. The caller must keep raw sampler,
buffer-view, and acceleration-structure handles alive through execution.

Bindless is opt-in per pipeline. Pass the caller-owned layout and set at the index it should
occupy; `vir` splices that layout into the reflected pipeline layout and binds the set without
updating or destroying it. The caller is also responsible for enabling any descriptor-indexing
features required by that layout and its shaders:

```rust
let info = GraphicsPipelineInfo::new()
    .with_shader(vertex_spirv)
    .with_shader(fragment_spirv)
    .with_bindless_set(2, bindless_layout, bindless_set);
```

Images reached through an external bindless set still need an entry in `begin_rendering` or
`begin_compute` when the graph must synchronize them, because their descriptor contents are
intentionally opaque to the IR. One resource-array value can stand for the whole set; the access
is applied to every element while the array remains a single attachment entry:

```rust
module
    .begin_rendering([
        (target, Access::ColorRW),
        (bindless_images, Access::FragmentRead),
    ])
    // ...
    .end_rendering();
```

A compute pass declares its accesses, and the graph barriers the dispatches against each other and
against any draw that later reads what they wrote:

```rust
frame
    .module
    .begin_compute([(instances, Access::ComputeWrite)])
    .bind_compute_pipeline(place)
    .push_constants(&push)
    .dispatch_invocations(triangle_count, 1, 1)
    .end_compute();
```

## Examples

The examples live in `vir-examples`. Their shaders are written in [Slang](https://shader-slang.org)
and compiled to SPIR-V at build time by `vir-examples/build.rs`.

```sh
cargo run -p vir-examples --example triangle
cargo run -p vir-examples --example vertex_buffer
cargo run -p vir-examples --example texture
cargo run -p vir-examples --example descriptors
cargo run -p vir-examples --example compute
cargo run -p vir-examples --example offscreen
cargo run -p vir-examples --example deferred
cargo run -p vir-examples --example egui
```

Each windowed example draws its controls into an egui overlay that the harness owns; the example
itself only records its pipelines, resources, and passes.

## Using it in your own project

`vir` is not published to crates.io. Depend on it from git:

```toml
[dependencies]
vir = { git = "https://github.com/exdal/vir.git", package = "vir" }
```

It brings in `ash`, so you supply the instance, physical device, and logical device (see
`vir-examples/src/device_builder.rs` for one way to build them), then create a `Context` and a
`RenderGraph` on top.

## License

MIT. See [LICENSE](LICENSE).

This project is heavily inspired by the [vuk](https://github.com/martty/vuk). Please check it out.
