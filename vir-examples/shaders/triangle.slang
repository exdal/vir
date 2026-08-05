// A vertex-buffer-less triangle: positions and colors are baked into the shader and
// selected by SV_VertexID, so the draw needs no bound geometry at all.
//
// Clip space here is Vulkan's: +Y points down, matching the unflipped viewport that
// `vir` sets up in `CommandBuffer::begin_rendering`.

struct VSOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

static const float2 POSITIONS[3] = {
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5),
};

static const float3 COLORS[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VSOutput vs_main(uint vertex_id: SV_VertexID) {
    VSOutput output;
    output.position = float4(POSITIONS[vertex_id], 0.0, 1.0);
    output.color = COLORS[vertex_id];
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input): SV_Target {
    return float4(input.color, 1.0);
}
