struct PushConstants {
    float2 scale;
    uint texture_index;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

/// The graph's bindless table. A variable-count combined image sampler array is what it
/// recognises as the slot textures are written into, so `texture_index` is all a draw needs
/// to name one.
[[vk::binding(0, 0)]]
Sampler2D textures[];

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

/// A quad as a triangle strip, so the draw is four vertices and no buffer at all.
static const float2 CORNERS[4] = {
    float2(-1.0, -1.0),
    float2(1.0, -1.0),
    float2(-1.0, 1.0),
    float2(1.0, 1.0),
};

[shader("vertex")]
VSOutput vs_main(uint vertex_id: SV_VertexID) {
    float2 corner = CORNERS[vertex_id];

    VSOutput output;
    output.position = float4(corner * push.scale, 0.0, 1.0);
    output.uv = corner * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input): SV_Target {
    return textures[push.texture_index].Sample(input.uv);
}
