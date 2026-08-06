/// Only the fragment stage reads this block: the geometry arrives through the vertex buffer,
/// so the vertex stage has nothing to take from it and reflection leaves it out of the range.
struct PushConstants {
    float3 highlight;
    float tint;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

/// What the example uploads: `float2 position` then `float3 color`, which is exactly what
/// reflection packs into binding 0.
struct VSInput {
    float2 position : POSITION;
    float3 color : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
VSOutput vs_main(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input): SV_Target {
    return float4(lerp(input.color, push.highlight, push.tint), 1.0);
}
