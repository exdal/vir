struct PushConstants {
    float2 offset;
    float scale;
    float tint;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

struct VSOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

struct VSInput {
    float2 position : POSITION;
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
    output.position = float4(POSITIONS[vertex_id] * push.scale + push.offset, 0.0, 1.0);
    output.color = COLORS[vertex_id];
    return output;
}

[shader("vertex")]
VSOutput vs_buffer(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input): SV_Target {
    return float4(lerp(input.color, float3(1.0, 1.0, 1.0), push.tint), 1.0);
}
