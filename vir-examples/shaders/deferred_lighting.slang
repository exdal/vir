/// The lighting half of the deferred pair. It draws no geometry: one triangle covers the
/// screen and every pixel is shaded out of what the geometry pass left in the G-buffer.

struct PushConstants {
    float3 camera_position;
    float ambient;
    float3 light_direction;
    float light_intensity;
    float3 light_color;
    float specular_strength;
    uint mode;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

[[vk::binding(0, 0)]]
Sampler2D albedo_texture;
[[vk::binding(1, 0)]]
Sampler2D normal_texture;
[[vk::binding(2, 0)]]
Sampler2D position_texture;

static const uint MODE_SHADED = 0;
static const uint MODE_ALBEDO = 1;
static const uint MODE_NORMAL = 2;
static const uint MODE_POSITION = 3;

static const float3 BACKGROUND = float3(0.02, 0.02, 0.05);
static const float SHININESS = 48.0;

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

/// One oversized triangle rather than a quad, so the draw is three vertices and no buffer.
[shader("vertex")]
VSOutput vs_main(uint vertex_id: SV_VertexID) {
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);

    VSOutput output;
    output.uv = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input): SV_Target {
    float4 position_sample = position_texture.Sample(input.uv);

    // the geometry pass only wrote alpha where it drew, so the rest is background
    if (position_sample.a < 0.5) {
        return float4(BACKGROUND, 1.0);
    }

    float3 albedo = albedo_texture.Sample(input.uv).rgb;
    float3 normal = normalize(normal_texture.Sample(input.uv).xyz * 2.0 - 1.0);
    float3 world = position_sample.xyz;

    switch (push.mode) {
    case MODE_ALBEDO:
        return float4(albedo, 1.0);
    case MODE_NORMAL:
        return float4(normal * 0.5 + 0.5, 1.0);
    case MODE_POSITION:
        return float4(frac(world), 1.0);
    default:
        break;
    }

    // Blinn-Phong against one directional light, which is all the G-buffer has to feed
    float3 to_light = normalize(-push.light_direction);
    float3 to_view = normalize(push.camera_position - world);
    float3 half_vector = normalize(to_light + to_view);

    float diffuse = max(dot(normal, to_light), 0.0);
    float specular = pow(max(dot(normal, half_vector), 0.0), SHININESS) * push.specular_strength;
    // a surface facing away from the light keeps no highlight
    specular *= step(0.0001, diffuse);

    float3 lit = albedo * push.ambient
        + (albedo * diffuse + specular) * push.light_color * push.light_intensity;

    return float4(lit, 1.0);
}
