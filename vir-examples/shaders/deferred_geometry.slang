/// The geometry half of the deferred pair. It shades nothing: every draw writes what it knows
/// about the surface into the G-buffer and lets `deferred_lighting.slang` do the rest.

/// Vertices arrive in world space, so the only transform a draw needs is the camera's.
///
/// `view_proj` is stored row by row, which is the layout the build compiles Slang with. The
/// example transposes glam's column-major matrix on the way in.
struct PushConstants {
    float4x4 view_proj;
    float3 base_color;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

[[vk::binding(0, 0)]]
Sampler2D base_color_texture;

/// What the example uploads: `float3 position`, `float3 normal`, then `float2 uv`, which is
/// what reflection packs into binding 0.
struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct VSOutput {
    float4 clip : SV_Position;
    float3 world : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

/// The three color targets the region renders into, in the order the pass binds them.
struct GBuffer {
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 position : SV_Target2;
};

[shader("vertex")]
VSOutput vs_main(VSInput input) {
    VSOutput output;
    output.clip = mul(push.view_proj, float4(input.position, 1.0));
    output.world = input.position;
    output.normal = input.normal;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
GBuffer fs_main(VSOutput input) {
    // the texture is an `_SRGB` image, so what comes back is already linear
    float3 albedo = push.base_color * base_color_texture.Sample(input.uv).rgb;

    GBuffer output;
    output.albedo = float4(albedo, 1.0);
    // signed normals do not survive an unsigned target, so they are folded into [0, 1]
    output.normal = float4(normalize(input.normal) * 0.5 + 0.5, 1.0);
    // alpha is the coverage the lighting pass reads the background out of
    output.position = float4(input.world, 1.0);
    return output;
}
