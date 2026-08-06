struct PushConstants {
    float2 screen_size;
    uint texture_index;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

/// The graph's bindless table. A variable-count combined image sampler array is what it
/// recognises as the slot textures are written into, so `texture_index` is all a draw needs
/// to name one.
[[vk::binding(0, 0)]]
Sampler2D textures[];

/// Mirrors `egui::epaint::Vertex`. The color arrives as a packed `Color32` rather than a
/// `float4` because reflection only derives 32-bit attribute formats, so a `uint` is what
/// makes the tightly packed 20-byte vertex come out right.
struct VSInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD;
    uint color : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

float3 linear_from_srgb(float3 srgb) {
    float3 lower = srgb / 12.92;
    float3 higher = pow((srgb + 0.055) / 1.055, 2.4);
    return lerp(lower, higher, step(0.04045, srgb));
}

[shader("vertex")]
VSOutput vs_main(VSInput input) {
    float4 packed = float4(
                        float(input.color & 0xffu),
                        float((input.color >> 8) & 0xffu),
                        float((input.color >> 16) & 0xffu),
                        float((input.color >> 24) & 0xffu))
                    / 255.0;

    // egui hands colors over as sRGB with premultiplied alpha, and the target is an _SRGB
    // format, so the hardware encodes on write and the shader owes it linear values. The
    // transfer function is defined on straight color, so undo the premultiply across it.
    float3 straight = packed.a > 0.0 ? packed.rgb / packed.a : packed.rgb;

    VSOutput output;
    output.position = float4(
        2.0 * input.position.x / push.screen_size.x - 1.0,
        2.0 * input.position.y / push.screen_size.y - 1.0,
        0.0,
        1.0);
    output.color = float4(linear_from_srgb(straight) * packed.a, packed.a);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    // The atlas is an _SRGB image, so the sample comes back linear and premultiplied, which
    // is the space the interpolated vertex color is already in.
    float4 texel = textures[push.texture_index].Sample(input.uv);
    return input.color * texel;
}
