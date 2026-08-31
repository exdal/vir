// Every scalar Vulkan descriptor shape `vir` accepts. The compute entry point uses the first
// nine; input attachments are fragment-only, so the second entry point owns the last one.

struct UniformData {
    float4 value;
};

[[vk::binding(0, 0)]]
SamplerState sampler_only;

[[vk::binding(1, 0)]]
Texture2D<float4> sampled_image;

[[vk::binding(2, 0)]]
Sampler2D combined_image_sampler;

[[vk::binding(3, 0)]]
RWTexture2D<float4> storage_image;

[[vk::binding(4, 0)]]
Buffer<float4> uniform_texel_buffer;

[[vk::binding(5, 0)]]
RWBuffer<float4> storage_texel_buffer;

[[vk::binding(6, 0)]]
ConstantBuffer<UniformData> uniform_buffer;

[[vk::binding(7, 0)]]
RWStructuredBuffer<float4> storage_buffer;

[[vk::binding(8, 0)]]
RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(9, 0)]]
[[vk::input_attachment_index(0)]]
SubpassInput<float4> input_attachment;

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 dispatch_id: SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    float2 uv = float2(0.5, 0.5);
    float4 value = sampled_image.SampleLevel(sampler_only, uv, 0.0)
        + combined_image_sampler.SampleLevel(uv, 0.0)
        + uniform_texel_buffer[index]
        + uniform_buffer.value;

    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 1.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(acceleration_structure, RAY_FLAG_NONE, 0xff, ray);
    if (query.Proceed()) {
        value += 1.0;
    }

    storage_image[int2(0, 0)] = value;
    storage_texel_buffer[index] = value;
    storage_buffer[index] = value;
}

[shader("fragment")]
float4 fs_input() : SV_Target {
    return input_attachment.SubpassLoad();
}
