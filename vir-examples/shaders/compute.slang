/// One triangle's placement. Only the two dispatches ever look at this, so its layout is
/// theirs alone to agree on.
struct Instance {
    float2 center;
    float radius;
    float3 color;
};

/// A vertex as `vs_main` reads it: `float2 position` then `float3 color`, which is what
/// reflection packs into binding 0 and what the expand dispatch has to write.
struct Vertex {
    float2 position : POSITION;
    float3 color : COLOR;
};

/// Both buffers arrive as device addresses, so neither dispatch needs a descriptor set: the
/// draw binds the same vertex buffer through the vertex input state instead.
struct PushConstants {
    Instance *instances;
    Vertex *vertices;
    uint count;
    float time;
    float2 scale;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

static const float TAU = 6.28318530718;

float3 palette(float t) {
    return 0.55 + 0.45 * cos(TAU * (t + float3(0.0, 0.33, 0.67)));
}

/// Where every triangle sits this frame, one thread each.
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_place(uint3 thread: SV_DispatchThreadID) {
    uint index = thread.x;
    if (index >= push.count) {
        return;
    }

    float t = float(index) / float(push.count);
    float orbit = TAU * t + push.time * 0.35;
    float wobble = sin(push.time * 1.7 + TAU * t * 3.0);

    Instance instance;
    instance.center = float2(cos(orbit), sin(orbit)) * (0.55 + 0.12 * wobble);
    instance.radius = 0.05 + 0.02 * wobble;
    instance.color = palette(t);
    push.instances[index] = instance;
}

/// The corners of what the first dispatch placed. Reading `instances` here is what the graph
/// has to have made visible, and writing `vertices` is what the draw then waits on.
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_expand(uint3 thread: SV_DispatchThreadID) {
    uint index = thread.x;
    if (index >= push.count) {
        return;
    }

    Instance instance = push.instances[index];
    float spin = push.time * 1.5 + TAU * float(index) / float(push.count);

    for (uint corner = 0; corner < 3; ++corner) {
        float angle = spin + TAU * float(corner) / 3.0;

        Vertex vertex;
        vertex.position = (instance.center + float2(cos(angle), sin(angle)) * instance.radius) * push.scale;
        vertex.color = instance.color;
        push.vertices[index * 3 + corner] = vertex;
    }
}

struct VSOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
VSOutput vs_main(Vertex input) {
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input): SV_Target {
    return float4(input.color, 1.0);
}
