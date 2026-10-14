// © 2026 NVIDIA Corporation

struct Output {
    float4 position : SV_Position;
};

Output main(uint vertexId : SV_VertexID) {
    const float2 positions[] = {
        float2(-0.5, 0.5),
        float2(0.5, 0.5),
        float2(0.0, -0.5),
    };

    Output output;
    output.position = float4(positions[vertexId % 3], 0.0, 1.0);

    return output;
}

