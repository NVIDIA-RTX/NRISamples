// © 2026 NVIDIA Corporation

struct ControlPoint {
    float4 position : POSITION;
};

ControlPoint main(uint vertexId : SV_VertexID) {
    const float2 positions[] = {
        float2(-0.75, 0.75),
        float2(0.75, 0.75),
        float2(0.0, -0.75),
    };

    ControlPoint output;
    output.position = float4(positions[vertexId % 3], 0.5, 1.0);

    return output;
}

