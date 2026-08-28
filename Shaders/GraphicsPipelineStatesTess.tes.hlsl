// © 2026 NVIDIA Corporation

struct ControlPoint {
    float4 position : POSITION;
};

struct Constants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

struct Output {
    float4 position : SV_Position;
};

[domain("tri")]
Output main(Constants constants, float3 barycentrics : SV_DomainLocation, const OutputPatch<ControlPoint, 3> patch) {
    Output output;
    output.position = patch[0].position * barycentrics.x + patch[1].position * barycentrics.y + patch[2].position * barycentrics.z;

    return output;
}

