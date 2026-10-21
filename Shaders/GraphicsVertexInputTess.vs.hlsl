// © 2026 NVIDIA Corporation

struct ControlPoint {
    float4 position : POSITION;
};

ControlPoint main(float2 position : POSITION) {
    ControlPoint output;
    output.position = float4(position, 0.5, 1.0);

    return output;
}
