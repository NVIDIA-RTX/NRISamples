// © 2026 NVIDIA Corporation

struct Output {
    float4 position : SV_Position;
};

Output main(float2 position : POSITION) {
    Output output;
    output.position = float4(position, 0.5, 1.0);

    return output;
}
