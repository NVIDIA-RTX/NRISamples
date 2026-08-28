// © 2026 NVIDIA Corporation

struct Vertex {
    float4 position : SV_Position;
};

[maxvertexcount(3)]
void main(triangle Vertex input[3], inout TriangleStream<Vertex> output) {
    output.Append(input[0]);
    output.Append(input[1]);
    output.Append(input[2]);
}

