// © 2026 NVIDIA Corporation

struct Vertex {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(out vertices Vertex vertices[3], out indices uint3 triangles[1]) {
    SetMeshOutputCounts(3, 1);

    vertices[0].position = float4(-0.75, 0.75, 0.0, 1.0);
    vertices[1].position = float4(0.75, 0.75, 0.0, 1.0);
    vertices[2].position = float4(0.0, -0.75, 0.0, 1.0);
    vertices[0].color = float3(1.0, 0.0, 0.0);
    vertices[1].color = float3(0.0, 1.0, 0.0);
    vertices[2].color = float3(0.0, 0.0, 1.0);
    triangles[0] = uint3(0, 1, 2);
}

