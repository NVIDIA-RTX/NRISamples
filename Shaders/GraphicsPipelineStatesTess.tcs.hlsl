// © 2026 NVIDIA Corporation

struct ControlPoint {
    float4 position : POSITION;
};

struct Constants {
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

Constants GetConstants(InputPatch<ControlPoint, 3> patch) {
    Constants output;
    output.edges[0] = 1.0;
    output.edges[1] = 1.0;
    output.edges[2] = 1.0;
    output.inside = 1.0;

    return output;
}

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("GetConstants")]
ControlPoint main(InputPatch<ControlPoint, 3> patch, uint index : SV_OutputControlPointID) {
    return patch[index];
}

