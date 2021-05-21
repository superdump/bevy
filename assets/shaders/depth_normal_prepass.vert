#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;

layout(location = 0) out vec3 v_ViewNormal;

layout(set = 0, binding = 0) uniform CameraViewProj {
    mat4 ViewProj;
};
layout(set = 1, binding = 0) uniform CameraView {
    mat4 View;
};

layout(set = 2, binding = 0) uniform Transform {
    mat4 Model;
};

void main() {
    // For non-uniform scaling, the Model matrix must be inverse-transposed which has the effect of
    // applying inverse scaling while retaining the correct rotation
    // The normals must be re-normalised after applying the inverse-transpose because this can affect
    // the length of the normal
    // The normals need to rotate inverse to the view rotation
    // Using mat3 is important else the translation in the Model matrix can have other unintended effects
    v_ViewNormal = normalize(inverse(mat3(View)) * transpose(inverse(mat3(Model))) * Vertex_Normal) * 0.5 + 0.5;

    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
}
