#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;
layout(location = 2) in vec2 Vertex_Uv;

layout(location = 0) out vec3 v_ViewNormal;
layout(location = 1) out vec2 v_Uv;

layout(set = 0, binding = 0) uniform ViewTransform {
    mat4 View;
    mat4 ViewInv;
    mat4 ViewProj;
    vec3 ViewWorldPosition;
};

layout(set = 1, binding = 0) uniform MeshTransform {
    mat4 Model;
    mat4 ModelInvTrans;
};

void main() {
    // For non-uniform scaling, the Model matrix must be inverse-transposed which has the effect of
    // applying inverse scaling while retaining the correct rotation
    // The normals must be re-normalised after applying the inverse-transpose because this can affect
    // the length of the normal
    // The normals need to rotate inverse to the view rotation
    // Using mat3 is important else the translation in the Model matrix can have other unintended effects
    v_ViewNormal = mat3(ViewInv) * mat3(ModelInvTrans) * Vertex_Normal * 0.5 + 0.5;
    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
}
