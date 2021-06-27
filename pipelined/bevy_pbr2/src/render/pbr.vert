#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;
layout(location = 2) in vec2 Vertex_Uv;

layout(location = 0) out vec4 v_WorldPosition;
layout(location = 1) out vec3 v_WorldNormal;
layout(location = 2) out vec2 v_Uv;

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
    v_Uv = Vertex_Uv;
    vec4 world_position = Model * vec4(Vertex_Position, 1.0);
    v_WorldPosition = world_position;
    // For non-uniform scaling, the Model matrix must be inverse-transposed which has the effect of
    // applying inverse scaling while retaining the correct rotation
    // The normals must be re-normalised after applying the inverse-transpose because this can affect
    // the length of the normal
    // The normals need to rotate inverse to the view rotation
    // Using mat3 is important else the translation in the Model matrix can have other unintended effects
    v_WorldNormal = mat3(ModelInvTrans) * Vertex_Normal;
    gl_Position = ViewProj * world_position;
}
