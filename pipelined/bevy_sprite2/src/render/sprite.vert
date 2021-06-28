#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec2 Vertex_Uv;

layout(location = 0) out vec2 v_Uv;

// View bindings - set 0
layout(set = 0, binding = 0) uniform ViewTransform {
    mat4 View;
    mat4 ViewInv;
    mat4 Proj;
    mat4 ProjInv;
    mat4 ViewProj;
    vec3 ViewWorldPosition;
};

void main() {
    v_Uv = Vertex_Uv;
    gl_Position = ViewProj * vec4(Vertex_Position, 1.0);
}
