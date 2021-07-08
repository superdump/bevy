#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec2 Vertex_Uv;

layout(location = 0) out vec2 v_Uv;

layout(set = 0, binding = 0) uniform View {
    mat4 ViewProj;
    vec4 ViewWorldPosition;
    // vec3 ViewWorldPosition;
};

void vertex() {
    v_Uv = Vertex_Uv;
    gl_Position = ViewProj * vec4(Vertex_Position, 1.0);
}
