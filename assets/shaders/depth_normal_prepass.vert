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
    v_ViewNormal = (inverse(View) * Model * vec4(Vertex_Normal, 0.0)).xyz * 0.5 + 0.5;

    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
}
