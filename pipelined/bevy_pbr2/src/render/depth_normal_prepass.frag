#version 450

layout(location = 0) in vec3 v_ViewNormal;
layout(location = 1) in vec2 v_Uv;

layout(location = 0) out vec4 o_Target;

void main() {
    vec3 N = normalize(v_ViewNormal);
    o_Target = vec4(N, 1.0);
}
