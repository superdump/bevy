
#version 450

layout(location = 0) in vec3 v_WorldNormal;

layout(location = 0) out vec4 o_Target;

void main() {
    o_Target = vec4(v_WorldNormal, 0.0);
}
