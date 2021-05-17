#version 450

layout(location = 0) in vec2 v_Uv;

layout(set = 0, binding = 0) uniform texture2D normal_texture;
layout(set = 0, binding = 1) uniform sampler normal_texture_sampler;

layout(location = 0) out vec4 o_Target;

void main() {
    o_Target = vec4(texture(sampler2D(normal_texture, normal_texture_sampler), v_Uv).rgb, 1.0);
}
