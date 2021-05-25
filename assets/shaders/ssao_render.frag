#version 450

layout(location = 0) in vec2 v_Uv;

layout(set = 0, binding = 0) uniform texture2D ssao_texture;
layout(set = 0, binding = 1) uniform sampler ssao_texture_sampler;

layout(location = 0) out vec4 o_Target;

void main() {
    o_Target = vec4(vec3(texture(sampler2D(ssao_texture, ssao_texture_sampler), v_Uv).r), 1.0);
}
