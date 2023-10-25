#version 450
layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 o_target;

layout(set = 1, binding = 0) uniform custom_material {
    vec4 color;
};

layout(set = 1, binding = 1) uniform texture2D custom_material_texture;
layout(set = 1, binding = 2) uniform sampler custom_material_sampler;

// wgsl modules can be imported and used in glsl
// FIXME - this doesn't work any more ...
// #import bevy_pbr::pbr_functions as pbr_funcs

void main() {
    o_target = color * texture(sampler2D(custom_material_texture, custom_material_sampler), v_uv);
    // o_target = pbr_funcs::tone_mapping(color * texture(sampler2D(custom_material_texture, custom_material_sampler), v_uv));
}
