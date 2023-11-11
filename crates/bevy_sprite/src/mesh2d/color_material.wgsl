#import bevy_sprite::{
    mesh2d_bindings::mesh,
    mesh2d_vertex_output::VertexOutput,
    mesh2d_view_bindings::view,
}

#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

struct ColorMaterial {
    color: vec4<f32>,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32,
};
const COLOR_MATERIAL_FLAGS_TEXTURE_BIT: u32 = 1u;

#ifdef MATERIAL_BUFFER_BATCH_SIZE
@group(1) @binding(0) var<uniform> materials: array<ColorMaterial, #{MATERIAL_BUFFER_BATCH_SIZE}u>;
#else // MATERIAL_BUFFER_BATCH_SIZE
@group(1) @binding(0) var<storage> materials: array<ColorMaterial>;
#endif // MATERIAL_BUFFER_BATCH_SIZE

@group(1) @binding(1) var texture: texture_2d<f32>;
@group(1) @binding(2) var texture_sampler: sampler;

@fragment
fn fragment(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let material_index = mesh[in.instance_index].material_index;
    let material = materials[material_index];

    // Calculate gradients here outside non-uniform control flow
    let duvdx = dpdx(in.uv) * view.pow_2_mip_bias;
    let duvdy = dpdy(in.uv) * view.pow_2_mip_bias;

    var output_color: vec4<f32> = material.color;
#ifdef VERTEX_COLORS
    output_color = output_color * in.color;
#endif
    if ((material.flags & COLOR_MATERIAL_FLAGS_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSampleGrad(texture, texture_sampler, in.uv, duvdx, duvdy);
    }
#ifdef TONEMAP_IN_SHADER
    output_color = tonemapping::tone_mapping(output_color, view.color_grading);
#endif
    return output_color;
}
