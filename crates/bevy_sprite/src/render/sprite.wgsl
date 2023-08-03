#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

#import bevy_render::view  View

@group(0) @binding(0)
var<uniform> view: View;

struct VertexInput {
    @builtin(vertex_index) index: u32,
    // Instance-rate vertex buffer members prefixed with i_
    @location(0) i_col0_tx: vec4<f32>,
    @location(1) i_col1_ty: vec4<f32>,
    @location(2) i_col2_tz: vec4<f32>,
    @location(3) i_color: vec4<f32>,
    @location(4) i_uv_offset_scale: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
};

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let vertex_position = vec3<f32>(
        f32(in.index & 0x1u),
        f32((in.index & 0x2u) >> 1u),
        0.0
    );

    out.clip_position = view.view_proj * mat4x4<f32>(
        vec4<f32>(in.i_col0_tx.xyz, 0.0),
        vec4<f32>(in.i_col1_ty.xyz, 0.0),
        vec4<f32>(in.i_col2_tz.xyz, 0.0),
        vec4<f32>(in.i_col0_tx.w, in.i_col1_ty.w, in.i_col2_tz.w, 1.0),
    ) * vec4<f32>(vertex_position, 1.0);
    out.uv = vec2<f32>(vertex_position.xy) * in.i_uv_offset_scale.zw + in.i_uv_offset_scale.xy;
    out.color = in.i_color;

    return out;
}

@group(1) @binding(0)
var sprite_texture: texture_2d<f32>;
@group(1) @binding(1)
var sprite_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = in.color * textureSample(sprite_texture, sprite_sampler, in.uv);

#ifdef TONEMAP_IN_SHADER
    color = bevy_core_pipeline::tonemapping::tone_mapping(color, view.color_grading);
#endif

    return color;
}
