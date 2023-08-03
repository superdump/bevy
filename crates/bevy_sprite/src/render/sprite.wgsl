#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

#import bevy_render::view  View

@group(0) @binding(0)
var<uniform> view: View;

struct Quad {
    col0_tx: vec4<f32>,
    col1_ty: vec4<f32>,
    col2_tz: vec4<f32>,
    color: vec4<f32>,
    uv_offset_scale: vec4<f32>,
}

#ifdef PER_OBJECT_BUFFER_BATCH_SIZE
@group(2) @binding(0)
var<uniform> quads: array<Quad, #{PER_OBJECT_BUFFER_BATCH_SIZE}u>;
#else // PER_OBJECT_BUFFER_BATCH_SIZE
@group(2) @binding(0)
var<storage> quads: array<Quad>;
#endif // PER_OBJECT_BUFFER_BATCH_SIZE

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let vertex_position = vec3<f32>(
        f32(vertex_index & 0x1u),
        f32((vertex_index & 0x2u) >> 1u),
        0.0
    );

#ifdef PER_OBJECT_BUFFER_BATCH_SIZE
    let instance_index = (vertex_index >> 2u) % #{PER_OBJECT_BUFFER_BATCH_SIZE}u;
#else // PER_OBJECT_BUFFER_BATCH_SIZE
    let instance_index = vertex_index >> 2u;
#endif // PER_OBJECT_BUFFER_BATCH_SIZE

    let quad = quads[instance_index];

    out.clip_position = view.view_proj * mat4x4<f32>(
        vec4<f32>(quad.col0_tx.xyz, 0.0),
        vec4<f32>(quad.col1_ty.xyz, 0.0),
        vec4<f32>(quad.col2_tz.xyz, 0.0),
        vec4<f32>(quad.col0_tx.w, quad.col1_ty.w, quad.col2_tz.w, 1.0),
    ) * vec4<f32>(vertex_position, 1.0);
    out.uv = vec2<f32>(vertex_position.xy) * quad.uv_offset_scale.zw + quad.uv_offset_scale.xy;
    out.instance_index = instance_index;

    return out;
}

@group(1) @binding(0)
var sprite_texture: texture_2d<f32>;
@group(1) @binding(1)
var sprite_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(sprite_texture, sprite_sampler, in.uv);
    color = quads[in.instance_index].color * color;

#ifdef TONEMAP_IN_SHADER
    color = bevy_core_pipeline::tonemapping::tone_mapping(color, view.color_grading);
#endif

    return color;
}
