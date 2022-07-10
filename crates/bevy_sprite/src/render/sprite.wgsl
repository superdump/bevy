struct View {
    view_proj: mat4x4<f32>;
    world_position: vec3<f32>;
};
[[group(0), binding(0)]]
var<uniform> view: View;

let SPRITE_QUAD_FLAGS_FLIP_X_BIT: u32 = 1u;
let SPRITE_QUAD_FLAGS_FLIP_Y_BIT: u32 = 2u;

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
#ifdef COLORED
    [[location(3)]] color: vec4<f32>;
#endif
};

[[stage(vertex)]]
fn vertex(
    [[builtin(vertex_index)]] vertex_index: u32,
    [[location(0)]] i_center: vec2<f32>,
    [[location(1)]] i_half_extents: vec2<f32>,
    [[location(2)]] i_uv_offset: vec2<f32>,
    [[location(3)]] i_uv_size: vec2<f32>,
    [[location(4)]] i_flags: u32,
#ifdef COLORED
    [[location(5)]] i_color: vec4<f32>,
#endif
) -> VertexOutput {
    var out: VertexOutput;

    let xy = vec2<i32>(i32(vertex_index & 0x1u), i32((vertex_index & 0x2u) >> 1u));
    out.uv = vec2<f32>(xy.xy);
    let relative_pos_unit = out.uv * 2.0 - 1.0;
    let relative_pos = vec2<f32>(relative_pos_unit * i_half_extents);

    if ((i_flags & SPRITE_QUAD_FLAGS_FLIP_X_BIT) != 0u) {
        out.uv.x = 1.0 - out.uv.x;
    }
    if ((i_flags & SPRITE_QUAD_FLAGS_FLIP_Y_BIT) == 0u) {
        out.uv.y = 1.0 - out.uv.y;
    }
    out.uv = i_uv_offset + out.uv * i_uv_size;

    out.world_position = vec4<f32>(i_center.xy + relative_pos, 0.0, 1.0);
    out.world_normal = vec3<f32>(0.0, 0.0, 1.0);

    out.clip_position = view.view_proj * out.world_position;
#ifdef COLORED
    out.color = i_color;
#endif
    return out;
}

[[group(1), binding(0)]]
var sprite_texture: texture_2d<f32>;
[[group(1), binding(1)]]
var sprite_sampler: sampler;

[[stage(fragment)]]
fn fragment(
    [[builtin(position)]] clip_position: vec4<f32>,
    [[location(0)]] world_position: vec4<f32>,
    [[location(1)]] world_normal: vec3<f32>,
    [[location(2)]] uv: vec2<f32>,
#ifdef COLORED
    [[location(3)]] color: vec4<f32>,
#endif
) -> [[location(0)]] vec4<f32> {
    var output_color = textureSample(sprite_texture, sprite_sampler, uv);
#ifdef COLORED
    output_color = color * output_color;
#endif
    return output_color;
}
