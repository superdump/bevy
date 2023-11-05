#import bevy_sprite::{
    mesh2d_functions as mesh_functions,
    mesh2d_vertex_output::VertexOutput,
    mesh2d_view_bindings::view,
}
#import bevy_render::{
    maths::{affine_to_square, mat2x4_f32_to_mat3x3_unpack},
}

#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

struct Vertex {
    // @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(3) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(4) color: vec4<f32>,
#endif

    // Affine 4x3 matrix transposed to 3x4
    // Use bevy_render::maths::affine_to_square to unpack
    @location(6) i_model_a: vec4<f32>,
    @location(7) i_model_b: vec4<f32>,
    @location(8) i_model_c: vec4<f32>,
    // 3x3 matrix packed in mat2x4 and f32 as:
    // [0].xyz, [1].x,
    // [1].yz, [2].xy
    // [2].z
    // Use bevy_render::maths::mat2x4_f32_to_mat3x3_unpack to unpack
    @location(9) i_inverse_transpose_model_a: vec4<f32>,
    @location(10) i_inverse_transpose_model_b: vec4<f32>,
    @location(11) i_inverse_transpose_model_c: f32,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    @location(12) i_flags: u32,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_POSITIONS
    // var model = mesh_functions::get_model_matrix(vertex.instance_index);
    var model = affine_to_square(mat3x4(
        vertex.i_model_a,
        vertex.i_model_b,
        vertex.i_model_c,
    ));
    out.world_position = mesh_functions::mesh2d_position_local_to_world(
        model,
        vec4<f32>(vertex.position, 1.0)
    );
    out.position = mesh_functions::mesh2d_position_world_to_clip(out.world_position);
#endif

#ifdef VERTEX_NORMALS
    // out.world_normal = mesh_functions::mesh2d_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.world_normal = mat2x4_f32_to_mat3x3_unpack(
        mat2x4(
            vertex.i_inverse_transpose_model_a,
            vertex.i_inverse_transpose_model_b,
        ),
        vertex.i_inverse_transpose_model_c,
    ) * vertex.normal;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh2d_tangent_local_to_world(
        model,
        vertex.tangent
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
    return out;
}

@fragment
fn fragment(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
#ifdef VERTEX_COLORS
    var color = in.color;
#ifdef TONEMAP_IN_SHADER
    color = tonemapping::tone_mapping(color, view.color_grading);
#endif
    return color;
#else
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
#endif
}
