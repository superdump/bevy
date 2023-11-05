#define_import_path bevy_pbr::forward_io

struct Vertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
// (Alternate UVs are at location 3, but they're currently unused here.)
#ifdef VERTEX_TANGENTS
    @location(4) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef SKINNED
    @location(6) joint_indices: vec4<u32>,
    @location(7) joint_weights: vec4<f32>,
#endif
#ifdef MORPH_TARGETS
    @builtin(vertex_index) index: u32,
#endif

    // Affine 4x3 matrices transposed to 3x4
    // Use bevy_render::maths::affine_to_square to unpack
    @location(8) i_model_a: vec4<f32>,
    @location(9) i_model_b: vec4<f32>,
    @location(10) i_model_c: vec4<f32>,
    @location(11) i_previous_model_a: vec4<f32>,
    @location(12) i_previous_model_b: vec4<f32>,
    @location(13) i_previous_model_c: vec4<f32>,
    // 3x3 matrix packed in mat2x4 and f32 as:
    // [0].xyz, [1].x,
    // [1].yz, [2].xy
    // [2].z
    // Use bevy_pbr::mesh_functions::mat2x4_f32_to_mat3x3_unpack to unpack
    @location(14) i_inverse_transpose_model_a: vec4<f32>,
    @location(15) i_inverse_transpose_model_b: vec4<f32>,
    @location(16) i_inverse_transpose_model_c: f32,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    @location(17) i_flags: u32,
};

struct VertexOutput {
    // This is `clip position` when the struct is used as a vertex stage output
    // and `frag coord` when used as a fragment stage input
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(3) world_tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(4) color: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(5) @interpolate(flat) flags: u32,
#endif
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}
