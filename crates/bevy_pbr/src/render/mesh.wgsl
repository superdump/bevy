#import bevy_pbr::{
    mesh_functions,
    skinning,
    morph::morph,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}
#import bevy_render::{
    instance_index::get_instance_index,
    maths::{affine_to_square, mat2x4_f32_to_mat3x3_unpack},
}

#ifdef MORPH_TARGETS
fn morph_vertex(vertex_in: Vertex) -> Vertex {
    var vertex = vertex_in;
    let weight_count = bevy_pbr::morph::layer_count();
    for (var i: u32 = 0u; i < weight_count; i ++) {
        let weight = bevy_pbr::morph::weight_at(i);
        if weight == 0.0 {
            continue;
        }
        vertex.position += weight * morph(vertex.index, bevy_pbr::morph::position_offset, i);
#ifdef VERTEX_NORMALS
        vertex.normal += weight * morph(vertex.index, bevy_pbr::morph::normal_offset, i);
#endif
#ifdef VERTEX_TANGENTS
        vertex.tangent += vec4(weight * morph(vertex.index, bevy_pbr::morph::tangent_offset, i), 0.0);
#endif
    }
    return vertex;
}
#endif

@vertex
fn vertex(vertex_no_morph: Vertex) -> VertexOutput {
    var out: VertexOutput;

#ifdef MORPH_TARGETS
    var vertex = morph_vertex(vertex_no_morph);
#else
    var vertex = vertex_no_morph;
#endif

#ifdef SKINNED
    var model = skinning::skin_model(vertex.joint_indices, vertex.joint_weights);
#else
    // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    // See https://github.com/gfx-rs/naga/issues/2416 .
    // var model = mesh_functions::get_model_matrix(vertex_no_morph.instance_index);
    var model = affine_to_square(mat3x4(
        vertex.i_model_a,
        vertex.i_model_b,
        vertex.i_model_c,
    ));
#endif

#ifdef VERTEX_NORMALS
#ifdef SKINNED
    out.world_normal = skinning::skin_normals(model, vertex.normal);
#else
    // out.world_normal = mesh_functions::mesh_normal_local_to_world(
    //     vertex.normal,
    //     // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    //     // See https://github.com/gfx-rs/naga/issues/2416
    //     get_instance_index(vertex_no_morph.instance_index)
    // );
    out.world_normal = mat2x4_f32_to_mat3x3_unpack(
        mat2x4(
            vertex.i_inverse_transpose_model_a,
            vertex.i_inverse_transpose_model_b,
        ),
        vertex.i_inverse_transpose_model_c,
    ) * vertex.normal;
#endif
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_TANGENTS
    // out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
    //     model,
    //     vertex.tangent,
    //     // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    //     // See https://github.com/gfx-rs/naga/issues/2416
    //     get_instance_index(vertex_no_morph.instance_index)
    // );
    out.world_tangent = vec4<f32>(
        normalize(
            mat3x3<f32>(
                model[0].xyz,
                model[1].xyz,
                model[2].xyz
            ) * vertex.tangent.xyz
        ),
        // NOTE: Multiplying by the sign of the determinant of the 3x3 model matrix accounts for
        // situations such as negative scaling.
        vertex.tangent.w * f32(bool(vertex.i_flags & MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT)) * 2.0 - 1.0,
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    // See https://github.com/gfx-rs/naga/issues/2416
    // out.instance_index = get_instance_index(vertex_no_morph.instance_index);
    out.flags = vertex.i_flags;
#endif

    return out;
}

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
#ifdef VERTEX_COLORS
    return mesh.color;
#else
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
#endif
}
