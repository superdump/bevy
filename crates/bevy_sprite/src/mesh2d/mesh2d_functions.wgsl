#define_import_path bevy_sprite::mesh2d_functions

#import bevy_sprite::{
    mesh2d_view_bindings::view,
    mesh2d_bindings::mesh,
}
#import bevy_render::{
    instance_index::get_instance_index,
    maths::{affine_to_square, mat2x4_f32_to_mat3x3_unpack},
}

fn get_local_to_world_matrix(instance_index: u32) -> mat4x4<f32> {
    return affine_to_square(mesh[get_instance_index(instance_index)].local_to_world);
}

fn mesh2d_position_local_to_world(local_to_world: mat4x4<f32>, vertex_position: vec4<f32>) -> vec4<f32> {
    return local_to_world * vertex_position;
}

fn mesh2d_position_world_to_clip(world_position: vec4<f32>) -> vec4<f32> {
    return view.world_to_ndc * world_position;
}

// NOTE: The intermediate world_position assignment is important
// for precision purposes when using the 'equals' depth comparison
// function.
fn mesh2d_position_local_to_clip(local_to_world: mat4x4<f32>, vertex_position: vec4<f32>) -> vec4<f32> {
    let world_position = mesh2d_position_local_to_world(local_to_world, vertex_position);
    return mesh2d_position_world_to_clip(world_position);
}

fn mesh2d_normal_local_to_world(vertex_normal: vec3<f32>, instance_index: u32) -> vec3<f32> {
    return mat2x4_f32_to_mat3x3_unpack(
        mesh[instance_index].inverse_transpose_local_to_world_a,
        mesh[instance_index].inverse_transpose_local_to_world_b,
    ) * vertex_normal;
}

fn mesh2d_tangent_local_to_world(local_to_world: mat4x4<f32>, vertex_tangent: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        mat3x3<f32>(
            local_to_world[0].xyz,
            local_to_world[1].xyz,
            local_to_world[2].xyz
        ) * vertex_tangent.xyz,
        vertex_tangent.w
    );
}
