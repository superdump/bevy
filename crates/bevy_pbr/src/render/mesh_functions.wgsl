#define_import_path bevy_pbr::mesh_functions

#import bevy_pbr::mesh_view_bindings  view
#import bevy_pbr::mesh_bindings       mesh
#import bevy_pbr::mesh_types          MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT
#ifdef DATA_TEXTURE_MESH_UNIFORM
#import bevy_pbr::mesh_bindings       mesh_data_texture
#import bevy_pbr::mesh_types          Mesh
#endif // DATA_TEXTURE_MESH_UNIFORM
#import bevy_render::instance_index   get_instance_index

#ifdef DATA_TEXTURE_MESH_UNIFORM
var<private> mesh_loaded: u32 = 0;

fn fetch_texel(
    texels_in_layer: u32,
    dimensions: vec2<u32>,
    index: u32
) -> vec4<f32> {
    let array_index = index / texels_in_layer;
    let texel_in_layer = index % texels_in_layer;
    let coords = vec2(
        texel_in_layer % dimensions.x,
        texel_in_layer / dimensions.x,
    );
    return textureLoad(
        mesh_data_texture,
        coords,
        array_index,
        0
    );
}

fn load_mesh(instance_index: u32) {
    if (mesh_loaded != 0u) {
        return;
    }
    let dimensions = textureDimensions(mesh_data_texture);
    let texels_in_layer = dimensions.x * dimensions.y;
    let texel_index = instance_index * 9u;
    let temp = fetch_texel(texels_in_layer, dimensions, texel_index + 8u).rg;
    mesh = Mesh(
        mat3x4(
            fetch_texel(texels_in_layer, dimensions, texel_index),
            fetch_texel(texels_in_layer, dimensions, texel_index + 1u),
            fetch_texel(texels_in_layer, dimensions, texel_index + 2u),
        ),
        mat3x4(
            fetch_texel(texels_in_layer, dimensions, texel_index + 3u),
            fetch_texel(texels_in_layer, dimensions, texel_index + 4u),
            fetch_texel(texels_in_layer, dimensions, texel_index + 5u),
        ),
        mat2x4(
            fetch_texel(texels_in_layer, dimensions, texel_index + 6u),
            fetch_texel(texels_in_layer, dimensions, texel_index + 7u),
        ),
        temp.x,
        bitcast<u32>(temp.y),
    );
    mesh_loaded = 1u;
}
#endif // DATA_TEXTURE_MESH_UNIFORM

fn affine_to_square(affine: mat3x4<f32>) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(
        affine[0],
        affine[1],
        affine[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    ));
}

fn mat2x4_f32_to_mat3x3_unpack(
    a: mat2x4<f32>,
    b: f32,
) -> mat3x3<f32> {
    return mat3x3<f32>(
        a[0].xyz,
        vec3<f32>(a[0].w, a[1].xy),
        vec3<f32>(a[1].zw, b),
    );
}

fn get_model_matrix(instance_index: u32) -> mat4x4<f32> {
#ifdef DATA_TEXTURE_MESH_UNIFORM
    load_mesh(instance_index);
    return affine_to_square(mesh.model);
#else // DATA_TEXTURE_MESH_UNIFORM
    return affine_to_square(mesh[get_instance_index(instance_index)].model);
#endif // DATA_TEXTURE_MESH_UNIFORM
}

fn get_previous_model_matrix(instance_index: u32) -> mat4x4<f32> {
#ifdef DATA_TEXTURE_MESH_UNIFORM
    load_mesh(instance_index);
    return affine_to_square(mesh.previous_model);
#else // DATA_TEXTURE_MESH_UNIFORM
    return affine_to_square(mesh[get_instance_index(instance_index)].previous_model);
#endif // DATA_TEXTURE_MESH_UNIFORM
}

fn get_flags(instance_index: u32) -> u32 {
#ifdef DATA_TEXTURE_MESH_UNIFORM
    load_mesh(instance_index);
    return mesh.flags;
#else // DATA_TEXTURE_MESH_UNIFORM
    return mesh[get_instance_index(instance_index)].flags;
#endif // DATA_TEXTURE_MESH_UNIFORM
}

fn mesh_position_local_to_world(model: mat4x4<f32>, vertex_position: vec4<f32>) -> vec4<f32> {
    return model * vertex_position;
}

fn mesh_position_world_to_clip(world_position: vec4<f32>) -> vec4<f32> {
    return view.view_proj * world_position;
}

// NOTE: The intermediate world_position assignment is important
// for precision purposes when using the 'equals' depth comparison
// function.
fn mesh_position_local_to_clip(model: mat4x4<f32>, vertex_position: vec4<f32>) -> vec4<f32> {
    let world_position = mesh_position_local_to_world(model, vertex_position);
    return mesh_position_world_to_clip(world_position);
}

fn mesh_normal_local_to_world(vertex_normal: vec3<f32>, instance_index: u32) -> vec3<f32> {
    // NOTE: The mikktspace method of normal mapping requires that the world normal is
    // re-normalized in the vertex shader to match the way mikktspace bakes vertex tangents
    // and normal maps so that the exact inverse process is applied when shading. Blender, Unity,
    // Unreal Engine, Godot, and more all use the mikktspace method. Do not change this code
    // unless you really know what you are doing.
    // http://www.mikktspace.com/
#ifdef DATA_TEXTURE_MESH_UNIFORM
    load_mesh(instance_index);
    return normalize(
        mat2x4_f32_to_mat3x3_unpack(
            mesh.inverse_transpose_model_a,
            mesh.inverse_transpose_model_b,
        ) * vertex_normal
    );
#else // DATA_TEXTURE_MESH_UNIFORM
    return normalize(
        mat2x4_f32_to_mat3x3_unpack(
            mesh[instance_index].inverse_transpose_model_a,
            mesh[instance_index].inverse_transpose_model_b,
        ) * vertex_normal
    );
#endif // DATA_TEXTURE_MESH_UNIFORM
}

// Calculates the sign of the determinant of the 3x3 model matrix based on a
// mesh flag
fn sign_determinant_model_3x3m(instance_index: u32) -> f32 {
    // bool(u32) is false if 0u else true
    // f32(bool) is 1.0 if true else 0.0
    // * 2.0 - 1.0 remaps 0.0 or 1.0 to -1.0 or 1.0 respectively
#ifdef DATA_TEXTURE_MESH_UNIFORM
    load_mesh(instance_index);
    return f32(bool(mesh.flags & MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT)) * 2.0 - 1.0;
#else // DATA_TEXTURE_MESH_UNIFORM
    return f32(bool(mesh[instance_index].flags & MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT)) * 2.0 - 1.0;
#endif // DATA_TEXTURE_MESH_UNIFORM
}

fn mesh_tangent_local_to_world(model: mat4x4<f32>, vertex_tangent: vec4<f32>, instance_index: u32) -> vec4<f32> {
    // NOTE: The mikktspace method of normal mapping requires that the world tangent is
    // re-normalized in the vertex shader to match the way mikktspace bakes vertex tangents
    // and normal maps so that the exact inverse process is applied when shading. Blender, Unity,
    // Unreal Engine, Godot, and more all use the mikktspace method. Do not change this code
    // unless you really know what you are doing.
    // http://www.mikktspace.com/
    return vec4<f32>(
        normalize(
            mat3x3<f32>(
                model[0].xyz,
                model[1].xyz,
                model[2].xyz
            ) * vertex_tangent.xyz
        ),
        // NOTE: Multiplying by the sign of the determinant of the 3x3 model matrix accounts for
        // situations such as negative scaling.
        vertex_tangent.w * sign_determinant_model_3x3m(instance_index)
    );
}
