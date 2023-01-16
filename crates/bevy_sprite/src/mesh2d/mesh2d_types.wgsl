#define_import_path bevy_sprite::mesh2d_types

struct Mesh2d {
    /// Affine transform packed as column-major 3x3 in the xyz elements
    /// and translation in the w elements
    local_to_world: array<vec4<f32>, 3>,
    // model: mat4x4<f32>,
    // inverse_transpose_model: mat4x4<f32>,
    // // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    // flags: u32,
};

struct Meshes2d {
    data: array<Mesh2d, #{MESHES_2D_UNIFORM_ARRAY_LEN}u>,
}
