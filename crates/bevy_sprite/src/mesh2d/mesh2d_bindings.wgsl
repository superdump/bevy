#define_import_path bevy_sprite::mesh2d_bindings

#import bevy_sprite::mesh2d_types

@group(2) @binding(0)
var<uniform> meshes: Meshes2d;

var<private> local_to_world: mat4x4<f32>;
