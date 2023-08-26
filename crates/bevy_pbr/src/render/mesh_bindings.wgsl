#define_import_path bevy_pbr::mesh_bindings

#import bevy_pbr::mesh_types Mesh

#ifdef MESH_BINDGROUP_1

#ifdef DATA_TEXTURE_MESH_UNIFORM
@group(1) @binding(0)
var mesh_data_texture: texture_2d_array<f32>;
var<private> mesh: Mesh;
#else
@group(1) @binding(0)
var<storage> mesh: array<Mesh>;
#endif // DATA_TEXTURE_MESH_UNIFORM

#else // MESH_BINDGROUP_1

#ifdef DATA_TEXTURE_MESH_UNIFORM
@group(2) @binding(0)
var mesh_data_texture: texture_2d_array<f32>;
var<private> mesh: Mesh;
#else
@group(2) @binding(0)
var<storage> mesh: array<Mesh>;
#endif // DATA_TEXTURE_MESH_UNIFORM

#endif // MESH_BINDGROUP_1
