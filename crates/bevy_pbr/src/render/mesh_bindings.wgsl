#define_import_path bevy_pbr::mesh_bindings

#import bevy_pbr::mesh_types

@group(2) @binding(0)
#ifdef MESH_BATCH_SIZE
var<uniform> mesh: array<Mesh, #{MESH_BATCH_SIZE}u>;
#else
var<storage> mesh: array<Mesh>;
#endif

#ifdef SKINNED
@group(2) @binding(1)
var<uniform> joint_matrices: SkinnedMesh;
#import bevy_pbr::skinning
#endif
