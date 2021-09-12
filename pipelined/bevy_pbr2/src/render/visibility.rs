use bevy_asset::{Assets, Handle};
use bevy_ecs::prelude::*;
use bevy_render2::{mesh::Mesh, primitives::Aabb};

pub fn calculate_aabbs(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    without_aabb: Query<(Entity, &Handle<Mesh>), Without<Aabb>>,
) {
    for (entity, mesh_handle) in without_aabb.iter() {
        if let Some(mesh) = meshes.get(mesh_handle) {
            if let Some(aabb) = mesh.compute_aabb() {
                commands.entity(entity).insert(aabb);
            }
        }
    }
}
