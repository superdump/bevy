use bevy_asset::{Assets, Handle};
use bevy_ecs::prelude::*;
use bevy_math::{const_vec3, Vec3};
use bevy_render2::{
    mesh::{Mesh, VertexAttributeValues},
    primitives::Aabb,
};
use bevy_transform::components::{GlobalTransform, Transform};

pub fn calculate_aabbs(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    new_meshes: Query<(Entity, &GlobalTransform, &Handle<Mesh>), Without<Aabb>>,
    moved_meshes: Query<
        (Entity, &GlobalTransform, &Handle<Mesh>),
        (Changed<GlobalTransform>, With<Aabb>),
    >,
) {
    for (entity, transform, mesh_handle) in new_meshes.iter().chain(moved_meshes.iter()) {
        if let Some(mesh) = meshes.get(mesh_handle) {
            if let Some(aabb) = mesh_to_aabb(mesh, transform) {
                commands.entity(entity).insert(aabb);
            }
        }
    }
}

const VEC3_MIN: Vec3 = const_vec3!([std::f32::MIN, std::f32::MIN, std::f32::MIN]);
const VEC3_MAX: Vec3 = const_vec3!([std::f32::MAX, std::f32::MAX, std::f32::MAX]);

pub fn mesh_to_aabb(mesh: &Mesh, transform: &GlobalTransform) -> Option<Aabb> {
    if let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        let scale_rotation = Transform {
            translation: Vec3::ZERO,
            scale: transform.scale,
            rotation: transform.rotation,
        };
        if let Some(positions) = match positions {
            VertexAttributeValues::Float32x3(values) => Some(
                values
                    .iter()
                    .map(|coordinates| scale_rotation.mul_vec3(Vec3::from(*coordinates)))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        } {
            let mut minimum = VEC3_MAX;
            let mut maximum = VEC3_MIN;
            for p in positions {
                minimum = minimum.min(p);
                maximum = maximum.max(p);
            }
            if minimum != VEC3_MAX && maximum != VEC3_MIN {
                return Some(Aabb { minimum, maximum });
            }
        }
    }

    None
}
