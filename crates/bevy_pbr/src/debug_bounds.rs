use bevy_app::{App, CoreStage, Plugin};
use bevy_asset::{Assets, Handle};
use bevy_ecs::{
    entity::Entity,
    prelude::*,
    system::{Commands, ParamSet, Query, ResMut},
};
use bevy_hierarchy::{BuildChildren, Children};
use bevy_render::{
    mesh::Mesh,
    prelude::{Color, Visibility},
    primitives::BoundingVolume,
    view::VisibilitySystems,
};
use bevy_transform::TransformSystem;

use crate::{NotShadowCaster, PbrBundle, StandardMaterial};

#[derive(Default)]
pub struct DebugBoundsPlugin<T: BoundingVolume> {
    marker: std::marker::PhantomData<T>,
}

/// A plugin that provides functionality for generating and updating bounding volumes for meshes.
impl<T> Plugin for DebugBoundsPlugin<T>
where
    T: 'static + BoundingVolume + Component,
    Mesh: From<&'static T>,
{
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(
            CoreStage::PostUpdate,
            update_debug_meshes::<T>
                .after(VisibilitySystems::CalculateBounds)
                .after(VisibilitySystems::UpdateOrthographicFrusta)
                .after(VisibilitySystems::UpdatePerspectiveFrusta)
                .after(TransformSystem::TransformPropagate),
        )
        .add_system_to_stage(
            CoreStage::PostUpdate,
            update_debug_mesh_visibility
                .after(update_debug_meshes)
                .before(VisibilitySystems::CheckVisibility),
        );
    }
}

/// Marks an entity that should have a mesh added as a child to represent the mesh's bounding volume.
#[derive(Component)]
pub struct DebugBounds;

/// Marks the debug bounding volume mesh, which exists as a child of a [BoundingVolumeDebug] entity
#[derive(Component)]
pub struct DebugBoundsMesh;

/// Updates existing debug meshes, and creates new debug meshes on entities with a bounding volume
/// component marked with [BoundingVolumeDebug] and no existing debug mesh.
#[allow(clippy::type_complexity)]
pub fn update_debug_meshes<T>(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(&T, Entity, Option<&Children>), (Changed<T>, With<DebugBounds>)>,
    debug_mesh_query: Query<Entity, (With<Handle<Mesh>>, With<DebugBoundsMesh>)>,
) where
    T: 'static + BoundingVolume + Component,
    Mesh: From<&'static T>,
{
    for (bound_vol, entity, optional_children) in query.iter() {
        let mut updated_existing_child = false;
        if let Some(children) = optional_children {
            for child in children.iter() {
                if debug_mesh_query.contains(*child) {
                    updated_existing_child = true;
                    break;
                }
            }
        }
        // if the entity had a child, we don't need to create a new one
        if !updated_existing_child {
            let mesh_handle = meshes.add(bound_vol.new_debug_mesh());
            commands.entity(entity).with_children(|parent| {
                parent
                    .spawn_bundle(PbrBundle {
                        mesh: mesh_handle,
                        material: materials.add(StandardMaterial {
                            base_color: Color::rgb(0.0, 1.0, 0.0),
                            unlit: true,
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .insert_bundle((DebugBoundsMesh, NotShadowCaster));
            });
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn update_debug_mesh_visibility(
    mut query: ParamSet<(
        Query<(&Children, &Visibility), (With<DebugBounds>, Changed<Visibility>)>,
        Query<&mut Visibility, With<DebugBoundsMesh>>,
    )>,
) {
    let child_list: Vec<(Box<Children>, bool)> = query
        .p0()
        .iter()
        .map(|(children, visible)| (Box::new((*children).clone()), visible.is_visible))
        .collect();
    for (children, parent_visible) in child_list.iter() {
        for child in children.iter() {
            if let Ok(mut child_visible) = query.p1().get_mut(*child) {
                child_visible.is_visible = *parent_visible;
            }
        }
    }
}
