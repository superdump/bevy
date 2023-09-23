#[allow(clippy::module_inception)]
mod mesh;
pub mod morph;
/// Generation for some primitive shape meshes.
pub mod shape;

pub use mesh::*;

use crate::{
    prelude::Image,
    render_asset::{prepare_assets, RenderAssetPlugin},
    Render, RenderApp, RenderSet,
};
use bevy_app::{App, Plugin};
use bevy_asset::{AssetApp, Handle};
use bevy_ecs::{entity::Entity, schedule::IntoSystemConfigs};

/// Adds the [`Mesh`] as an asset and makes sure that they are extracted and prepared for the GPU.
pub struct MeshPlugin;

impl Plugin for MeshPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Mesh>()
            .init_asset::<skinning::SkinnedMeshInverseBindposes>()
            .register_asset_reflect::<Mesh>()
            .register_type::<Option<Handle<Image>>>()
            .register_type::<Option<Vec<String>>>()
            .register_type::<Option<Indices>>()
            .register_type::<Indices>()
            .register_type::<skinning::SkinnedMesh>()
            .register_type::<Vec<Entity>>()
            .init_resource::<MegaMesh>()
            // 'Mesh' must be prepared after 'Image' as meshes rely on the morph target image being ready
            .add_plugins(RenderAssetPlugin::<Mesh, Image>::default());
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<MegaMesh>()
            .add_systems(
                Render,
                prepare_mega_mesh
                    .in_set(RenderSet::PrepareAssets)
                    .after(prepare_assets::<Image>)
                    .before(prepare_assets::<Mesh>),
            );
    }
}
