use crate::{Extract, ExtractSchedule, Render, RenderApp, RenderSet};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{Asset, AssetEvent, AssetId, Assets};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    prelude::*,
    schedule::SystemConfigs,
    system::{StaticSystemParam, SystemParam, SystemParamItem},
};
use bevy_utils::{
    slotmap::{new_key_type, SlotMap},
    HashMap, HashSet,
};
use crossbeam_channel::{Receiver, Sender};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

pub enum PrepareAssetError<E: Send + Sync + 'static> {
    RetryNextUpdate(E),
}

/// Describes how an asset gets extracted and prepared for rendering.
///
/// In the [`ExtractSchedule`](crate::ExtractSchedule) step the asset is transferred
/// from the "main world" into the "render world".
/// Therefore it is converted into a [`RenderAsset::ExtractedAsset`], which may be the same type
/// as the render asset itself.
///
/// After that in the [`RenderSet::PrepareAssets`](crate::RenderSet::PrepareAssets) step the extracted asset
/// is transformed into its GPU-representation of type [`RenderAsset::PreparedAsset`].
pub trait RenderAsset: Asset {
    /// The representation of the asset in the "render world".
    type ExtractedAsset: Send + Sync + 'static;
    /// The GPU-representation of the asset.
    type PreparedAsset: Send + Sync + 'static;
    /// Specifies all ECS data required by [`RenderAsset::prepare_asset`].
    /// For convenience use the [`lifetimeless`](bevy_ecs::system::lifetimeless) [`SystemParam`].
    type Param: SystemParam;
    /// Converts the asset into a [`RenderAsset::ExtractedAsset`].
    fn extract_asset(&self) -> Self::ExtractedAsset;
    /// Prepares the `extracted asset` for the GPU by transforming it into
    /// a [`RenderAsset::PreparedAsset`]. Therefore ECS data may be accessed via the `param`.
    fn prepare_asset(
        extracted_asset: Self::ExtractedAsset,
        param: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>>;
}

/// This plugin extracts the changed assets from the "app world" into the "render world"
/// and prepares them for the GPU. They can then be accessed from the [`RenderAssets`] resource.
///
/// Therefore it sets up the [`ExtractSchedule`](crate::ExtractSchedule) and
/// [`RenderSet::PrepareAssets`](crate::RenderSet::PrepareAssets) steps for the specified [`RenderAsset`].
///
/// The `AFTER` generic parameter can be used to specify that `A::prepare_asset` should not be run until
/// `prepare_assets::<AFTER>` has completed. This allows the `prepare_asset` function to depend on another
/// prepared [`RenderAsset`], for example `Mesh::prepare_asset` relies on `RenderAssets::<Image>` for morph
/// targets, so the plugin is created as `RenderAssetPlugin::<Mesh, Image>::default()`.
pub struct RenderAssetPlugin<A: RenderAsset, AFTER: RenderAssetDependency + 'static = ()> {
    phantom: PhantomData<fn() -> (A, AFTER)>,
}

impl<A: RenderAsset, AFTER: RenderAssetDependency + 'static> Default
    for RenderAssetPlugin<A, AFTER>
{
    fn default() -> Self {
        Self {
            phantom: Default::default(),
        }
    }
}

impl<A: RenderAsset, AFTER: RenderAssetDependency + 'static> Plugin
    for RenderAssetPlugin<A, AFTER>
{
    fn build(&self, app: &mut App) {
        let (render_asset_key_sender, render_asset_key_receiver) =
            create_render_asset_key_channel::<A>();
        app.insert_resource(render_asset_key_receiver)
            .add_systems(PostUpdate, update_main_world_render_asset_keys::<A>);
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedAssets<A>>()
                .init_resource::<RenderAssets<A>>()
                .init_resource::<RenderAssetKeyUpdates<A>>()
                .insert_resource(render_asset_key_sender)
                .init_resource::<PrepareNextFrameAssets<A>>()
                .add_systems(ExtractSchedule, extract_render_asset::<A>);
            AFTER::register_system(
                render_app,
                prepare_assets::<A>.in_set(RenderSet::PrepareAssets),
            );
        }
    }
}

// helper to allow specifying dependencies between render assets
pub trait RenderAssetDependency {
    fn register_system(render_app: &mut App, system: SystemConfigs);
}

impl RenderAssetDependency for () {
    fn register_system(render_app: &mut App, system: SystemConfigs) {
        render_app.add_systems(Render, system);
    }
}

impl<A: RenderAsset> RenderAssetDependency for A {
    fn register_system(render_app: &mut App, system: SystemConfigs) {
        render_app.add_systems(Render, system.after(prepare_assets::<A>));
    }
}

/// Temporarily stores the extracted and removed assets of the current frame.
#[derive(Resource)]
pub struct ExtractedAssets<A: RenderAsset> {
    extracted: Vec<(AssetId<A>, A::ExtractedAsset)>,
    removed: Vec<AssetId<A>>,
}

impl<A: RenderAsset> Default for ExtractedAssets<A> {
    fn default() -> Self {
        Self {
            extracted: Default::default(),
            removed: Default::default(),
        }
    }
}

new_key_type! { pub struct InnerRenderAssetKey; }

#[derive(Component, PartialOrd, Ord, Hash)]
pub struct RenderAssetKey<A: RenderAsset> {
    pub inner: InnerRenderAssetKey,
    marker: PhantomData<A>,
}

impl<A: RenderAsset> Clone for RenderAssetKey<A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: self.marker.clone(),
        }
    }
}

impl<A: RenderAsset> Copy for RenderAssetKey<A> {}

impl<A: RenderAsset> PartialEq for RenderAssetKey<A> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<A: RenderAsset> Eq for RenderAssetKey<A> {}

impl<A: RenderAsset> RenderAssetKey<A> {
    pub fn new(inner: InnerRenderAssetKey) -> Self {
        Self {
            inner,
            marker: Default::default(),
        }
    }
}

#[derive(Resource, Clone)]
pub struct RenderAssetKeyReceiver<A: RenderAsset> {
    pub inner: Receiver<Vec<(Vec<Entity>, RenderAssetKey<A>)>>,
    marker: PhantomData<A>,
}

impl<A: RenderAsset> RenderAssetKeyReceiver<A> {
    pub fn new(receiver: Receiver<Vec<(Vec<Entity>, RenderAssetKey<A>)>>) -> Self {
        Self {
            inner: receiver,
            marker: PhantomData,
        }
    }
}

impl<A: RenderAsset> Deref for RenderAssetKeyReceiver<A> {
    type Target = Receiver<Vec<(Vec<Entity>, RenderAssetKey<A>)>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<A: RenderAsset> DerefMut for RenderAssetKeyReceiver<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Resource)]
pub struct RenderAssetKeySender<A: RenderAsset> {
    pub inner: Sender<Vec<(Vec<Entity>, RenderAssetKey<A>)>>,
    marker: PhantomData<A>,
}

impl<A: RenderAsset> RenderAssetKeySender<A> {
    pub fn new(receiver: Sender<Vec<(Vec<Entity>, RenderAssetKey<A>)>>) -> Self {
        Self {
            inner: receiver,
            marker: PhantomData,
        }
    }
}

impl<A: RenderAsset> Deref for RenderAssetKeySender<A> {
    type Target = Sender<Vec<(Vec<Entity>, RenderAssetKey<A>)>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<A: RenderAsset> DerefMut for RenderAssetKeySender<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub fn create_render_asset_key_channel<A: RenderAsset>(
) -> (RenderAssetKeySender<A>, RenderAssetKeyReceiver<A>) {
    let (s, r) = crossbeam_channel::unbounded::<Vec<(Vec<Entity>, RenderAssetKey<A>)>>();
    (
        RenderAssetKeySender::<A>::new(s),
        RenderAssetKeyReceiver::<A>::new(r),
    )
}
pub fn update_main_world_render_asset_keys<A: RenderAsset>(
    mut commands: Commands,
    render_asset_key_receiver: Res<RenderAssetKeyReceiver<A>>,
) {
    while let Ok(mut received) = render_asset_key_receiver.try_recv() {
        for (entities, key) in received.drain(..) {
            commands.insert_or_spawn_batch(entities.into_iter().map(move |entity| (entity, key)));
        }
    }
}

/// Stores all GPU representations ([`RenderAsset::PreparedAssets`](RenderAsset::PreparedAsset))
/// of [`RenderAssets`](RenderAsset) as long as they exist.
#[derive(Resource)]
pub struct RenderAssets<A: RenderAsset> {
    asset_id_to_key: HashMap<AssetId<A>, RenderAssetKey<A>>,
    prepared_assets: SlotMap<InnerRenderAssetKey, A::PreparedAsset>,
}

impl<A: RenderAsset> Default for RenderAssets<A> {
    fn default() -> Self {
        Self {
            asset_id_to_key: Default::default(),
            prepared_assets: Default::default(),
        }
    }
}

impl<A: RenderAsset> RenderAssets<A> {
    #[inline]
    pub fn get_with_asset_id(&self, id: impl Into<AssetId<A>>) -> Option<&A::PreparedAsset> {
        let Some(key) = self.asset_id_to_key.get(&id.into()) else {
            return None;
        };
        self.get_with_key(*key)
    }

    #[inline]
    pub fn get_with_key(&self, key: RenderAssetKey<A>) -> Option<&A::PreparedAsset> {
        self.prepared_assets.get(key.inner)
    }

    #[inline]
    pub fn get_mut_with_asset_id(
        &mut self,
        id: impl Into<AssetId<A>>,
    ) -> Option<&mut A::PreparedAsset> {
        let Some(key) = self.asset_id_to_key.get(&id.into()) else {
            return None;
        };
        self.get_mut_with_key(*key)
    }

    #[inline]
    pub fn get_mut_with_key(&mut self, key: RenderAssetKey<A>) -> Option<&mut A::PreparedAsset> {
        self.prepared_assets.get_mut(key.inner)
    }

    pub fn insert(
        &mut self,
        id: impl Into<AssetId<A>>,
        value: A::PreparedAsset,
    ) -> RenderAssetKey<A> {
        let key = RenderAssetKey::<A>::new(self.prepared_assets.insert(value));
        self.asset_id_to_key.insert(id.into(), key);
        key
    }

    pub fn remove(&mut self, id: impl Into<AssetId<A>>) -> Option<A::PreparedAsset> {
        let Some(key) = self.asset_id_to_key.remove(&id.into()) else {
            return None;
        };
        self.prepared_assets.remove(key.inner)
    }

    // pub fn iter(&self) -> impl Iterator<Item = (AssetId<A>, &A::PreparedAsset)> {
    //     self.0.iter().map(|(k, v)| (*k, v))
    // }

    // pub fn iter_mut(&mut self) -> impl Iterator<Item = (AssetId<A>, &mut A::PreparedAsset)> {
    //     self.0.iter_mut().map(|(k, v)| (*k, v))
    // }
}

/// This system extracts all created or modified assets of the corresponding [`RenderAsset`] type
/// into the "render world".
fn extract_render_asset<A: RenderAsset>(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<A>>>,
    assets: Extract<Res<Assets<A>>>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                changed_assets.insert(*id);
            }
            AssetEvent::Removed { id } => {
                changed_assets.remove(id);
                removed.push(*id);
            }
            AssetEvent::LoadedWithDependencies { .. } => {
                // TODO: handle this
            }
        }
    }

    let mut extracted_assets = Vec::new();
    for id in changed_assets.drain() {
        if let Some(asset) = assets.get(id) {
            extracted_assets.push((id, asset.extract_asset()));
        }
    }

    commands.insert_resource(ExtractedAssets {
        extracted: extracted_assets,
        removed,
    });
}

// TODO: consider storing inside system?
/// All assets that should be prepared next frame.
#[derive(Resource)]
pub struct PrepareNextFrameAssets<A: RenderAsset> {
    assets: Vec<(AssetId<A>, A::ExtractedAsset)>,
}

impl<A: RenderAsset> Default for PrepareNextFrameAssets<A> {
    fn default() -> Self {
        Self {
            assets: Default::default(),
        }
    }
}

#[derive(Resource, Deref, DerefMut)]
pub struct RenderAssetKeyUpdates<A: RenderAsset>(Vec<(AssetId<A>, RenderAssetKey<A>)>);

impl<A: RenderAsset> Default for RenderAssetKeyUpdates<A> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// This system prepares all assets of the corresponding [`RenderAsset`] type
/// which where extracted this frame for the GPU.
pub fn prepare_assets<R: RenderAsset>(
    mut extracted_assets: ResMut<ExtractedAssets<R>>,
    mut render_assets: ResMut<RenderAssets<R>>,
    mut key_updates: ResMut<RenderAssetKeyUpdates<R>>,
    mut prepare_next_frame: ResMut<PrepareNextFrameAssets<R>>,
    param: StaticSystemParam<<R as RenderAsset>::Param>,
) {
    let mut param = param.into_inner();
    let queued_assets = std::mem::take(&mut prepare_next_frame.assets);
    for (id, extracted_asset) in queued_assets {
        match R::prepare_asset(extracted_asset, &mut param) {
            Ok(prepared_asset) => {
                let key = render_assets.insert(id, prepared_asset);
                key_updates.push((id, key));
            }
            Err(PrepareAssetError::RetryNextUpdate(extracted_asset)) => {
                prepare_next_frame.assets.push((id, extracted_asset));
            }
        }
    }

    for removed in std::mem::take(&mut extracted_assets.removed) {
        render_assets.remove(removed);
    }

    for (id, extracted_asset) in std::mem::take(&mut extracted_assets.extracted) {
        match R::prepare_asset(extracted_asset, &mut param) {
            Ok(prepared_asset) => {
                let key = render_assets.insert(id, prepared_asset);
                key_updates.push((id, key));
            }
            Err(PrepareAssetError::RetryNextUpdate(extracted_asset)) => {
                prepare_next_frame.assets.push((id, extracted_asset));
            }
        }
    }
}
