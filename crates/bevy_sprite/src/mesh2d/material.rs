use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{Asset, AssetApp, AssetEvent, AssetId, AssetServer, Assets, Handle};
use bevy_core_pipeline::{
    core_2d::Transparent2d,
    tonemapping::{DebandDither, Tonemapping},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::entity::EntityHashMap;
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::SRes, SystemParamItem},
};
use bevy_log::error;
use bevy_render::{
    camera::{CachedEntityPipelines, CachedViewPipelines},
    mesh::{Mesh, MeshVertexBufferLayout},
    prelude::Image,
    render_asset::{prepare_assets, RenderAssets},
    render_phase::{
        AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
        RenderPhase, SetItemPipeline, TrackedRenderPass,
    },
    render_resource::{
        AsBindGroup, AsBindGroupError, BindGroup, BindGroupId, BindGroupLayout,
        OwnedBindingResource, PipelineCache, RenderPipelineDescriptor, Shader, ShaderRef,
        SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
    },
    renderer::RenderDevice,
    texture::FallbackImage,
    view::{ExtractedView, InheritedVisibility, Msaa, ViewVisibility, Visibility, VisibleEntities},
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::components::{GlobalTransform, Transform};
use bevy_utils::{
    slotmap::{new_key_type, SlotMap},
    FloatOrd, HashMap, HashSet,
};
use crossbeam_channel::{Receiver, Sender};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::{
    DrawMesh2d, Mesh2dHandle, Mesh2dPipeline, Mesh2dPipelineKey, RenderMesh2dInstances,
    SetMesh2dBindGroup, SetMesh2dViewBindGroup,
};

/// Materials are used alongside [`Material2dPlugin`] and [`MaterialMesh2dBundle`]
/// to spawn entities that are rendered with a specific [`Material2d`] type. They serve as an easy to use high level
/// way to render [`Mesh2dHandle`] entities with custom shader logic.
///
/// Material2ds must implement [`AsBindGroup`] to define how data will be transferred to the GPU and bound in shaders.
/// [`AsBindGroup`] can be derived, which makes generating bindings straightforward. See the [`AsBindGroup`] docs for details.
///
/// # Example
///
/// Here is a simple Material2d implementation. The [`AsBindGroup`] derive has many features. To see what else is available,
/// check out the [`AsBindGroup`] documentation.
/// ```
/// # use bevy_sprite::{Material2d, MaterialMesh2dBundle};
/// # use bevy_ecs::prelude::*;
/// # use bevy_reflect::TypePath;
/// # use bevy_render::{render_resource::{AsBindGroup, ShaderRef}, texture::Image, color::Color};
/// # use bevy_asset::{Handle, AssetServer, Assets, Asset};
///
/// #[derive(AsBindGroup, Debug, Clone, Asset, TypePath)]
/// pub struct CustomMaterial {
///     // Uniform bindings must implement `ShaderType`, which will be used to convert the value to
///     // its shader-compatible equivalent. Most core math types already implement `ShaderType`.
///     #[uniform(0)]
///     color: Color,
///     // Images can be bound as textures in shaders. If the Image's sampler is also needed, just
///     // add the sampler attribute with a different binding index.
///     #[texture(1)]
///     #[sampler(2)]
///     color_texture: Handle<Image>,
/// }
///
/// // All functions on `Material2d` have default impls. You only need to implement the
/// // functions that are relevant for your material.
/// impl Material2d for CustomMaterial {
///     fn fragment_shader() -> ShaderRef {
///         "shaders/custom_material.wgsl".into()
///     }
/// }
///
/// // Spawn an entity using `CustomMaterial`.
/// fn setup(mut commands: Commands, mut materials: ResMut<Assets<CustomMaterial>>, asset_server: Res<AssetServer>) {
///     commands.spawn(MaterialMesh2dBundle {
///         material: materials.add(CustomMaterial {
///             color: Color::RED,
///             color_texture: asset_server.load("some_image.png"),
///         }),
///         ..Default::default()
///     });
/// }
/// ```
/// In WGSL shaders, the material's binding would look like this:
///
/// ```wgsl
/// struct CustomMaterial {
///     color: vec4<f32>,
/// }
///
/// @group(2) @binding(0) var<uniform> material: CustomMaterial;
/// @group(2) @binding(1) var color_texture: texture_2d<f32>;
/// @group(2) @binding(2) var color_sampler: sampler;
/// ```
pub trait Material2d: AsBindGroup + Asset + Clone + Sized {
    /// Returns this material's vertex shader. If [`ShaderRef::Default`] is returned, the default mesh vertex shader
    /// will be used.
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's fragment shader. If [`ShaderRef::Default`] is returned, the default mesh fragment shader
    /// will be used.
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Add a bias to the view depth of the mesh which can be used to force a specific render order.
    #[inline]
    fn depth_bias(&self) -> f32 {
        0.0
    }

    /// Customizes the default [`RenderPipelineDescriptor`].
    #[allow(unused_variables)]
    #[inline]
    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayout,
        key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        Ok(())
    }
}

/// Adds the necessary ECS resources and render logic to enable rendering entities using the given [`Material2d`]
/// asset type (which includes [`Material2d`] types).
pub struct Material2dPlugin<M: Material2d>(PhantomData<M>);

impl<M: Material2d> Default for Material2dPlugin<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<M: Material2d> Plugin for Material2dPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        let (render_material2d_key_sender, render_material2d_key_receiver) =
            create_render_material_key_channel::<M>();
        app.init_asset::<M>()
            .insert_resource(render_material2d_key_receiver.clone())
            .add_systems(PostUpdate, update_main_world_render_material2d_keys::<M>);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent2d, DrawMaterial2d<M>>()
                .init_resource::<ExtractedMaterials2d<M>>()
                .init_resource::<RenderMaterials2d<M>>()
                .init_resource::<RenderMaterial2dKeyUpdates<M>>()
                .insert_resource(render_material2d_key_sender)
                .insert_resource(render_material2d_key_receiver)
                .init_resource::<RenderMaterial2dInstances<M>>()
                .init_resource::<CachedEntityPipelines<View2dPipelineKey>>()
                .init_resource::<SpecializedMeshPipelines<Material2dPipeline<M>>>()
                .add_systems(
                    ExtractSchedule,
                    (extract_materials_2d::<M>, extract_material_meshes_2d::<M>),
                )
                .add_systems(
                    Render,
                    (
                        update_view_2d_pipeline_keys.after(RenderSet::ExtractCommands),
                        prepare_materials_2d::<M>
                            .in_set(RenderSet::PrepareAssets)
                            .after(prepare_assets::<Image>),
                        update_render_world_render_material2d_keys::<M>
                            .in_set(RenderSet::PrepareAssets)
                            .after(prepare_materials_2d::<M>),
                        queue_material2d_meshes::<M>
                            .in_set(RenderSet::QueueMeshes)
                            .after(prepare_materials_2d::<M>)
                            .after(update_view_2d_pipeline_keys),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<Material2dPipeline<M>>();
        }
    }
}

pub struct RenderMaterial2dInstance<M: Material2d> {
    asset_id: Option<AssetId<M>>,
    key: Option<RenderMaterial2dKey>,
}

#[derive(Resource, Deref, DerefMut)]
pub struct RenderMaterial2dInstances<M: Material2d>(EntityHashMap<RenderMaterial2dInstance<M>>);

impl<M: Material2d> Default for RenderMaterial2dInstances<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

fn extract_material_meshes_2d<M: Material2d>(
    mut material_instances: ResMut<RenderMaterial2dInstances<M>>,
    query: Extract<
        Query<(
            Entity,
            &ViewVisibility,
            Ref<Handle<M>>,
            Option<&RenderMaterial2dKey>,
        )>,
    >,
) {
    material_instances.clear();
    for (entity, view_visibility, handle, key) in &query {
        if view_visibility.get() {
            material_instances.insert(
                entity,
                RenderMaterial2dInstance {
                    asset_id: if key.is_none() || handle.is_changed() {
                        Some(handle.id())
                    } else {
                        None
                    },
                    key: key.cloned(),
                },
            );
        }
    }
}

/// Render pipeline data for a given [`Material2d`]
#[derive(Resource)]
pub struct Material2dPipeline<M: Material2d> {
    pub mesh2d_pipeline: Mesh2dPipeline,
    pub material2d_layout: BindGroupLayout,
    pub vertex_shader: Option<Handle<Shader>>,
    pub fragment_shader: Option<Handle<Shader>>,
    marker: PhantomData<M>,
}

pub struct Material2dKey<M: Material2d> {
    pub mesh_key: Mesh2dPipelineKey,
    pub bind_group_data: M::Data,
}

impl<M: Material2d> Eq for Material2dKey<M> where M::Data: PartialEq {}

impl<M: Material2d> PartialEq for Material2dKey<M>
where
    M::Data: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.mesh_key == other.mesh_key && self.bind_group_data == other.bind_group_data
    }
}

impl<M: Material2d> Clone for Material2dKey<M>
where
    M::Data: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mesh_key: self.mesh_key,
            bind_group_data: self.bind_group_data.clone(),
        }
    }
}

impl<M: Material2d> Hash for Material2dKey<M>
where
    M::Data: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.mesh_key.hash(state);
        self.bind_group_data.hash(state);
    }
}

impl<M: Material2d> Clone for Material2dPipeline<M> {
    fn clone(&self) -> Self {
        Self {
            mesh2d_pipeline: self.mesh2d_pipeline.clone(),
            material2d_layout: self.material2d_layout.clone(),
            vertex_shader: self.vertex_shader.clone(),
            fragment_shader: self.fragment_shader.clone(),
            marker: PhantomData,
        }
    }
}

impl<M: Material2d> SpecializedMeshPipeline for Material2dPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = Material2dKey<M>;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh2d_pipeline.specialize(key.mesh_key, layout)?;
        if let Some(vertex_shader) = &self.vertex_shader {
            descriptor.vertex.shader = vertex_shader.clone();
        }

        if let Some(fragment_shader) = &self.fragment_shader {
            descriptor.fragment.as_mut().unwrap().shader = fragment_shader.clone();
        }
        descriptor.layout = vec![
            self.mesh2d_pipeline.view_layout.clone(),
            self.mesh2d_pipeline.mesh_layout.clone(),
            self.material2d_layout.clone(),
        ];

        M::specialize(&mut descriptor, layout, key)?;
        Ok(descriptor)
    }
}

impl<M: Material2d> FromWorld for Material2dPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();
        let material2d_layout = M::bind_group_layout(render_device);

        Material2dPipeline {
            mesh2d_pipeline: world.resource::<Mesh2dPipeline>().clone(),
            material2d_layout,
            vertex_shader: match M::vertex_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            fragment_shader: match M::fragment_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            marker: PhantomData,
        }
    }
}

type DrawMaterial2d<M> = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetMesh2dBindGroup<1>,
    SetMaterial2dBindGroup<M, 2>,
    DrawMesh2d,
);

pub struct SetMaterial2dBindGroup<M: Material2d, const I: usize>(PhantomData<M>);
impl<P: PhaseItem, M: Material2d, const I: usize> RenderCommand<P>
    for SetMaterial2dBindGroup<M, I>
{
    type Param = (
        SRes<RenderMaterials2d<M>>,
        SRes<RenderMaterial2dInstances<M>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        (materials, material_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let materials = materials.into_inner();
        let material_instances = material_instances.into_inner();
        let Some(material_instance) = material_instances.get(&item.entity()) else {
            return RenderCommandResult::Failure;
        };
        let Some(material_key) = material_instance.key else {
            return RenderCommandResult::Failure;
        };
        let Some(material2d) = materials.get_with_key(material_key) else {
            return RenderCommandResult::Failure;
        };
        pass.set_bind_group(I, &material2d.bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub const fn tonemapping_pipeline_key(tonemapping: Tonemapping) -> Mesh2dPipelineKey {
    match tonemapping {
        Tonemapping::None => Mesh2dPipelineKey::TONEMAP_METHOD_NONE,
        Tonemapping::Reinhard => Mesh2dPipelineKey::TONEMAP_METHOD_REINHARD,
        Tonemapping::ReinhardLuminance => Mesh2dPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE,
        Tonemapping::AcesFitted => Mesh2dPipelineKey::TONEMAP_METHOD_ACES_FITTED,
        Tonemapping::AgX => Mesh2dPipelineKey::TONEMAP_METHOD_AGX,
        Tonemapping::SomewhatBoringDisplayTransform => {
            Mesh2dPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
        }
        Tonemapping::TonyMcMapface => Mesh2dPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
        Tonemapping::BlenderFilmic => Mesh2dPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
    }
}

#[derive(Component, Debug, PartialEq, Eq, Default)]
pub struct View2dPipelineKey(Mesh2dPipelineKey);

pub fn update_view_2d_pipeline_keys(
    mut commands: Commands,
    msaa: Res<Msaa>,
    mut views: Query<
        (
            Entity,
            &ExtractedView,
            Option<&Tonemapping>,
            Option<&DebandDither>,
        ),
        With<RenderPhase<Transparent2d>>,
    >,
) {
    for (entity, view, tonemapping, dither) in &mut views {
        let mut view_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr);

        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= Mesh2dPipelineKey::TONEMAP_IN_SHADER;
                view_key |= tonemapping_pipeline_key(*tonemapping);
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= Mesh2dPipelineKey::DEBAND_DITHER;
            }
        }

        commands.entity(entity).insert(View2dPipelineKey(view_key));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn queue_material2d_meshes<M: Material2d>(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    material2d_pipeline: Res<Material2dPipeline<M>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<Material2dPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderMaterials2d<M>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    mut cached_pipelines: ResMut<CachedEntityPipelines<View2dPipelineKey>>,
    render_material_instances: Res<RenderMaterial2dInstances<M>>,
    mut views: Query<(
        Entity,
        &View2dPipelineKey,
        &VisibleEntities,
        &mut RenderPhase<Transparent2d>,
    )>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    if render_material_instances.is_empty() {
        return;
    }

    for (view_entity, view_key, visible_entities, mut transparent_phase) in &mut views {
        let draw_transparent_pbr = transparent_draw_functions.read().id::<DrawMaterial2d<M>>();

        for visible_entity in &visible_entities.entities {
            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };
            let Some(material_instance) = render_material_instances.get(visible_entity) else {
                continue;
            };
            let Some(material_key) = material_instance.key else {
                continue;
            };
            let Some(material2d) = render_materials.get_with_key(material_key) else {
                continue;
            };
            let mut pipeline_id = None;
            match cached_pipelines.get(visible_entity) {
                Some(cached_view_pipelines) => match cached_view_pipelines.get(&view_entity) {
                    Some(cached_view_pipeline) => {
                        if cached_view_pipeline.view_key == *view_key {
                            pipeline_id = cached_view_pipeline.pipeline;
                        }
                    }
                    None => {}
                },
                None => {
                    cached_pipelines.insert(*visible_entity, CachedViewPipelines::default());
                }
            };
            if pipeline_id.is_none() {
                let Some(mesh_asset_key) = mesh_instance.mesh_asset_key else {
                    continue;
                };
                let Some(mesh) = render_meshes.get_with_key(mesh_asset_key) else {
                    continue;
                };
                let mesh_key = view_key.0
                    | Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology);

                let pipeline = pipelines.specialize(
                    &pipeline_cache,
                    &material2d_pipeline,
                    Material2dKey {
                        mesh_key,
                        bind_group_data: material2d.key.clone(),
                    },
                    &mesh.layout,
                );

                pipeline_id = match pipeline {
                    Ok(id) => Some(id),
                    Err(err) => {
                        error!("{}", err);
                        continue;
                    }
                };
            }

            mesh_instance.material_bind_group_id = material2d.get_bind_group_id();

            let mesh_z = mesh_instance.transforms.transform.translation.z;
            transparent_phase.add(Transparent2d {
                entity: *visible_entity,
                draw_function: draw_transparent_pbr,
                pipeline: pipeline_id.unwrap(),
                // NOTE: Back-to-front ordering for transparent with ascending sort means far should have the
                // lowest sort key and getting closer should increase. As we have
                // -z in front of the camera, the largest distance is -far with values increasing toward the
                // camera. As such we can just use mesh_z as the distance
                sort_key: FloatOrd(mesh_z + material2d.depth_bias),
                // Batching is done in batch_and_prepare_render_phase
                batch_range: 0..1,
                dynamic_offset: None,
            });
        }
    }
}

#[derive(Component, Clone, Copy, Default, PartialEq, Eq, Deref, DerefMut)]
pub struct Material2dBindGroupId(Option<BindGroupId>);

/// Data prepared for a [`Material2d`] instance.
pub struct PreparedMaterial2d<T: Material2d> {
    pub bindings: Vec<(u32, OwnedBindingResource)>,
    pub bind_group: BindGroup,
    pub key: T::Data,
    pub depth_bias: f32,
}

impl<T: Material2d> PreparedMaterial2d<T> {
    pub fn get_bind_group_id(&self) -> Material2dBindGroupId {
        Material2dBindGroupId(Some(self.bind_group.id()))
    }
}

#[derive(Resource)]
pub struct ExtractedMaterials2d<M: Material2d> {
    extracted: Vec<(AssetId<M>, M)>,
    removed: Vec<AssetId<M>>,
}

impl<M: Material2d> Default for ExtractedMaterials2d<M> {
    fn default() -> Self {
        Self {
            extracted: Default::default(),
            removed: Default::default(),
        }
    }
}

new_key_type! {
    #[derive(Component)]
    pub struct RenderMaterial2dKey;
}

#[derive(Resource, Clone)]
pub struct RenderMaterial2dKeyReceiver<M: Material2d> {
    pub inner: Receiver<Vec<(Vec<Entity>, RenderMaterial2dKey)>>,
    marker: PhantomData<M>,
}

impl<M: Material2d> RenderMaterial2dKeyReceiver<M> {
    pub fn new(receiver: Receiver<Vec<(Vec<Entity>, RenderMaterial2dKey)>>) -> Self {
        Self {
            inner: receiver,
            marker: PhantomData,
        }
    }
}

impl<M: Material2d> Deref for RenderMaterial2dKeyReceiver<M> {
    type Target = Receiver<Vec<(Vec<Entity>, RenderMaterial2dKey)>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M: Material2d> DerefMut for RenderMaterial2dKeyReceiver<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Resource)]
pub struct RenderMaterial2dKeySender<M: Material2d> {
    pub inner: Sender<Vec<(Vec<Entity>, RenderMaterial2dKey)>>,
    marker: PhantomData<M>,
}

impl<M: Material2d> RenderMaterial2dKeySender<M> {
    pub fn new(receiver: Sender<Vec<(Vec<Entity>, RenderMaterial2dKey)>>) -> Self {
        Self {
            inner: receiver,
            marker: PhantomData,
        }
    }
}

impl<M: Material2d> Deref for RenderMaterial2dKeySender<M> {
    type Target = Sender<Vec<(Vec<Entity>, RenderMaterial2dKey)>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M: Material2d> DerefMut for RenderMaterial2dKeySender<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub fn create_render_material_key_channel<M: Material2d>(
) -> (RenderMaterial2dKeySender<M>, RenderMaterial2dKeyReceiver<M>) {
    let (s, r) = crossbeam_channel::unbounded::<Vec<(Vec<Entity>, RenderMaterial2dKey)>>();
    (
        RenderMaterial2dKeySender::<M>::new(s),
        RenderMaterial2dKeyReceiver::<M>::new(r),
    )
}

/// Stores all prepared representations of [`Material2d`] assets for as long as they exist.
#[derive(Resource)]
pub struct RenderMaterials2d<T: Material2d> {
    asset_id_to_key: HashMap<AssetId<T>, RenderMaterial2dKey>,
    prepared_materials: SlotMap<RenderMaterial2dKey, PreparedMaterial2d<T>>,
}

impl<T: Material2d> RenderMaterials2d<T> {
    #[inline]
    pub fn insert(
        &mut self,
        asset_id: impl Into<AssetId<T>>,
        prepared_material: PreparedMaterial2d<T>,
    ) -> RenderMaterial2dKey {
        let key = self.prepared_materials.insert(prepared_material);
        self.asset_id_to_key.insert(asset_id.into(), key);
        key
    }

    #[inline]
    pub fn remove(&mut self, asset_id: impl Into<AssetId<T>>) {
        let Some(key) = self.asset_id_to_key.remove::<AssetId<T>>(&asset_id.into()) else {
            return;
        };
        self.prepared_materials.remove(key);
    }

    #[inline]
    pub fn get_with_asset_id(
        &self,
        asset_id: impl Into<AssetId<T>>,
    ) -> Option<&PreparedMaterial2d<T>> {
        self.asset_id_to_key
            .get::<AssetId<T>>(&asset_id.into())
            .and_then(|key| self.prepared_materials.get(*key))
    }

    #[inline]
    pub fn get_with_key(&self, key: RenderMaterial2dKey) -> Option<&PreparedMaterial2d<T>> {
        self.prepared_materials.get(key)
    }
}

impl<T: Material2d> Default for RenderMaterials2d<T> {
    fn default() -> Self {
        Self {
            asset_id_to_key: Default::default(),
            prepared_materials: Default::default(),
        }
    }
}

/// This system extracts all created or modified assets of the corresponding [`Material2d`] type
/// into the "render world".
pub fn extract_materials_2d<M: Material2d>(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<M>>>,
    assets: Extract<Res<Assets<M>>>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.read() {
        #[allow(clippy::match_same_arms)]
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                changed_assets.insert(*id);
            }
            AssetEvent::Removed { id } => {
                changed_assets.remove(id);
                removed.push(*id);
            }
            AssetEvent::Unused { .. } => {}
            AssetEvent::LoadedWithDependencies { .. } => {
                // TODO: handle this
            }
        }
    }

    let mut extracted_assets = Vec::new();
    for id in changed_assets.drain() {
        if let Some(asset) = assets.get(id) {
            extracted_assets.push((id, asset.clone()));
        }
    }

    commands.insert_resource(ExtractedMaterials2d {
        extracted: extracted_assets,
        removed,
    });
}

/// All [`Material2d`] values of a given type that should be prepared next frame.
pub struct PrepareNextFrameMaterials<M: Material2d> {
    assets: Vec<(AssetId<M>, M)>,
}

impl<M: Material2d> Default for PrepareNextFrameMaterials<M> {
    fn default() -> Self {
        Self {
            assets: Default::default(),
        }
    }
}

/// This system prepares all assets of the corresponding [`Material2d`] type
/// which where extracted this frame for the GPU.
pub fn prepare_materials_2d<M: Material2d>(
    mut prepare_next_frame: Local<PrepareNextFrameMaterials<M>>,
    mut extracted_assets: ResMut<ExtractedMaterials2d<M>>,
    mut render_materials: ResMut<RenderMaterials2d<M>>,
    mut key_updates: ResMut<RenderMaterial2dKeyUpdates<M>>,
    render_device: Res<RenderDevice>,
    images: Res<RenderAssets<Image>>,
    fallback_image: Res<FallbackImage>,
    pipeline: Res<Material2dPipeline<M>>,
) {
    let queued_assets = std::mem::take(&mut prepare_next_frame.assets);
    for (id, material) in queued_assets {
        match prepare_material2d(
            &material,
            &render_device,
            &images,
            &fallback_image,
            &pipeline,
        ) {
            Ok(prepared_asset) => {
                let key = render_materials.insert(id, prepared_asset);
                key_updates.push((id, key));
            }
            Err(AsBindGroupError::RetryNextUpdate) => {
                prepare_next_frame.assets.push((id, material));
            }
        }
    }

    for removed in std::mem::take(&mut extracted_assets.removed) {
        render_materials.remove(removed);
    }

    for (asset_id, material) in std::mem::take(&mut extracted_assets.extracted) {
        match prepare_material2d(
            &material,
            &render_device,
            &images,
            &fallback_image,
            &pipeline,
        ) {
            Ok(prepared_asset) => {
                let key = render_materials.insert(asset_id, prepared_asset);
                key_updates.push((asset_id, key));
            }
            Err(AsBindGroupError::RetryNextUpdate) => {
                prepare_next_frame.assets.push((asset_id, material));
            }
        }
    }
}

pub fn update_main_world_render_material2d_keys<M: Material2d>(
    mut commands: Commands,
    render_material2d_key_receiver: Res<RenderMaterial2dKeyReceiver<M>>,
) {
    while let Ok(mut received) = render_material2d_key_receiver.try_recv() {
        for (entities, key) in received.drain(..) {
            commands.insert_or_spawn_batch(entities.into_iter().map(move |entity| (entity, key)));
        }
    }
}

#[derive(Resource, Deref, DerefMut)]
pub struct RenderMaterial2dKeyUpdates<M: Material2d>(Vec<(AssetId<M>, RenderMaterial2dKey)>);

impl<M: Material2d> Default for RenderMaterial2dKeyUpdates<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub fn update_render_world_render_material2d_keys<M: Material2d>(
    mut key_updates: ResMut<RenderMaterial2dKeyUpdates<M>>,
    mut render_material_instances: ResMut<RenderMaterial2dInstances<M>>,
    mut key_map: Local<HashMap<AssetId<M>, RenderMaterial2dKey>>,
    render_material2d_key_sender: Res<RenderMaterial2dKeySender<M>>,
) {
    let mut map: HashMap<RenderMaterial2dKey, Vec<Entity>> = HashMap::new();
    key_map.extend(key_updates.drain(..));
    for (&entity, material_instance) in render_material_instances.iter_mut() {
        if material_instance.key.is_some() {
            continue;
        }
        material_instance.key = key_map
            .get(material_instance.asset_id.as_ref().unwrap())
            .cloned();
        if let Some(key) = material_instance.key {
            map.entry(key).or_default().push(entity);
        }
    }
    let to_send = map.into_iter().map(|(k, v)| (v, k)).collect::<Vec<_>>();
    if !to_send.is_empty() {
        match render_material2d_key_sender.try_send(to_send) {
            Ok(_) => {}
            Err(_) => panic!("Failed to send"),
        }
    }
}

fn prepare_material2d<M: Material2d>(
    material: &M,
    render_device: &RenderDevice,
    images: &RenderAssets<Image>,
    fallback_image: &FallbackImage,
    pipeline: &Material2dPipeline<M>,
) -> Result<PreparedMaterial2d<M>, AsBindGroupError> {
    let prepared = material.as_bind_group(
        &pipeline.material2d_layout,
        render_device,
        images,
        fallback_image,
    )?;
    Ok(PreparedMaterial2d {
        bindings: prepared.bindings,
        bind_group: prepared.bind_group,
        key: prepared.data,
        depth_bias: material.depth_bias(),
    })
}

/// A component bundle for entities with a [`Mesh2dHandle`] and a [`Material2d`].
#[derive(Bundle, Clone)]
pub struct MaterialMesh2dBundle<M: Material2d> {
    pub mesh: Mesh2dHandle,
    pub material: Handle<M>,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    /// User indication of whether an entity is visible
    pub visibility: Visibility,
    // Inherited visibility of an entity.
    pub inherited_visibility: InheritedVisibility,
    // Indication of whether an entity is visible in any view.
    pub view_visibility: ViewVisibility,
}

impl<M: Material2d> Default for MaterialMesh2dBundle<M> {
    fn default() -> Self {
        Self {
            mesh: Default::default(),
            material: Default::default(),
            transform: Default::default(),
            global_transform: Default::default(),
            visibility: Default::default(),
            inherited_visibility: Default::default(),
            view_visibility: Default::default(),
        }
    }
}
