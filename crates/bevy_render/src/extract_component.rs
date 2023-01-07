use crate::{
    render_resource::{encase::internal::WriteInto, DynamicUniformBuffer, ShaderType},
    renderer::{RenderDevice, RenderQueue},
    view::ComputedVisibility,
    Extract, RenderApp, RenderStage,
};
use bevy_app::{App, Plugin};
use bevy_asset::{Asset, Handle};
use bevy_ecs::{
    component::Component,
    prelude::*,
    query::{QueryItem, ReadOnlyWorldQuery, WorldQuery},
    system::lifetimeless::Read,
};
use encase::{MaxCapacityArray, ShaderSize};
use std::{marker::PhantomData, num::NonZeroU64, ops::Deref};
use wgpu::BindingResource;

/// Stores the index of a uniform inside of [`ComponentUniforms`].
#[derive(Component)]
pub struct DynamicUniformIndex<C: Component> {
    index: u32,
    marker: PhantomData<C>,
}

impl<C: Component> DynamicUniformIndex<C> {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// Stores the indices of a uniform inside of [`ComponentVecUniforms`].
#[derive(Component)]
pub struct DynamicUniformIndices<C: Component> {
    /// The dynamic offset within the uniform buffer at which to bind
    offset: u32,
    /// The index within the array in the binding at that offset at which the
    /// entity's component data can be looked up in a shader
    index: u32,
    marker: PhantomData<C>,
}

impl<C: Component> DynamicUniformIndices<C> {
    #[inline]
    pub fn offset(&self) -> u32 {
        self.offset
    }

    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// Describes how a component gets extracted for rendering.
///
/// Therefore the component is transferred from the "app world" into the "render world"
/// in the [`RenderStage::Extract`](crate::RenderStage::Extract) step.
pub trait ExtractComponent: Component {
    /// ECS [`WorldQuery`] to fetch the components to extract.
    type Query: WorldQuery + ReadOnlyWorldQuery;
    /// Filters the entities with additional constraints.
    type Filter: WorldQuery + ReadOnlyWorldQuery;

    /// The output from extraction.
    ///
    /// Returning `None` based on the queried item can allow early optimization,
    /// for example if there is an `enabled: bool` field on `Self`, or by only accepting
    /// values within certain thresholds.
    ///
    /// The output may be different from the queried component.
    /// This can be useful for example if only a subset of the fields are useful
    /// in the render world.
    ///
    /// `Out` has a [`Bundle`] trait bound instead of a [`Component`] trait bound in order to allow use cases
    /// such as tuples of components as output.
    type Out: Bundle;

    // TODO: https://github.com/rust-lang/rust/issues/29661
    // type Out: Component = Self;

    /// Defines how the component is transferred into the "render world".
    fn extract_component(item: QueryItem<'_, Self::Query>) -> Option<Self::Out>;
}

/// This plugin prepares the components of the corresponding type for the GPU
/// by transforming them into uniforms.
///
/// They can then be accessed from the [`ComponentUniforms`] resource.
/// For referencing the newly created uniforms a [`DynamicUniformIndex`] is inserted
/// for every processed entity.
///
/// Therefore it sets up the [`RenderStage::Prepare`](crate::RenderStage::Prepare) step
/// for the specified [`ExtractComponent`].
pub struct UniformComponentPlugin<C>(PhantomData<fn() -> C>);

impl<C> Default for UniformComponentPlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: Component + ShaderType + WriteInto + Clone> Plugin for UniformComponentPlugin<C> {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            let limits = render_app
                .world
                .get_resource::<RenderDevice>()
                .expect(
                    "RenderDevice Resource must exist before a UniformComponentVecPlugin is added",
                )
                .limits();
            render_app
                .insert_resource(ComponentUniforms::<C>::from_alignment(
                    limits.min_uniform_buffer_offset_alignment,
                ))
                .add_system_to_stage(RenderStage::Prepare, prepare_uniform_components::<C>);
        }
    }
}

/// Stores all uniforms of the component type.
#[derive(Resource)]
pub struct ComponentUniforms<C: Component + ShaderType> {
    uniforms: DynamicUniformBuffer<C>,
}

impl<C: Component + ShaderType> Deref for ComponentUniforms<C> {
    type Target = DynamicUniformBuffer<C>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.uniforms
    }
}

impl<C: Component + ShaderType + WriteInto> ComponentUniforms<C> {
    pub fn from_alignment(alignment: u32) -> Self {
        Self {
            uniforms: DynamicUniformBuffer::<C>::from_alignment(alignment),
        }
    }

    #[inline]
    pub fn uniforms(&self) -> &DynamicUniformBuffer<C> {
        &self.uniforms
    }
}

/// This system prepares all components of the corresponding component type.
/// They are transformed into uniforms and stored in the [`ComponentUniforms`] resource.
fn prepare_uniform_components<C: Component>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut component_uniforms: ResMut<ComponentUniforms<C>>,
    components: Query<(Entity, &C)>,
) where
    C: ShaderType + WriteInto + Clone,
{
    component_uniforms.uniforms.clear();
    let entities = components
        .iter()
        .map(|(entity, component)| {
            (
                entity,
                DynamicUniformIndex::<C> {
                    index: component_uniforms.uniforms.push(component.clone()),
                    marker: PhantomData,
                },
            )
        })
        .collect::<Vec<_>>();
    commands.insert_or_spawn_batch(entities);

    component_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

/// This plugin prepares the components of the corresponding type for the GPU
/// by transforming them into uniforms.
///
/// They can then be accessed from the [`ComponentVecUniforms`] resource.
/// For referencing the newly created uniforms a [`DynamicUniformIndices`] is inserted
/// for every processed entity.
///
/// Therefore it sets up the [`RenderStage::Prepare`](crate::RenderStage::Prepare) step
/// for the specified [`ExtractComponent`].
pub struct UniformComponentVecPlugin<C>(PhantomData<fn() -> C>);

impl<C> Default for UniformComponentVecPlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

// 1MB else we will make really large arrays on macOS which reports very large
// `max_uniform_buffer_binding_size`. On macOS this ends up being the minimum
// size of the uniform buffer as well as the size of each chunk of data at a
// dynamic offset.
pub const MAX_REASONABLE_UNIFORM_BUFFER_BINDING_SIZE: u32 = 1 << 20;

impl<C: Component + ShaderType + ShaderSize + WriteInto + Clone> Plugin
    for UniformComponentVecPlugin<C>
{
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            let limits = render_app
                .world
                .get_resource::<RenderDevice>()
                .expect(
                    "RenderDevice Resource must exist before a UniformComponentVecPlugin is added",
                )
                .limits();
            render_app
                .insert_resource(ComponentVecUniforms::<C>::new(
                    (limits
                        .max_uniform_buffer_binding_size
                        .min(MAX_REASONABLE_UNIFORM_BUFFER_BINDING_SIZE)
                        as u64
                        / C::min_size().get()) as usize,
                    limits.min_uniform_buffer_offset_alignment,
                ))
                .add_system_to_stage(RenderStage::Prepare, prepare_uniform_component_vecs::<C>);
        }
    }
}

/// Stores all uniforms of the component type.
#[derive(Resource)]
pub struct ComponentVecUniforms<C: Component + ShaderType + ShaderSize> {
    uniforms: DynamicUniformBuffer<MaxCapacityArray<Vec<C>>>,
    temp: MaxCapacityArray<Vec<C>>,
    current_offset: u32,
    dynamic_offset_alignment: u32,
}

#[inline]
fn round_up(v: u64, a: u64) -> u64 {
    ((v + a - 1) / a) * a
}

impl<C: Component + ShaderType + ShaderSize + WriteInto + Clone> ComponentVecUniforms<C> {
    pub fn new(capacity: usize, alignment: u32) -> Self {
        Self {
            temp: MaxCapacityArray(Vec::with_capacity(capacity), capacity),
            current_offset: 0,
            uniforms: DynamicUniformBuffer::<MaxCapacityArray<Vec<C>>>::from_alignment(alignment),
            dynamic_offset_alignment: alignment,
        }
    }

    #[inline]
    pub fn size(&self) -> NonZeroU64 {
        self.temp.size()
    }

    pub fn clear(&mut self) {
        self.uniforms.clear();
        self.current_offset = 0;
        self.temp.0.clear();
    }

    pub fn push(&mut self, component: C) -> DynamicUniformIndices<C> {
        let result = DynamicUniformIndices::<C> {
            offset: self.current_offset,
            index: self.temp.0.len() as u32,
            marker: PhantomData,
        };
        self.temp.0.push(component);
        if self.temp.0.len() == self.temp.1 {
            self.flush();
        }
        result
    }

    pub fn flush(&mut self) {
        self.uniforms.push(self.temp.clone());

        self.current_offset +=
            round_up(self.temp.size().get(), self.dynamic_offset_alignment as u64) as u32;

        self.temp.0.clear();
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if !self.temp.0.is_empty() {
            self.flush();
        }
        self.uniforms.write_buffer(device, queue);
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        self.uniforms.binding()
    }
}

/// This system prepares all components of the corresponding component type.
/// They are transformed into uniforms and stored in the [`ComponentVecUniforms`] resource.
fn prepare_uniform_component_vecs<C: Component + ShaderType + ShaderSize>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut component_vec_uniforms: ResMut<ComponentVecUniforms<C>>,
    components: Query<(Entity, &C)>,
) where
    C: ShaderType + WriteInto + Clone,
{
    component_vec_uniforms.clear();
    let entities = components
        .iter()
        .map(|(entity, component)| (entity, component_vec_uniforms.push(component.clone())))
        .collect::<Vec<_>>();
    commands.insert_or_spawn_batch(entities);

    component_vec_uniforms.write_buffer(&render_device, &render_queue);
}

/// This plugin extracts the components into the "render world".
///
/// Therefore it sets up the [`RenderStage::Extract`](crate::RenderStage::Extract) step
/// for the specified [`ExtractComponent`].
pub struct ExtractComponentPlugin<C, F = ()> {
    only_extract_visible: bool,
    marker: PhantomData<fn() -> (C, F)>,
}

impl<C, F> Default for ExtractComponentPlugin<C, F> {
    fn default() -> Self {
        Self {
            only_extract_visible: false,
            marker: PhantomData,
        }
    }
}

impl<C, F> ExtractComponentPlugin<C, F> {
    pub fn extract_visible() -> Self {
        Self {
            only_extract_visible: true,
            marker: PhantomData,
        }
    }
}

impl<C: ExtractComponent> Plugin for ExtractComponentPlugin<C> {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            if self.only_extract_visible {
                render_app
                    .add_system_to_stage(RenderStage::Extract, extract_visible_components::<C>);
            } else {
                render_app.add_system_to_stage(RenderStage::Extract, extract_components::<C>);
            }
        }
    }
}

impl<T: Asset> ExtractComponent for Handle<T> {
    type Query = Read<Handle<T>>;
    type Filter = ();
    type Out = Handle<T>;

    #[inline]
    fn extract_component(handle: QueryItem<'_, Self::Query>) -> Option<Self::Out> {
        Some(handle.clone_weak())
    }
}

/// This system extracts all components of the corresponding [`ExtractComponent`] type.
fn extract_components<C: ExtractComponent>(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    query: Extract<Query<(Entity, C::Query), C::Filter>>,
) {
    let mut values = Vec::with_capacity(*previous_len);
    for (entity, query_item) in &query {
        if let Some(component) = C::extract_component(query_item) {
            values.push((entity, component));
        }
    }
    *previous_len = values.len();
    commands.insert_or_spawn_batch(values);
}

/// This system extracts all visible components of the corresponding [`ExtractComponent`] type.
fn extract_visible_components<C: ExtractComponent>(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    query: Extract<Query<(Entity, &ComputedVisibility, C::Query), C::Filter>>,
) {
    let mut values = Vec::with_capacity(*previous_len);
    for (entity, computed_visibility, query_item) in &query {
        if computed_visibility.is_visible() {
            if let Some(component) = C::extract_component(query_item) {
                values.push((entity, component));
            }
        }
    }
    *previous_len = values.len();
    commands.insert_or_spawn_batch(values);
}
