//! Convenience logic for turning components from the main world into render
//! instances in the render world.
//!
//! This is essentially the same as the `extract_component` module, but
//! higher-performance because it avoids the ECS overhead.

use std::{cell::Cell, marker::PhantomData};

use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetId, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    prelude::Entity,
    query::{QueryItem, ReadOnlyWorldQuery, WorldQuery},
    system::{lifetimeless::Read, Local, Query, ResMut, Resource},
};
use bevy_utils::EntityHashMap;
use thread_local::ThreadLocal;

use crate::{prelude::ViewVisibility, Extract, ExtractSchedule, RenderApp};

/// Describes how to extract data needed for rendering from a component or
/// components.
///
/// Before rendering, any applicable components will be transferred from the
/// main world to the render world in the [`ExtractSchedule`] step.
///
/// This is essentially the same as
/// [`ExtractComponent`](crate::extract_component::ExtractComponent), but
/// higher-performance because it avoids the ECS overhead.
pub trait RenderInstance: Send + Sync + Sized + 'static {
    /// ECS [`WorldQuery`] to fetch the components to extract.
    type Query: WorldQuery + ReadOnlyWorldQuery;
    /// Filters the entities with additional constraints.
    type Filter: WorldQuery + ReadOnlyWorldQuery;

    /// Defines how the component is transferred into the "render world".
    fn extract_to_render_instance(item: QueryItem<'_, Self::Query>) -> Option<Self>;
}

/// This plugin extracts one or more components into the "render world" as
/// render instances.
///
/// Therefore it sets up the [`ExtractSchedule`] step for the specified
/// [`RenderInstances`].
#[derive(Default)]
pub struct RenderInstancePlugin<RI: RenderInstance, const VISIBLE: bool, const PARALLEL: bool>(
    PhantomData<fn() -> RI>,
);

/// Stores all render instances of a type in the render world.
#[derive(Resource, Deref, DerefMut)]
pub struct RenderInstances<RI>(EntityHashMap<Entity, RI>)
where
    RI: RenderInstance;

impl<RI> Default for RenderInstances<RI>
where
    RI: RenderInstance,
{
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<RI, const VISIBLE: bool, const PARALLEL: bool> RenderInstancePlugin<RI, VISIBLE, PARALLEL>
where
    RI: RenderInstance,
{
    /// Creates a new [`RenderInstancePlugin`] that unconditionally extracts to
    /// the render world, whether the entity is visible or not.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<RI, const VISIBLE: bool, const PARALLEL: bool> Plugin
    for RenderInstancePlugin<RI, VISIBLE, PARALLEL>
where
    RI: RenderInstance,
{
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<RenderInstances<RI>>();
            // wasm32 currently doesn't support threading so we avoid the overhead automatically
            // in this plugin by only running extraction single-threaded in that case.
            let parallel = PARALLEL && !cfg!(target_arch = "wasm32");
            match (VISIBLE, parallel) {
                (true, true) => render_app.add_systems(
                    ExtractSchedule,
                    extract_visible_to_render_instances_in_parallel::<RI>,
                ),
                (true, false) => render_app
                    .add_systems(ExtractSchedule, extract_visible_to_render_instances::<RI>),
                (false, true) => render_app.add_systems(
                    ExtractSchedule,
                    extract_to_render_instances_in_parallel::<RI>,
                ),
                (false, false) => {
                    render_app.add_systems(ExtractSchedule, extract_to_render_instances::<RI>)
                }
            };
        }
    }
}

pub fn extract_to_render_instances_in_parallel<RI>(
    mut instances: ResMut<RenderInstances<RI>>,
    mut thread_local_queues: Local<ThreadLocal<Cell<Vec<(Entity, RI)>>>>,
    query: Extract<Query<(Entity, RI::Query), RI::Filter>>,
) where
    RI: RenderInstance,
{
    query.par_iter().for_each(|(entity, other)| {
        if let Some(render_instance) = RI::extract_to_render_instance(other) {
            let tls = thread_local_queues.get_or_default();
            let mut queue = tls.take();
            queue.push((entity, render_instance));
            tls.set(queue);
        }
    });

    instances.clear();
    for queue in thread_local_queues.iter_mut() {
        instances.extend(queue.get_mut().drain(..));
    }
}

pub fn extract_to_render_instances<RI>(
    mut instances: ResMut<RenderInstances<RI>>,
    query: Extract<Query<(Entity, RI::Query), RI::Filter>>,
) where
    RI: RenderInstance,
{
    instances.clear();
    for (entity, other) in &query {
        if let Some(render_instance) = RI::extract_to_render_instance(other) {
            instances.insert(entity, render_instance);
        }
    }
}

pub fn extract_visible_to_render_instances<RI>(
    mut instances: ResMut<RenderInstances<RI>>,
    query: Extract<Query<(Entity, &ViewVisibility, RI::Query), RI::Filter>>,
) where
    RI: RenderInstance,
{
    instances.clear();
    for (entity, view_visibility, other) in &query {
        if view_visibility.get() {
            if let Some(render_instance) = RI::extract_to_render_instance(other) {
                instances.insert(entity, render_instance);
            }
        }
    }
}

pub fn extract_visible_to_render_instances_in_parallel<RI>(
    mut instances: ResMut<RenderInstances<RI>>,
    mut thread_local_queues: Local<ThreadLocal<Cell<Vec<(Entity, RI)>>>>,
    query: Extract<Query<(Entity, &ViewVisibility, RI::Query), RI::Filter>>,
) where
    RI: RenderInstance,
{
    query
        .par_iter()
        .for_each(|(entity, view_visibility, other)| {
            if !view_visibility.get() {
                return;
            }
            if let Some(render_instance) = RI::extract_to_render_instance(other) {
                let tls = thread_local_queues.get_or_default();
                let mut queue = tls.take();
                queue.push((entity, render_instance));
                tls.set(queue);
            }
        });

    instances.clear();
    for queue in thread_local_queues.iter_mut() {
        instances.extend(queue.get_mut().drain(..));
    }
}

impl<A> RenderInstance for AssetId<A>
where
    A: Asset,
{
    type Query = Read<Handle<A>>;
    type Filter = ();

    #[inline]
    fn extract_to_render_instance(item: QueryItem<'_, Self::Query>) -> Option<Self> {
        Some(item.id())
    }
}
