use std::{marker::PhantomData, ops::Range};

use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityHashMap},
    prelude::Res,
    system::{Query, StaticSystemParam, SystemParam, SystemParamItem},
};
use nonmax::NonMaxU32;
use wgpu::Limits;

use crate::{
    render_phase::{CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem, RenderPhase},
    render_resource::{
        CachedRenderPipelineId, DynamicOffsetCalculator, GpuArrayBuffer, GpuArrayBufferable,
    },
    renderer::{RenderDevice, RenderQueue},
};

/// Add this component to mesh entities to disable automatic batching
#[derive(Component)]
pub struct NoAutomaticBatching;

/// Data necessary to be equal for two draw commands to be mergeable
///
/// This is based on the following assumptions:
/// - Only entities with prepared assets (pipelines, materials, meshes) are
///   queued to phases
/// - View bindings are constant across a phase for a given draw function as
///   phases are per-view
/// - `batch_and_prepare_render_phase` is the only system that performs this
///   batching and has sole responsibility for preparing the per-object data.
///   As such the mesh binding and dynamic offsets are assumed to only be
///   variable as a result of the `batch_and_prepare_render_phase` system, e.g.
///   due to having to split data across separate uniform bindings within the
///   same buffer due to the maximum uniform buffer binding size.
#[derive(PartialEq)]
struct BatchMeta<T: PartialEq> {
    /// The pipeline id encompasses all pipeline configuration including vertex
    /// buffers and layouts, shaders and their specializations, bind group
    /// layouts, etc.
    pipeline_id: CachedRenderPipelineId,
    /// The draw function id defines the RenderCommands that are called to
    /// set the pipeline and bindings, and make the draw command
    draw_function_id: DrawFunctionId,
    dynamic_offset: Option<NonMaxU32>,
    user_data: T,
}

impl<T: PartialEq> BatchMeta<T> {
    fn new(
        item: &impl CachedRenderPipelinePhaseItem,
        dynamic_offset: Option<NonMaxU32>,
        user_data: T,
    ) -> Self {
        BatchMeta {
            pipeline_id: item.cached_pipeline(),
            draw_function_id: item.draw_function(),
            dynamic_offset,
            user_data,
        }
    }
}

pub trait GetPerInstanceData {
    type Param: SystemParam + 'static;
    /// The per-instance data to be inserted into the [`GpuArrayBuffer`]
    /// containing these data for all instances.
    type InstanceData: GpuArrayBufferable + Sync + Send + 'static;
    /// Get the per-instance data to be inserted into the [`GpuArrayBuffer`].
    fn get_instance_data(
        param: &SystemParamItem<Self::Param>,
        query_item: Entity,
    ) -> Option<Self::InstanceData>;
}

#[derive(Component)]
pub struct PhaseItemOffsets<I: PhaseItem> {
    pub offsets: EntityHashMap<u32>,
    marker: PhantomData<I>,
}

impl<I: PhaseItem> Default for PhaseItemOffsets<I> {
    fn default() -> Self {
        Self {
            offsets: Default::default(),
            marker: Default::default(),
        }
    }
}

pub fn prepare_render_phase<I: CachedRenderPipelinePhaseItem, F: GetPerInstanceData>(
    mut views: Query<(
        &RenderPhase<I>,
        &mut PhaseItemOffsets<I>,
        &mut GpuArrayBuffer<F::InstanceData>,
    )>,
    param: StaticSystemParam<F::Param>,
) {
    let system_param_item = param.into_inner();
    for (phase, mut offsets, mut gpu_array_buffer) in &mut views {
        offsets.offsets.clear();
        let offsets = &mut offsets.offsets;
        for item in &phase.items {
            let Some(instance_data) = F::get_instance_data(&system_param_item, item.entity())
            else {
                continue;
            };
            let buffer_index = gpu_array_buffer.push(instance_data);
            if let Some(dynamic_offset) = buffer_index.dynamic_offset {
                offsets.insert(item.entity(), dynamic_offset.get());
            }
        }
    }
}

/// A trait to support getting data used for batching draw commands via phase
/// items.
pub trait GetBatchData {
    type Param: SystemParam + 'static;
    /// Data used for comparison between phase items. If the pipeline id, draw
    /// function id, per-instance data buffer dynamic offset and this data
    /// matches, the draws can be batched.
    type BatchData: PartialEq;
    /// If the instance can be batched, return the data used for
    /// comparison when deciding whether draws can be batched, else return None
    /// for the `CompareData`.
    fn get_batch_data(
        param: &SystemParamItem<Self::Param>,
        query_item: Entity,
    ) -> Option<Self::BatchData>;
}

#[derive(Component)]
pub struct PhaseItemRanges<I: PhaseItem> {
    pub ranges: EntityHashMap<Range<u32>>,
    marker: PhantomData<I>,
}

impl<I: PhaseItem> Default for PhaseItemRanges<I> {
    fn default() -> Self {
        Self {
            ranges: Default::default(),
            marker: Default::default(),
        }
    }
}

#[derive(Component)]
pub struct PhaseItemOffsetCalculator<T: GpuArrayBufferable, I: PhaseItem> {
    calculator: DynamicOffsetCalculator<T>,
    marker: PhantomData<I>,
}

impl<T: GpuArrayBufferable, I: PhaseItem> PhaseItemOffsetCalculator<T, I> {
    pub fn new(limits: &Limits) -> Self {
        Self {
            calculator: DynamicOffsetCalculator::<T>::new(limits),
            marker: PhantomData,
        }
    }
}

/// Batch the items in a render phase. This means comparing metadata needed to draw each phase item
/// and trying to combine the draws into a batch.
pub fn batch_render_phase<
    I: CachedRenderPipelinePhaseItem,
    F: GetBatchData + GetPerInstanceData,
>(
    mut views: Query<(
        &RenderPhase<I>,
        &mut PhaseItemRanges<I>,
        &mut PhaseItemOffsetCalculator<F::InstanceData, I>,
    )>,
    param: StaticSystemParam<<F as GetBatchData>::Param>,
) {
    let system_param_item = param.into_inner();

    let process_item =
        |calc: &mut DynamicOffsetCalculator<F::InstanceData>, range: &mut Range<u32>, item: &I| {
            let compare_data = F::get_batch_data(&system_param_item, item.entity());

            let (index, dynamic_offset) = calc.indices();
            calc.increment();
            *range = index..index + 1;

            if I::AUTOMATIC_BATCHING {
                compare_data.map(|compare_data| BatchMeta::new(item, dynamic_offset, compare_data))
            } else {
                None
            }
        };

    for (phase, mut ranges, mut offset_calculator) in &mut views {
        ranges.ranges.clear();
        let ranges = &mut ranges.ranges;

        offset_calculator.calculator.clear();
        let calc = &mut offset_calculator.calculator;

        let items = phase.items.iter().map(|item| {
            let mut range = 0..1;
            let batch_data = process_item(calc, &mut range, item);
            (item.entity(), range, batch_data)
        });
        items.reduce(
            |(start_entity, mut start_range, prev_batch_meta), (entity, range, batch_meta)| {
                if batch_meta.is_some() && prev_batch_meta == batch_meta {
                    start_range.end = range.end;
                    *ranges.entry(start_entity).or_default() = start_range.clone();
                    (start_entity, start_range, prev_batch_meta)
                } else {
                    *ranges.entry(start_entity).or_default() = start_range.clone();
                    *ranges.entry(entity).or_default() = range.clone();
                    (entity, range, batch_meta)
                }
            },
        );
        for item in &phase.items {
            dbg!(item.entity());
            dbg!(ranges.get(&item.entity()));
        }
    }
}

pub fn write_batched_instance_buffer<F: GetPerInstanceData>(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu_array_buffers: Query<&mut GpuArrayBuffer<F::InstanceData>>,
) {
    for mut gpu_array_buffer in &mut gpu_array_buffers {
        gpu_array_buffer.write_buffer(&render_device, &render_queue);
        gpu_array_buffer.clear();
    }
}
