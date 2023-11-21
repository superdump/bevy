pub mod copy_lighting_id;
pub mod node;

use std::{cmp::Reverse, ops::Range};

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_render::{
    mesh::Mesh,
    render_phase::{CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem},
    render_resource::{BindGroupId, CachedRenderPipelineId, TextureFormat},
};
use bevy_utils::{nonmax::NonMaxU32, FloatOrd};

pub const DEFERRED_PREPASS_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;
pub const DEFERRED_LIGHTING_PASS_ID_FORMAT: TextureFormat = TextureFormat::R8Uint;
pub const DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth16Unorm;

/// Opaque phase of the 3D Deferred pass.
///
/// Sorted front-to-back by the z-distance in front of the camera.
///
/// Used to render all 3D meshes with materials that have no transparency.
pub struct Opaque3dDeferred {
    pub entity: Entity,
    pub pipeline_id: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub material_bind_group_id: Option<BindGroupId>,
    pub mesh_asset_id: Option<AssetId<Mesh>>,
    pub distance: f32,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}

impl PhaseItem for Opaque3dDeferred {
    // NOTE: Values increase towards the camera. Front-to-back ordering for opaque means we need a descending sort.
    type SortKey = Reverse<FloatOrd>;

    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        Reverse(FloatOrd(self.distance))
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn sort(items: &mut [Self]) {
        // Key negated to match reversed SortKey ordering
        radsort::sort_by_key(items, |item| {
            (
                item.pipeline_id.id(),
                item.material_bind_group_id.map_or(0, |bgid| bgid.0.get()),
                item.mesh_asset_id().unwrap_or(0),
                -item.distance,
            )
        });
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn dynamic_offset(&self) -> Option<NonMaxU32> {
        self.dynamic_offset
    }

    #[inline]
    fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
        &mut self.dynamic_offset
    }

    #[inline]
    fn material_bind_group_id(&self) -> Option<BindGroupId> {
        self.material_bind_group_id
    }

    #[inline]
    fn mesh_asset_id(&self) -> Option<u64> {
        self.mesh_asset_id.map(|maid| match maid {
            AssetId::Index { index, .. } => index.generation as u64 | ((index.index as u64) << 32),
            AssetId::Uuid { uuid } => uuid.as_u64_pair().1,
        })
    }
}

impl CachedRenderPipelinePhaseItem for Opaque3dDeferred {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline_id
    }
}

/// Alpha mask phase of the 3D Deferred pass.
///
/// Sorted front-to-back by the z-distance in front of the camera.
///
/// Used to render all meshes with a material with an alpha mask.
pub struct AlphaMask3dDeferred {
    pub entity: Entity,
    pub pipeline_id: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub material_bind_group_id: Option<BindGroupId>,
    pub mesh_asset_id: Option<AssetId<Mesh>>,
    pub distance: f32,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}

impl PhaseItem for AlphaMask3dDeferred {
    // NOTE: Values increase towards the camera. Front-to-back ordering for opaque means we need a descending sort.
    type SortKey = Reverse<FloatOrd>;

    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        Reverse(FloatOrd(self.distance))
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn sort(items: &mut [Self]) {
        // Key negated to match reversed SortKey ordering
        radsort::sort_by_key(items, |item| {
            (
                item.pipeline_id.id(),
                item.material_bind_group_id.map_or(0, |bgid| bgid.0.get()),
                item.mesh_asset_id().unwrap_or(0),
                -item.distance,
            )
        });
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn dynamic_offset(&self) -> Option<NonMaxU32> {
        self.dynamic_offset
    }

    #[inline]
    fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
        &mut self.dynamic_offset
    }

    #[inline]
    fn material_bind_group_id(&self) -> Option<BindGroupId> {
        self.material_bind_group_id
    }

    #[inline]
    fn mesh_asset_id(&self) -> Option<u64> {
        self.mesh_asset_id.map(|maid| match maid {
            AssetId::Index { index, .. } => index.generation as u64 | ((index.index as u64) << 32),
            AssetId::Uuid { uuid } => uuid.as_u64_pair().1,
        })
    }
}

impl CachedRenderPipelinePhaseItem for AlphaMask3dDeferred {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline_id
    }
}
