//! Run a prepass before the main pass to generate depth, normals, and/or motion vectors textures, sometimes called a thin g-buffer.
//! These textures are useful for various screen-space effects and reducing overdraw in the main pass.
//!
//! The prepass only runs for opaque meshes or meshes with an alpha mask. Transparent meshes are ignored.
//!
//! To enable the prepass, you need to add a prepass component to a [`crate::prelude::Camera3d`].
//!
//! [`DepthPrepass`]
//! [`NormalPrepass`]
//! [`MotionVectorPrepass`]
//!
//! The textures are automatically added to the default mesh view bindings. You can also get the raw textures
//! by querying the [`ViewPrepassTextures`] component on any camera with a prepass component.
//!
//! The depth prepass will always run and generate the depth buffer as a side effect, but it won't copy it
//! to a separate texture unless the [`DepthPrepass`] is activated. This means that if any prepass component is present
//! it will always create a depth buffer that will be used by the main pass.
//!
//! When using the default mesh view bindings you should be able to use `prepass_depth()`,
//! `prepass_normal()`, and `prepass_motion_vector()` to load the related textures.
//! These functions are defined in `bevy_pbr::prepass_utils`. See the `shader_prepass` example that shows how to use them.
//!
//! The prepass runs for each `Material`. You can control if the prepass should run per-material by setting the `prepass_enabled`
//! flag on the `MaterialPlugin`.
//!
//! Currently only works for 3D.

pub mod node;

use std::{cmp::Reverse, ops::Range};

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use bevy_render::{
    mesh::Mesh,
    render_phase::{CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem},
    render_resource::{BindGroupId, CachedRenderPipelineId, Extent3d, TextureFormat},
    texture::CachedTexture,
};
use bevy_utils::{nonmax::NonMaxU32, FloatOrd};

pub const NORMAL_PREPASS_FORMAT: TextureFormat = TextureFormat::Rgb10a2Unorm;
pub const MOTION_VECTOR_PREPASS_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// If added to a [`crate::prelude::Camera3d`] then depth values will be copied to a separate texture available to the main pass.
#[derive(Component, Default, Reflect)]
pub struct DepthPrepass;

/// If added to a [`crate::prelude::Camera3d`] then vertex world normals will be copied to a separate texture available to the main pass.
/// Normals will have normal map textures already applied.
#[derive(Component, Default, Reflect)]
pub struct NormalPrepass;

/// If added to a [`crate::prelude::Camera3d`] then screen space motion vectors will be copied to a separate texture available to the main pass.
#[derive(Component, Default, Reflect)]
pub struct MotionVectorPrepass;

/// If added to a [`crate::prelude::Camera3d`] then deferred materials will be rendered to the deferred gbuffer texture and will be available to subsequent passes.
/// Note the default deferred lighting plugin also requires `DepthPrepass` to work correctly.
#[derive(Component, Default, Reflect)]
pub struct DeferredPrepass;

/// Textures that are written to by the prepass.
///
/// This component will only be present if any of the relevant prepass components are also present.
#[derive(Component)]
pub struct ViewPrepassTextures {
    /// The depth texture generated by the prepass.
    /// Exists only if [`DepthPrepass`] is added to the [`ViewTarget`](bevy_render::view::ViewTarget)
    pub depth: Option<CachedTexture>,
    /// The normals texture generated by the prepass.
    /// Exists only if [`NormalPrepass`] is added to the [`ViewTarget`](bevy_render::view::ViewTarget)
    pub normal: Option<CachedTexture>,
    /// The motion vectors texture generated by the prepass.
    /// Exists only if [`MotionVectorPrepass`] is added to the `ViewTarget`
    pub motion_vectors: Option<CachedTexture>,
    /// The deferred gbuffer generated by the deferred pass.
    /// Exists only if [`DeferredPrepass`] is added to the `ViewTarget`
    pub deferred: Option<CachedTexture>,
    /// A texture that specifies the deferred lighting pass id for a material.
    /// Exists only if [`DeferredPrepass`] is added to the `ViewTarget`
    pub deferred_lighting_pass_id: Option<CachedTexture>,
    /// The size of the textures.
    pub size: Extent3d,
}

/// Opaque phase of the 3D prepass.
///
/// Sorted front-to-back by the z-distance in front of the camera.
///
/// Used to render all 3D meshes with materials that have no transparency.
pub struct Opaque3dPrepass {
    pub entity: Entity,
    pub pipeline_id: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub material_bind_group_id: Option<BindGroupId>,
    pub mesh_asset_id: Option<AssetId<Mesh>>,
    pub distance: f32,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}

impl PhaseItem for Opaque3dPrepass {
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

impl CachedRenderPipelinePhaseItem for Opaque3dPrepass {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline_id
    }
}

/// Alpha mask phase of the 3D prepass.
///
/// Sorted front-to-back by the z-distance in front of the camera.
///
/// Used to render all meshes with a material with an alpha mask.
pub struct AlphaMask3dPrepass {
    pub entity: Entity,
    pub pipeline_id: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub material_bind_group_id: Option<BindGroupId>,
    pub mesh_asset_id: Option<AssetId<Mesh>>,
    pub distance: f32,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}

impl PhaseItem for AlphaMask3dPrepass {
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

impl CachedRenderPipelinePhaseItem for AlphaMask3dPrepass {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline_id
    }
}
