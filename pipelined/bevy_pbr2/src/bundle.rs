use crate::{Cascades, CascadesConfig, DirectionalLight, PointLight, StandardMaterial};
use bevy_asset::Handle;
use bevy_ecs::bundle::Bundle;
use bevy_render2::{
    mesh::Mesh,
    primitives::{CascadeFrusta, CubemapFrusta},
    view::{ComputedVisibility, Visibility, VisibleEntities},
};
use bevy_transform::components::{GlobalTransform, Transform};

#[derive(Bundle, Clone, Default)]
pub struct PbrBundle {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    /// User indication of whether an entity is visible
    pub visibility: Visibility,
    /// Algorithmically-computed indication of whether an entity is visible and should be extracted for rendering
    pub computed_visibility: ComputedVisibility,
}

#[derive(Clone, Debug, Default)]
pub struct CubemapVisibleEntities {
    data: [VisibleEntities; 6],
}

impl CubemapVisibleEntities {
    pub fn get(&self, i: usize) -> &VisibleEntities {
        &self.data[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut VisibleEntities {
        &mut self.data[i]
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &VisibleEntities> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut VisibleEntities> {
        self.data.iter_mut()
    }
}

#[derive(Clone, Debug)]
pub struct CascadesVisibleEntities {
    data: Vec<VisibleEntities>,
}

impl Default for CascadesVisibleEntities {
    fn default() -> Self {
        Self::with_capacity(CascadesConfig::default().len())
    }
}

impl CascadesVisibleEntities {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![Default::default(); capacity],
        }
    }

    pub fn clear(&mut self) {
        for visible_entities in self.data.iter_mut() {
            visible_entities.entities.clear();
        }
    }

    pub fn get(&self, i: usize) -> &VisibleEntities {
        &self.data[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut VisibleEntities {
        &mut self.data[i]
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &VisibleEntities> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut VisibleEntities> {
        self.data.iter_mut()
    }
}

/// A component bundle for "point light" entities
#[derive(Debug, Bundle, Default)]
pub struct PointLightBundle {
    pub point_light: PointLight,
    pub cubemap_visible_entities: CubemapVisibleEntities,
    pub cubemap_frusta: CubemapFrusta,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
}

/// A component bundle for "directional light" entities
#[derive(Debug, Bundle, Default)]
pub struct DirectionalLightBundle {
    pub directional_light: DirectionalLight,
    pub cascades_config: CascadesConfig,
    pub cascades: Cascades,
    pub cascade_frusta: CascadeFrusta,
    pub cascades_visible_entities: CascadesVisibleEntities,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
}
