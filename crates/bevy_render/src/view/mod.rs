pub mod visibility;
pub mod window;

use bevy_asset::Handle;
pub use visibility::*;
use wgpu::{
    Color, Extent3d, Operations, RenderPassColorAttachment, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages,
};
pub use window::*;

use crate::{
    camera::{Camera, ExtractedCamera},
    extract_component::ExtractComponent,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    prelude::Image,
    rangefinder::ViewRangefinder3d,
    render_asset::RenderAssets,
    render_resource::{DynamicUniformBuffer, ShaderType, Texture, TextureView},
    renderer::{RenderDevice, RenderQueue},
    texture::{BevyDefault, TextureCache},
    RenderApp, RenderStage,
};
use bevy_app::{App, Plugin};
use bevy_ecs::{prelude::*, query::QueryItem};
use bevy_math::{Mat4, Vec3};
use bevy_reflect::Reflect;
use bevy_transform::components::GlobalTransform;
use bevy_utils::HashMap;

pub struct ViewPlugin;

impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Msaa>()
            .init_resource::<Msaa>()
            // NOTE: windows.is_changed() handles cases where a window was resized
            .add_plugin(ExtractResourcePlugin::<Msaa>::default())
            .add_plugin(VisibilityPlugin);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ViewUniforms>()
                .add_system_to_stage(RenderStage::Prepare, prepare_view_uniforms)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_view_targets.after(WindowSystem::Prepare),
                );
        }
    }
}

/// Configuration resource for [Multi-Sample Anti-Aliasing](https://en.wikipedia.org/wiki/Multisample_anti-aliasing).
///
/// # Example
/// ```
/// # use bevy_app::prelude::App;
/// # use bevy_render::prelude::Msaa;
/// App::new()
///     .insert_resource(Msaa { samples: 4 })
///     .run();
/// ```
#[derive(Clone, ExtractResource, Reflect)]
#[reflect(Resource)]
pub struct Msaa {
    /// The number of samples to run for Multi-Sample Anti-Aliasing. Higher numbers result in
    /// smoother edges.
    /// Defaults to 4.
    ///
    /// Note that WGPU currently only supports 1 or 4 samples.
    /// Ultimately we plan on supporting whatever is natively supported on a given device.
    /// Check out this issue for more info: <https://github.com/gfx-rs/wgpu/issues/1832>
    pub samples: u32,
}

impl Default for Msaa {
    fn default() -> Self {
        Self { samples: 4 }
    }
}

#[derive(Default, Clone, Component, Debug, Reflect)]
#[reflect(Component)]
pub struct EnvironmentMap {
    pub handle: Handle<Image>,
}

impl ExtractComponent for EnvironmentMap {
    type Query = &'static Self;
    type Filter = With<Camera>;

    fn extract_component(item: QueryItem<Self::Query>) -> Self {
        item.clone()
    }
}

#[derive(Component)]
pub struct ExtractedView {
    pub projection: Mat4,
    pub transform: GlobalTransform,
    pub width: u32,
    pub height: u32,
}

impl ExtractedView {
    /// Creates a 3D rangefinder for a view
    pub fn rangefinder3d(&self) -> ViewRangefinder3d {
        ViewRangefinder3d::from_view_matrix(&self.transform.compute_matrix())
    }
}

// NOTE: These must match the bit flags in bevy_pbr/src/render/mesh_view_types.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    pub struct ViewFlags: u32 {
        const ENVIRONMENT_MAP = (1 << 0);
        const NONE            = 0;
        const UNINITIALIZED   = 0xFFFF;
    }
}

#[derive(Clone, ShaderType)]
pub struct ViewUniform {
    view_proj: Mat4,
    inverse_view_proj: Mat4,
    view: Mat4,
    inverse_view: Mat4,
    projection: Mat4,
    inverse_projection: Mat4,
    world_position: Vec3,
    width: f32,
    height: f32,
    flags: u32,
}

#[derive(Default)]
pub struct ViewUniforms {
    pub uniforms: DynamicUniformBuffer<ViewUniform>,
}

#[derive(Component)]
pub struct ViewUniformOffset {
    pub offset: u32,
}

#[derive(Component)]
pub struct ViewTarget {
    pub view: TextureView,
    pub sampled_target: Option<TextureView>,
}

impl ViewTarget {
    pub fn get_color_attachment(&self, ops: Operations<Color>) -> RenderPassColorAttachment {
        RenderPassColorAttachment {
            view: self.sampled_target.as_ref().unwrap_or(&self.view),
            resolve_target: if self.sampled_target.is_some() {
                Some(&self.view)
            } else {
                None
            },
            ops,
        }
    }
}

#[derive(Component)]
pub struct ViewDepthTexture {
    pub texture: Texture,
    pub view: TextureView,
}

fn prepare_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<ViewUniforms>,
    views: Query<(Entity, &ExtractedView, Option<&EnvironmentMap>)>,
) {
    view_uniforms.uniforms.clear();
    for (entity, camera, maybe_environment_map) in &views {
        let projection = camera.projection;
        let inverse_projection = projection.inverse();
        let view = camera.transform.compute_matrix();
        let inverse_view = view.inverse();
        let mut flags = ViewFlags::empty();
        if maybe_environment_map.is_some() {
            flags |= ViewFlags::ENVIRONMENT_MAP;
        }
        let view_uniforms = ViewUniformOffset {
            offset: view_uniforms.uniforms.push(ViewUniform {
                view_proj: projection * inverse_view,
                inverse_view_proj: view * inverse_projection,
                view,
                inverse_view,
                projection,
                inverse_projection,
                world_position: camera.transform.translation(),
                width: camera.width as f32,
                height: camera.height as f32,
                flags: flags.bits(),
            }),
        };

        commands.entity(entity).insert(view_uniforms);
    }

    view_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

#[allow(clippy::too_many_arguments)]
fn prepare_view_targets(
    mut commands: Commands,
    windows: Res<ExtractedWindows>,
    images: Res<RenderAssets<Image>>,
    msaa: Res<Msaa>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera)>,
) {
    let mut sampled_textures = HashMap::default();
    for (entity, camera) in &cameras {
        if let Some(target_size) = camera.physical_target_size {
            if let Some(texture_view) = camera.target.get_texture_view(&windows, &images) {
                let sampled_target = if msaa.samples > 1 {
                    let sampled_texture = sampled_textures
                        .entry(camera.target.clone())
                        .or_insert_with(|| {
                            texture_cache.get(
                                &render_device,
                                TextureDescriptor {
                                    label: Some("sampled_color_attachment_texture"),
                                    size: Extent3d {
                                        width: target_size.x,
                                        height: target_size.y,
                                        depth_or_array_layers: 1,
                                    },
                                    mip_level_count: 1,
                                    sample_count: msaa.samples,
                                    dimension: TextureDimension::D2,
                                    format: TextureFormat::bevy_default(),
                                    usage: TextureUsages::RENDER_ATTACHMENT,
                                },
                            )
                        });
                    Some(sampled_texture.default_view.clone())
                } else {
                    None
                };
                commands.entity(entity).insert(ViewTarget {
                    view: texture_view.clone(),
                    sampled_target,
                });
            }
        }
    }
}
