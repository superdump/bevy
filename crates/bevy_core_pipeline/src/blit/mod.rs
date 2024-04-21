use bevy_app::{App, Plugin};
use bevy_asset::{io::embedded::InternalAssets, load_internal_asset, uuid::Uuid, Handle};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d},
        *,
    },
    renderer::RenderDevice,
    RenderApp,
};

use crate::fullscreen_vertex_shader::{fullscreen_shader_vertex_state, FULLSCREEN_SHADER_UUID};

pub const BLIT_SHADER_UUID: Uuid = Uuid::from_u128(2312396983770133547);

/// Adds support for specialized "blit pipelines", which can be used to write one texture to another.
pub struct BlitPlugin;

impl Plugin for BlitPlugin {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.allow_ambiguous_resource::<SpecializedRenderPipelines<BlitPipeline>>();
        }
    }

    fn finish(&self, app: &mut App) {
        load_internal_asset!(app, BLIT_SHADER_UUID, "blit.wgsl", Shader::from_wgsl);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<BlitPipeline>()
            .init_resource::<SpecializedRenderPipelines<BlitPipeline>>();
    }
}

#[derive(Resource)]
pub struct BlitPipeline {
    pub texture_bind_group: BindGroupLayout,
    pub sampler: Sampler,
    pub fullscreen_vertex_shader_handle: Handle<Shader>,
    pub blit_shader_handle: Handle<Shader>,
}

impl FromWorld for BlitPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();

        let texture_bind_group = render_device.create_bind_group_layout(
            "blit_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    sampler(SamplerBindingType::NonFiltering),
                ),
            ),
        );

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let internal_assets = render_world.resource::<InternalAssets<Shader>>();
        let fullscreen_vertex_shader_handle = internal_assets
            .get(&FULLSCREEN_SHADER_UUID)
            .expect("FULLSCREEN_SHADER_UUID is not present in InternalAssets")
            .clone_weak();
        let blit_shader_handle = internal_assets
            .get(&BLIT_SHADER_UUID)
            .expect("BLIT_SHADER_UUID is not present in InternalAssets")
            .clone_weak();

        BlitPipeline {
            texture_bind_group,
            sampler,
            fullscreen_vertex_shader_handle,
            blit_shader_handle,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct BlitPipelineKey {
    pub texture_format: TextureFormat,
    pub blend_state: Option<BlendState>,
    pub samples: u32,
}

impl SpecializedRenderPipeline for BlitPipeline {
    type Key = BlitPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("blit pipeline".into()),
            layout: vec![self.texture_bind_group.clone()],
            vertex: fullscreen_shader_vertex_state(
                self.fullscreen_vertex_shader_handle.clone_weak(),
            ),
            fragment: Some(FragmentState {
                shader: self.blit_shader_handle.clone_weak(),
                shader_defs: vec![],
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState {
                    format: key.texture_format,
                    blend: key.blend_state,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.samples,
                ..Default::default()
            },
            push_constant_ranges: Vec::new(),
        }
    }
}
