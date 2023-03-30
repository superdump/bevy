mod node;

pub use node::*;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Assets, Handle, HandleUntyped};
use bevy_ecs::{
    prelude::{Component, Entity},
    query::With,
    schedule::IntoSystemConfigs,
    system::{Commands, Query, Res, ResMut, Resource},
};
use bevy_reflect::TypeUuid;
use bevy_render::{
    prelude::{shape::Cube, Mesh},
    render_asset::RenderAssets,
    render_resource::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, BlendState, BufferBindingType,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
        DepthStencilState, FragmentState, MultisampleState, PipelineCache, PrimitiveState,
        RenderPipelineDescriptor, SamplerBindingType, Shader, ShaderStages, ShaderType,
        SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState, StencilState,
        TextureFormat, TextureSampleType, TextureViewDimension, VertexBufferLayout, VertexState,
    },
    renderer::RenderDevice,
    texture::Image,
    view::{Msaa, ViewTarget, ViewUniform, ViewUniforms},
    Render, RenderApp, RenderSet,
};

const SKYBOX_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 55594763423201);

pub struct SkyboxPlugin;

impl Plugin for SkyboxPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SKYBOX_SHADER_HANDLE,
            "skybox/skybox.wgsl",
            Shader::from_wgsl
        );

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        let mesh = Mesh::from(Cube::new(1.0));
        let vertex_buffer_layout = mesh.get_mesh_vertex_buffer_layout().layout().clone();

        render_app
            .insert_resource(SkyboxMesh {
                handle: app.world.resource_mut::<Assets<Mesh>>().add(mesh),
            })
            .insert_resource(SkyboxPipeline::new(
                vertex_buffer_layout,
                app.world.resource::<RenderDevice>(),
            ))
            .init_resource::<SpecializedRenderPipelines<SkyboxPipeline>>()
            .add_systems(
                Render,
                (
                    prepare_skybox_pipelines.in_set(RenderSet::Prepare),
                    queue_skybox_bind_groups.in_set(RenderSet::Queue),
                ),
            );
    }
}

#[derive(Component)]
pub struct Skybox(pub Handle<Image>);

// ----------------------------------------------------------------------------

#[derive(Resource)]
struct SkyboxMesh {
    handle: Handle<Mesh>,
}

#[derive(Resource)]
struct SkyboxPipeline {
    bind_group_layout: BindGroupLayout,
    vertex_buffer_layout: VertexBufferLayout,
}

impl SkyboxPipeline {
    fn new(vertex_buffer_layout: VertexBufferLayout, render_device: &RenderDevice) -> Self {
        let bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("skybox_bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: Some(ViewUniform::min_size()),
                        },
                        count: None,
                    },
                ],
            });

        Self {
            bind_group_layout,
            vertex_buffer_layout,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct SkyboxPipelineKey {
    texture_format: TextureFormat,
    samples: u32,
}

impl SpecializedRenderPipeline for SkyboxPipeline {
    type Key = SkyboxPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("skybox_pipeline".into()),
            layout: vec![self.bind_group_layout],
            push_constant_ranges: Vec::new(),
            vertex: VertexState {
                shader: SKYBOX_SHADER_HANDLE.typed(),
                shader_defs: Vec::new(),
                entry_point: "skybox_vertex".into(),
                buffers: vec![self.vertex_buffer_layout.clone()],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: SKYBOX_SHADER_HANDLE.typed(),
                shader_defs: Vec::new(),
                entry_point: "skybox_fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: key.texture_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
        }
    }
}

#[derive(Component)]
struct SkyboxPipelineId(CachedRenderPipelineId);

fn prepare_skybox_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SkyboxPipeline>>,
    pipeline: Res<SkyboxPipeline>,
    msaa: Res<Msaa>,
    views: Query<(Entity, &ViewTarget), With<Skybox>>,
) {
    for (entity, view_target) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SkyboxPipelineKey {
                texture_format: view_target.main_texture_format(),
                samples: msaa.samples(),
            },
        );

        commands
            .entity(entity)
            .insert(SkyboxPipelineId(pipeline_id));
    }
}

#[derive(Component)]
struct SkyboxBindGroup(BindGroup);

fn queue_skybox_bind_groups(
    mut commands: Commands,
    pipeline: Res<SkyboxPipeline>,
    view_uniforms: Res<ViewUniforms>,
    images: Res<RenderAssets<Image>>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &Skybox)>,
) {
    for (entity, skybox) in &views {
        if let (Some(skybox), Some(view_uniforms)) =
            (images.get(&skybox.0), view_uniforms.uniforms.binding())
        {
            let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                label: Some("skybox_bind_group"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&skybox.texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&skybox.sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: view_uniforms,
                    },
                ],
            });

            commands.entity(entity).insert(SkyboxBindGroup(bind_group));
        }
    }
}
