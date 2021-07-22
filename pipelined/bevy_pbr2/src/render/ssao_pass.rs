use crate::{DepthNormalShaders, ExtractedMeshes, PbrShaders, ViewDepthNormal};
use bevy_asset::{AssetServer, Handle};
use bevy_ecs::{prelude::*, system::SystemState};
use bevy_math::Vec3;
use bevy_render2::{
    camera::{ActiveCameras, CameraPlugin},
    color::Color,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_phase::{Draw, DrawFunctions, Drawable, RenderPhase, TrackedRenderPass},
    render_resource::*,
    renderer::{RenderContext, RenderDevice},
    shader::Shader,
    texture::*,
    view::{ExtractedView, ViewMeta, ViewUniformOffset},
};
use crevice::std140::AsStd140;
use std::num::NonZeroU32;

#[derive(Debug, Clone)]
pub struct SsaoConfig {
    kernel: [Vec3; 32],
    kernel_size: u32,
    radius: f32,
    bias: f32,
    blue_noise_image: Option<Handle<Image>>,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            kernel: [
                // precalculated hemisphere kernel (low discrepancy noiser)
                Vec3::new(-0.668154, -0.084296, 0.219458),
                Vec3::new(-0.092521, 0.141327, 0.505343),
                Vec3::new(-0.041960, 0.700333, 0.365754),
                Vec3::new(0.722389, -0.015338, 0.084357),
                Vec3::new(-0.815016, 0.253065, 0.465702),
                Vec3::new(0.018993, -0.397084, 0.136878),
                Vec3::new(0.617953, -0.234334, 0.513754),
                Vec3::new(-0.281008, -0.697906, 0.240010),
                Vec3::new(0.303332, -0.443484, 0.588136),
                Vec3::new(-0.477513, 0.559972, 0.310942),
                Vec3::new(0.307240, 0.076276, 0.324207),
                Vec3::new(-0.404343, -0.615461, 0.098425),
                Vec3::new(0.152483, -0.326314, 0.399277),
                Vec3::new(0.435708, 0.630501, 0.169620),
                Vec3::new(0.878907, 0.179609, 0.266964),
                Vec3::new(-0.049752, -0.232228, 0.264012),
                Vec3::new(0.537254, -0.047783, 0.693834),
                Vec3::new(0.001000, 0.177300, 0.096643),
                Vec3::new(0.626400, 0.524401, 0.492467),
                Vec3::new(-0.708714, -0.223893, 0.182458),
                Vec3::new(-0.106760, 0.020965, 0.451976),
                Vec3::new(-0.285181, -0.388014, 0.241756),
                Vec3::new(0.241154, -0.174978, 0.574671),
                Vec3::new(-0.405747, 0.080275, 0.055816),
                Vec3::new(0.079375, 0.289697, 0.348373),
                Vec3::new(0.298047, -0.309351, 0.114787),
                Vec3::new(-0.616434, -0.117369, 0.475924),
                Vec3::new(-0.035249, 0.134591, 0.840251),
                Vec3::new(0.175849, 0.971033, 0.211778),
                Vec3::new(0.024805, 0.348056, 0.240006),
                Vec3::new(-0.267123, 0.204885, 0.688595),
                Vec3::new(-0.077639, -0.753205, 0.070938),
            ],
            kernel_size: 8,
            radius: 0.2,
            bias: 0.025,
            blue_noise_image: None,
        }
    }
}

pub fn load_blue_noise(mut ssao_config: ResMut<SsaoConfig>, asset_server: Res<AssetServer>) {
    ssao_config.blue_noise_image = Some(asset_server.load("textures/blue_noise.png"));
}

#[derive(Clone, AsStd140)]
pub struct SsaoConfigUniform {
    kernel: [Vec3; 32],
    kernel_size: u32,
    radius: f32,
    bias: f32,
}

pub struct SsaoShaders {
    pub pipeline: RenderPipeline,
    pub view_layout: BindGroupLayout,
    pub blue_noise_sampler: Sampler,
    pub ssao_layout: BindGroupLayout,
    pub ssao_sampler: Sampler,
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for SsaoShaders {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let fullscreen_shader = Shader::from_wgsl(include_str!("fullscreen.wgsl"));
        let fullscreen_shader_module = render_device.create_shader_module(&fullscreen_shader);
        let ssao_shader = Shader::from_wgsl(include_str!("ssao.wgsl"));
        let ssao_shader_module = render_device.create_shader_module(&ssao_shader);

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(336),
                    },
                    count: None,
                },
                // Depth Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Depth Texture Sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                // View Normal Texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // View Normal Texture Sampler
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let ssao_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // SsaoConfig
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(528),
                    },
                    count: None,
                },
                // Blue Noise Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Blue Noise Texture Sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            push_constant_ranges: &[],
            bind_group_layouts: &[&view_layout, &ssao_layout],
        });

        let pipeline = render_device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            vertex: VertexState {
                buffers: &[],
                module: &fullscreen_shader_module,
                entry_point: "vertex",
            },
            fragment: Some(FragmentState {
                module: &ssao_shader_module,
                entry_point: "fragment",
                targets: &[ColorTargetState {
                    format: TextureFormat::R8Unorm,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::Src,
                            dst_factor: BlendFactor::OneMinusSrc,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrite::ALL,
                }],
            }),
            depth_stencil: None,
            layout: Some(&pipeline_layout),
            multisample: MultisampleState::default(),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                clamp_depth: false,
                conservative: false,
            },
        });

        SsaoShaders {
            pipeline,
            view_layout,
            blue_noise_sampler: render_device.create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::Repeat,
                address_mode_v: AddressMode::Repeat,
                address_mode_w: AddressMode::Repeat,
                ..Default::default()
            }),
            ssao_layout,
            ssao_sampler: render_device.create_sampler(&SamplerDescriptor::default()),
        }
    }
}

type ExtractedSsaoConfig = SsaoConfig;

pub fn extract_ssao(
    mut commands: Commands,
    active_cameras: Res<ActiveCameras>,
    ssao_config: Res<SsaoConfig>,
) {
    if let Some(camera_3d) = active_cameras.get(CameraPlugin::CAMERA_3D) {
        if let Some(entity) = camera_3d.entity {
            commands
                .get_or_spawn(entity)
                .insert(RenderPhase::<SsaoPhase>::default());
        }
    }
    commands.insert_resource::<ExtractedSsaoConfig>(ssao_config.clone());
}

#[derive(Default)]
pub struct SsaoMeta {
    pub uniform: UniformVec<SsaoConfigUniform>,
}

pub struct ViewSsao {
    pub view_ssao_texture: Texture,
    pub view_ssao_texture_view: TextureView,
}

pub fn prepare_ssao(
    mut commands: Commands,
    extracted_ssao_config: Res<ExtractedSsaoConfig>,
    mut ssao_meta: ResMut<SsaoMeta>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedView), With<RenderPhase<SsaoPhase>>>,
) {
    ssao_meta.uniform.reserve_and_clear(1, &render_device);

    ssao_meta.uniform.push(SsaoConfigUniform {
        kernel: extracted_ssao_config.kernel,
        kernel_size: extracted_ssao_config.kernel_size,
        radius: extracted_ssao_config.radius,
        bias: extracted_ssao_config.bias,
    });

    // set up ssao for each view
    for (entity, view) in views.iter() {
        let view_ssao_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: view.width,
                    height: view.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R8Unorm,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            },
        );
        let view_ssao_texture_view =
            view_ssao_texture
                .texture
                .create_view(&TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: Some(TextureViewDimension::D2),
                    aspect: TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: NonZeroU32::new(1),
                });
        commands.entity(entity).insert(ViewSsao {
            view_ssao_texture: view_ssao_texture.texture,
            view_ssao_texture_view,
        });
    }

    ssao_meta.uniform.write_to_staging_buffer(&render_device);
}

fn image_handle_to_view_sampler<'a>(
    pbr_shaders: &'a PbrShaders,
    gpu_images: &'a RenderAssets<Image>,
    image_option: &Option<Handle<Image>>,
) -> (&'a TextureView, &'a Sampler) {
    image_option.as_ref().map_or(
        (
            &pbr_shaders.dummy_white_gpu_image.texture_view,
            &pbr_shaders.dummy_white_gpu_image.sampler,
        ),
        |image_handle| {
            let gpu_image = gpu_images
                .get(image_handle)
                .expect("only materials with valid textures should be drawn");
            (&gpu_image.texture_view, &gpu_image.sampler)
        },
    )
}

pub struct SsaoViewBindGroup {
    view_bind_group: BindGroup,
}

pub struct SsaoConfigBindGroup {
    ssao_config_bind_group: BindGroup,
}

pub fn queue_meshes(
    mut commands: Commands,
    draw_functions: Res<DrawFunctions>,
    render_device: Res<RenderDevice>,
    ssao_shaders: Res<SsaoShaders>,
    depth_normal_shaders: Res<DepthNormalShaders>,
    pbr_shaders: Res<PbrShaders>,
    view_meta: Res<ViewMeta>,
    ssao_meta: Res<SsaoMeta>,
    extracted_ssao_config: Res<ExtractedSsaoConfig>,
    gpu_images: Res<RenderAssets<Image>>,
    extracted_meshes: Res<ExtractedMeshes>,
    mut views: Query<(Entity, &ViewDepthNormal, &mut RenderPhase<SsaoPhase>)>,
) {
    if extracted_meshes.meshes.is_empty()
        || extracted_ssao_config.blue_noise_image.is_none()
        || gpu_images
            .get(extracted_ssao_config.blue_noise_image.as_ref().unwrap())
            .is_none()
        || view_meta.uniforms.len() < 1
    {
        return;
    }

    let (blue_noise_texture_view, _default_sampler) = image_handle_to_view_sampler(
        &pbr_shaders,
        &gpu_images,
        &extracted_ssao_config.blue_noise_image,
    );

    let ssao_config_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: ssao_meta.uniform.binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(blue_noise_texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&ssao_shaders.blue_noise_sampler),
            },
        ],
        label: None,
        layout: &ssao_shaders.ssao_layout,
    });

    commands.insert_resource(SsaoConfigBindGroup {
        ssao_config_bind_group,
    });

    for (i, (entity, view_depth_normal, mut ssao_phase)) in views.iter_mut().enumerate() {
        // TODO: cache this?
        let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: view_meta.uniforms.binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &view_depth_normal.view_depth_texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&depth_normal_shaders.depth_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(
                        &view_depth_normal.view_normal_texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&depth_normal_shaders.normal_sampler),
                },
            ],
            label: None,
            layout: &ssao_shaders.view_layout,
        });

        commands
            .entity(entity)
            .insert(SsaoViewBindGroup { view_bind_group });

        let draw_ssao = draw_functions.read().get_id::<DrawSsao>().unwrap();
        ssao_phase.add(Drawable {
            draw_function: draw_ssao,
            draw_key: i,
            sort_key: 0,
        });
    }
}

pub struct SsaoPhase;

pub struct SsaoPassNode {
    main_view_query: QueryState<(&'static ViewSsao, &'static RenderPhase<SsaoPhase>)>,
}

impl SsaoPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
        }
    }
}

impl Node for SsaoPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(SsaoPassNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.main_view_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let ssao_meta = world.get_resource::<SsaoMeta>().unwrap();
        ssao_meta
            .uniform
            .write_to_uniform_buffer(&mut render_context.command_encoder);

        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        if let Some((view_ssao, ssao_phase)) =
            self.main_view_query.get_manual(world, view_entity).ok()
        {
            let pass_descriptor = RenderPassDescriptor {
                label: Some("ssao"),
                color_attachments: &[RenderPassColorAttachment {
                    view: &view_ssao.view_ssao_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK.into()),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            };

            let draw_functions = world.get_resource::<DrawFunctions>().unwrap();

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut draw_functions = draw_functions.write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            for drawable in ssao_phase.drawn_things.iter() {
                let draw_function = draw_functions.get_mut(drawable.draw_function).unwrap();
                draw_function.draw(
                    world,
                    &mut tracked_pass,
                    view_entity,
                    drawable.draw_key,
                    drawable.sort_key,
                );
            }
        }

        Ok(())
    }
}

type DrawSsaoParams<'s, 'w> = (
    Res<'w, SsaoShaders>,
    Res<'w, SsaoConfigBindGroup>,
    Query<'w, 's, (&'w ViewUniformOffset, &'w SsaoViewBindGroup)>,
);
pub struct DrawSsao {
    params: SystemState<DrawSsaoParams<'static, 'static>>,
}

impl DrawSsao {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

impl Draw for DrawSsao {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        _draw_key: usize,
        _sort_key: usize,
    ) {
        let (ssao_shaders, ssao_config_bind_group, views) = self.params.get(world);
        let (view_uniform_offset, ssao_view_bind_group) = views.get(view).unwrap();
        pass.set_render_pipeline(&ssao_shaders.into_inner().pipeline);
        pass.set_bind_group(
            0,
            &ssao_view_bind_group.view_bind_group,
            &[view_uniform_offset.offset],
        );

        pass.set_bind_group(
            1,
            &ssao_config_bind_group.into_inner().ssao_config_bind_group,
            &[],
        );

        pass.draw(0..3, 0..1);
    }
}
