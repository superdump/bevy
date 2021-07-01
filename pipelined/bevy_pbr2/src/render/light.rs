use crate::{AmbientLight, DirectionalLight, ExtractedMeshes, MeshMeta, OmniLight, PbrShaders};
use bevy_ecs::{prelude::*, system::SystemState};
use bevy_math::{Mat4, UVec4, Vec2, Vec3, Vec4};
use bevy_render2::{
    camera::CameraProjection,
    color::Color,
    core_pipeline::Transparent3dPhase,
    mesh::Mesh,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_phase::{Draw, DrawFunctions, RenderPhase, TrackedRenderPass},
    render_resource::*,
    renderer::{RenderContext, RenderDevice},
    texture::*,
    view::{ExtractedView, ViewUniform, ViewUniformOffset},
};
use bevy_transform::components::GlobalTransform;
use crevice::std140::AsStd140;
use std::num::NonZeroU32;

pub struct ExtractedAmbientLight {
    color: Color,
    brightness: f32,
}

pub struct ExtractedOmniLight {
    color: Color,
    intensity: f32,
    range: f32,
    radius: f32,
    transform: GlobalTransform,
    shadow_bias_min_max: Vec2,
}

pub struct ExtractedDirectionalLight {
    color: Color,
    illuminance: f32,
    direction: Vec3,
    projection: Mat4,
    shadow_bias_min_max: Vec2,
}

#[repr(C)]
#[derive(Copy, Clone, AsStd140, Default, Debug)]
pub struct GpuOmniLight {
    view_proj: Mat4,
    color: Vec4,
    position: Vec3,
    range: f32,
    radius: f32,
    shadow_bias_min: f32,
    shadow_bias_max: f32,
}

#[repr(C)]
#[derive(Copy, Clone, AsStd140, Default, Debug)]
pub struct GpuDirectionalLight {
    view_projection: Mat4,
    color: Vec4,
    dir_to_light: Vec3,
    shadow_bias_min: f32,
    shadow_bias_max: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, AsStd140)]
pub struct GpuLights {
    omni_lights: [GpuOmniLight; MAX_OMNI_LIGHTS],
    directional_lights: [GpuDirectionalLight; MAX_DIRECTIONAL_LIGHTS],
    ambient_color: Vec4,
    n_omni_lights: u32,
    n_directional_lights: u32,
}

// NOTE: this must be kept in sync with the same constants in pbr.frag
pub const MAX_OMNI_LIGHTS: usize = 10;
pub const MAX_DIRECTIONAL_LIGHTS: usize = 1;
pub const SHADOW_SIZE: Extent3d = Extent3d {
    width: 1024,
    height: 1024,
    depth_or_array_layers: (MAX_OMNI_LIGHTS + MAX_DIRECTIONAL_LIGHTS) as u32,
};
pub const SHADOW_FORMAT: TextureFormat = TextureFormat::Depth32Float;

pub struct ShadowShaders {
    pub pipeline: RenderPipeline,
    pub view_layout: BindGroupLayout,
    pub light_sampler: Sampler,
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for ShadowShaders {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let pbr_shaders = world.get_resource::<PbrShaders>().unwrap();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        // TODO: change this to ViewUniform::std140_padded_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(
                            ViewUniform::std140_padded_size_static() as u64
                        ),
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            push_constant_ranges: &[],
            bind_group_layouts: &[&view_layout, &pbr_shaders.mesh_layout],
        });

        let pipeline = render_device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            vertex: VertexState {
                buffers: &[VertexBufferLayout {
                    array_stride: 32,
                    step_mode: InputStepMode::Vertex,
                    attributes: &[
                        // Position (GOTCHA! Vertex_Position isn't first in the buffer due to how Mesh sorts attributes (alphabetically))
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 0,
                        },
                        // Normal
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 1,
                        },
                        // Uv
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                }],
                module: &pbr_shaders.shader_module,
                entry_point: "vertex",
            },
            fragment: None,
            depth_stencil: Some(DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
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

        ShadowShaders {
            pipeline,
            view_layout,
            light_sampler: render_device.create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                compare: Some(CompareFunction::LessEqual),
                ..Default::default()
            }),
        }
    }
}

// TODO: ultimately these could be filtered down to lights relevant to actual views
pub fn extract_lights(
    mut commands: Commands,
    ambient_light: Res<AmbientLight>,
    omni_lights: Query<(Entity, &OmniLight, &GlobalTransform)>,
    directional_lights: Query<(Entity, &DirectionalLight)>,
) {
    commands.insert_resource(ExtractedAmbientLight {
        color: ambient_light.color,
        brightness: ambient_light.brightness,
    });
    for (entity, omni_light, transform) in omni_lights.iter() {
        commands.get_or_spawn(entity).insert(ExtractedOmniLight {
            color: omni_light.color,
            intensity: omni_light.intensity,
            range: omni_light.range,
            radius: omni_light.radius,
            transform: transform.clone(),
            shadow_bias_min_max: omni_light.shadow_bias_min_max,
        });
    }
    for (entity, directional_light) in directional_lights.iter() {
        commands
            .get_or_spawn(entity)
            .insert(ExtractedDirectionalLight {
                color: directional_light.color,
                illuminance: directional_light.illuminance,
                direction: directional_light.get_direction(),
                projection: directional_light.shadow_projection.get_projection_matrix(),
                shadow_bias_min_max: directional_light.shadow_bias_min_max,
            });
    }
}

pub struct ViewLight {
    pub depth_texture: TextureView,
}

pub struct ViewLights {
    pub light_depth_texture: Texture,
    pub light_depth_texture_view: TextureView,
    pub lights: Vec<Entity>,
    pub gpu_light_binding_index: u32,
}

#[derive(Default)]
pub struct LightMeta {
    pub view_gpu_lights: DynamicUniformVec<GpuLights>,
    pub shadow_view_bind_group: Option<BindGroup>,
}

pub fn prepare_lights(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut light_meta: ResMut<LightMeta>,
    views: Query<Entity, With<RenderPhase<Transparent3dPhase>>>,
    ambient_light: Res<ExtractedAmbientLight>,
    omni_lights: Query<&ExtractedOmniLight>,
    directional_lights: Query<&ExtractedDirectionalLight>,
) {
    // PERF: view.iter().count() could be views.iter().len() if we implemented ExactSizeIterator for archetype-only filters
    light_meta
        .view_gpu_lights
        .reserve_and_clear(views.iter().count(), &render_device);

    let ambient_color = ambient_light.color.as_rgba_linear() * ambient_light.brightness;
    // set up light data for each view
    for entity in views.iter() {
        let light_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                size: SHADOW_SIZE,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SHADOW_FORMAT,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
                label: None,
            },
        );
        let mut view_lights = Vec::new();

        let mut gpu_lights = GpuLights {
            ambient_color: ambient_color.into(),
            n_omni_lights: omni_lights.iter().len() as u32,
            n_directional_lights: directional_lights.iter().len() as u32,
            omni_lights: [GpuOmniLight::default(); MAX_OMNI_LIGHTS],
            directional_lights: [GpuDirectionalLight::default(); MAX_DIRECTIONAL_LIGHTS],
        };

        // TODO: this should select lights based on relevance to the view instead of the first ones that show up in a query
        for (i, light) in omni_lights.iter().enumerate().take(MAX_OMNI_LIGHTS) {
            let depth_texture_view =
                light_depth_texture
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: None,
                        format: None,
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: None,
                        base_array_layer: i as u32,
                        array_layer_count: NonZeroU32::new(1),
                    });

            let view_transform = GlobalTransform::from_translation(light.transform.translation)
                .looking_at(Vec3::default(), Vec3::Y);
            // TODO: configure light projection based on light configuration
            let projection =
                Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, light.range);

            gpu_lights.omni_lights[i] = GpuOmniLight {
                // premultiply color by intensity
                // we don't use the alpha at all, so no reason to multiply only [0..3]
                color: (light.color.as_rgba_linear() * light.intensity).into(),
                radius: light.radius,
                position: light.transform.translation,
                range: 1.0 / (light.range * light.range),
                // this could technically be copied to the gpu from the light's ViewUniforms
                view_proj: projection * view_transform.compute_matrix().inverse(),
                shadow_bias_min: light.shadow_bias_min_max.x,
                shadow_bias_max: light.shadow_bias_min_max.y,
            };

            let view_light_entity = commands
                .spawn()
                .insert_bundle((
                    ViewLight {
                        depth_texture: depth_texture_view,
                    },
                    ExtractedView {
                        width: SHADOW_SIZE.width,
                        height: SHADOW_SIZE.height,
                        transform: view_transform.clone(),
                        projection,
                    },
                    RenderPhase::<ShadowPhase>::default(),
                ))
                .id();
            view_lights.push(view_light_entity);
        }

        let n_omni_lights = view_lights.len();
        for (i, light) in directional_lights
            .iter()
            .enumerate()
            .take(MAX_DIRECTIONAL_LIGHTS)
        {
            // direction is negated to be ready for N.L
            let dir_to_light = -light.direction;

            // convert from illuminance (lux) to candelas
            //
            // exposure is hard coded at the moment but should be replaced
            // by values coming from the camera
            // see: https://google.github.io/filament/Filament.html#imagingpipeline/physicallybasedcamera/exposuresettings
            const APERTURE: f32 = 4.0;
            const SHUTTER_SPEED: f32 = 1.0 / 250.0;
            const SENSITIVITY: f32 = 100.0;
            let ev100 =
                f32::log2(APERTURE * APERTURE / SHUTTER_SPEED) - f32::log2(SENSITIVITY / 100.0);
            let exposure = 1.0 / (f32::powf(2.0, ev100) * 1.2);
            let intensity = light.illuminance * exposure;

            // NOTE: A directional light seems to have to have an eye position on the line along the direction of the light
            //       through the world origin. I (Rob Swain) do not yet understand why it cannot be translated away from this.
            let view = Mat4::look_at_rh(Vec3::ZERO, light.direction, Vec3::Y);
            // NOTE: This orthographic projection defines the volume within which shadows from a directional light can be cast
            let projection = light.projection;

            gpu_lights.directional_lights[i] = GpuDirectionalLight {
                // premultiply color by intensity
                // we don't use the alpha at all, so no reason to multiply only [0..3]
                color: (light.color.as_rgba_linear() * intensity).into(),
                dir_to_light,
                // NOTE: * view is correct, it should not be view.inverse() here
                view_projection: projection * view,
                shadow_bias_min: light.shadow_bias_min_max.x,
                shadow_bias_max: light.shadow_bias_min_max.y,
            };

            let depth_texture_view =
                light_depth_texture
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: None,
                        format: None,
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: None,
                        base_array_layer: (n_omni_lights + i) as u32,
                        array_layer_count: NonZeroU32::new(1),
                    });

            let view_light_entity = commands
                .spawn()
                .insert_bundle((
                    ViewLight {
                        depth_texture: depth_texture_view,
                    },
                    ExtractedView {
                        width: SHADOW_SIZE.width,
                        height: SHADOW_SIZE.height,
                        transform: GlobalTransform::from_matrix(view.inverse()),
                        projection,
                    },
                    RenderPhase::<ShadowPhase>::default(),
                ))
                .id();
            view_lights.push(view_light_entity);
        }

        commands.entity(entity).insert(ViewLights {
            light_depth_texture: light_depth_texture.texture,
            light_depth_texture_view: light_depth_texture.default_view,
            lights: view_lights,
            gpu_light_binding_index: light_meta.view_gpu_lights.push(gpu_lights),
        });
    }

    light_meta
        .view_gpu_lights
        .write_to_staging_buffer(&render_device);
}

pub struct ShadowPhase;

pub struct ShadowPassNode {
    main_view_query: QueryState<&'static ViewLights>,
    view_light_query: QueryState<(&'static ViewLight, &'static RenderPhase<ShadowPhase>)>,
}

impl ShadowPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
            view_light_query: QueryState::new(world),
        }
    }
}

impl Node for ShadowPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(ShadowPassNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.main_view_query.update_archetypes(world);
        self.view_light_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        if let Some(view_lights) = self.main_view_query.get_manual(world, view_entity).ok() {
            for view_light_entity in view_lights.lights.iter().copied() {
                let (view_light, shadow_phase) = self
                    .view_light_query
                    .get_manual(world, view_light_entity)
                    .unwrap();
                let pass_descriptor = RenderPassDescriptor {
                    label: Some("shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &view_light.depth_texture,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                };

                let draw_functions = world.get_resource::<DrawFunctions>().unwrap();

                let render_pass = render_context
                    .command_encoder
                    .begin_render_pass(&pass_descriptor);
                let mut draw_functions = draw_functions.write();
                let mut tracked_pass = TrackedRenderPass::new(render_pass);
                for drawable in shadow_phase.drawn_things.iter() {
                    let draw_function = draw_functions.get_mut(drawable.draw_function).unwrap();
                    draw_function.draw(
                        world,
                        &mut tracked_pass,
                        view_light_entity,
                        drawable.draw_key,
                        drawable.sort_key,
                    );
                }
            }
        }

        Ok(())
    }
}

type DrawShadowMeshParams<'s, 'w> = (
    Res<'w, ShadowShaders>,
    Res<'w, ExtractedMeshes>,
    Res<'w, LightMeta>,
    Res<'w, MeshMeta>,
    Res<'w, RenderAssets<Mesh>>,
    Query<'w, 's, &'w ViewUniformOffset>,
);
pub struct DrawShadowMesh {
    params: SystemState<DrawShadowMeshParams<'static, 'static>>,
}

impl DrawShadowMesh {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

impl Draw for DrawShadowMesh {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        draw_key: usize,
        _sort_key: usize,
    ) {
        let (shadow_shaders, extracted_meshes, light_meta, mesh_meta, meshes, views) =
            self.params.get(world);
        let view_uniform_offset = views.get(view).unwrap();
        let extracted_mesh = &extracted_meshes.into_inner().meshes[draw_key];
        pass.set_render_pipeline(&shadow_shaders.into_inner().pipeline);
        pass.set_bind_group(
            0,
            light_meta
                .into_inner()
                .shadow_view_bind_group
                .as_ref()
                .unwrap(),
            &[view_uniform_offset.offset],
        );

        pass.set_bind_group(
            1,
            mesh_meta
                .into_inner()
                .mesh_transform_bind_group
                .as_ref()
                .unwrap(),
            &[extracted_mesh.transform_binding_offset],
        );

        let gpu_mesh = meshes.into_inner().get(&extracted_mesh.mesh).unwrap();
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        if let Some(index_info) = &gpu_mesh.index_info {
            pass.set_index_buffer(index_info.buffer.slice(..), 0, IndexFormat::Uint32);
            pass.draw_indexed(0..index_info.count, 0, 0..1);
        } else {
            panic!("non-indexed drawing not supported yet")
        }
    }
}
