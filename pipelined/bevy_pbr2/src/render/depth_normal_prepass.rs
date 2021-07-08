use crate::{ExtractedMeshes, MeshMeta, PbrShaders};
use bevy_ecs::{prelude::*, system::SystemState};
use bevy_render2::{
    camera::{ActiveCameras, CameraPlugin},
    color::Color,
    mesh::Mesh,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_phase::{Draw, DrawFunctions, Drawable, RenderPhase, TrackedRenderPass},
    render_resource::*,
    renderer::{RenderContext, RenderDevice},
    shader::Shader,
    texture::*,
    view::{ExtractedView, ViewMeta, ViewUniformOffset},
};
use std::num::NonZeroU32;

pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

pub struct DepthNormalShaders {
    pub pipeline: RenderPipeline,
    pub view_layout: BindGroupLayout,
    pub depth_sampler: Sampler,
    pub normal_sampler: Sampler,
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for DepthNormalShaders {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let pbr_shaders = world.get_resource::<PbrShaders>().unwrap();

        let shader = Shader::from_wgsl(include_str!("depth_normal_prepass.wgsl"));
        let shader_module = render_device.create_shader_module(&shader);

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(336),
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
                module: &shader_module,
                entry_point: "vertex",
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "fragment",
                targets: &[ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
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
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
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

        DepthNormalShaders {
            pipeline,
            view_layout,
            depth_sampler: render_device.create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                compare: None,
                ..Default::default()
            }),
            normal_sampler: render_device.create_sampler(&SamplerDescriptor::default()),
        }
    }
}

pub fn extract_depth_normal_prepass_camera_phase(
    mut commands: Commands,
    active_cameras: Res<ActiveCameras>,
) {
    if let Some(camera_3d) = active_cameras.get(CameraPlugin::CAMERA_3D) {
        if let Some(entity) = camera_3d.entity {
            commands
                .get_or_spawn(entity)
                .insert(RenderPhase::<DepthNormalPhase>::default());
        }
    }
}

pub struct ViewDepthNormal {
    pub view_depth_texture: Texture,
    pub view_depth_texture_view: TextureView,
    pub view_normal_texture: Texture,
    pub view_normal_texture_view: TextureView,
}

pub fn prepare_view_depth_normals(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedView), With<RenderPhase<DepthNormalPhase>>>,
) {
    // set up depth normal for each view
    for (entity, view) in views.iter() {
        // FIXME: Make ViewDepthTexture sampled and then use that instead?
        let view_depth_texture = texture_cache.get(
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
                format: TextureFormat::Depth32Float, /* PERF: vulkan docs recommend using 24
                                                      * bit depth for better performance */
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            },
        );
        let view_depth_texture_view =
            view_depth_texture
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
        let view_normal_texture = texture_cache.get(
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
                format: TextureFormat::bevy_default(),
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            },
        );
        let view_normal_texture_view =
            view_normal_texture
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
        commands.entity(entity).insert(ViewDepthNormal {
            view_depth_texture: view_depth_texture.texture,
            view_depth_texture_view,
            view_normal_texture: view_normal_texture.texture,
            view_normal_texture_view,
        });
    }
}

pub struct DepthNormalMeshBindGroups {
    view_bind_group: BindGroup,
}

pub fn queue_meshes(
    mut commands: Commands,
    draw_functions: Res<DrawFunctions>,
    render_device: Res<RenderDevice>,
    depth_normal_shaders: Res<DepthNormalShaders>,
    view_meta: Res<ViewMeta>,
    extracted_meshes: Res<ExtractedMeshes>,
    mut views: Query<(Entity, &ExtractedView, &mut RenderPhase<DepthNormalPhase>)>,
) {
    if extracted_meshes.meshes.is_empty() {
        return;
    }

    for (entity, view, mut depth_normal_phase) in views.iter_mut() {
        // TODO: cache this?
        let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: view_meta.uniforms.binding(),
            }],
            label: None,
            layout: &depth_normal_shaders.view_layout,
        });

        commands
            .entity(entity)
            .insert(DepthNormalMeshBindGroups { view_bind_group });

        let draw_depth_normal = draw_functions
            .read()
            .get_id::<DrawDepthNormalMesh>()
            .unwrap();
        let view_matrix = view.transform.compute_matrix();
        let view_row_2 = view_matrix.row(2);
        for (i, mesh) in extracted_meshes.meshes.iter().enumerate() {
            // NOTE: row 2 of the view matrix dotted with column 3 of the model matrix
            //       gives the z component of translation of the mesh in view space
            let mesh_z = view_row_2.dot(mesh.transform.col(3));
            // FIXME: Switch from usize to u64 for portability and use sort key encoding
            //        similar to https://realtimecollisiondetection.net/blog/?p=86 as appropriate
            // FIXME: What is the best way to map from view space z to a number of bits of unsigned integer?
            let sort_key = (mesh_z * 1000.0) as usize;
            depth_normal_phase.add(Drawable {
                draw_function: draw_depth_normal,
                draw_key: i,
                sort_key,
            });
        }
    }
}

pub struct DepthNormalPhase;

pub struct DepthNormalPassNode {
    main_view_query: QueryState<(
        &'static ViewDepthNormal,
        &'static RenderPhase<DepthNormalPhase>,
    )>,
}

impl DepthNormalPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
        }
    }
}

impl Node for DepthNormalPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(
            DepthNormalPassNode::IN_VIEW,
            SlotType::Entity,
        )]
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
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        if let Some((view_depth_normal, depth_normal_phase)) =
            self.main_view_query.get_manual(world, view_entity).ok()
        {
            let pass_descriptor = RenderPassDescriptor {
                label: Some("depth_normal_prepass"),
                color_attachments: &[RenderPassColorAttachment {
                    view: &view_depth_normal.view_normal_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK.into()),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &view_depth_normal.view_depth_texture_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
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
            for drawable in depth_normal_phase.drawn_things.iter() {
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

type DrawDepthNormalMeshParams<'s, 'w> = (
    Res<'w, DepthNormalShaders>,
    Res<'w, ExtractedMeshes>,
    Res<'w, MeshMeta>,
    Res<'w, RenderAssets<Mesh>>,
    Query<'w, 's, (&'w ViewUniformOffset, &'w DepthNormalMeshBindGroups)>,
);
pub struct DrawDepthNormalMesh {
    params: SystemState<DrawDepthNormalMeshParams<'static, 'static>>,
}

impl DrawDepthNormalMesh {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

impl Draw for DrawDepthNormalMesh {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        draw_key: usize,
        _sort_key: usize,
    ) {
        let (depth_normal_shaders, extracted_meshes, mesh_meta, meshes, views) =
            self.params.get(world);
        let (view_uniform_offset, depth_normal_mesh_bind_groups) = views.get(view).unwrap();
        let extracted_mesh = &extracted_meshes.into_inner().meshes[draw_key];
        pass.set_render_pipeline(&depth_normal_shaders.into_inner().pipeline);
        pass.set_bind_group(
            0,
            &depth_normal_mesh_bind_groups.view_bind_group,
            &[view_uniform_offset.offset],
        );

        let transform_bindgroup_key = mesh_meta.mesh_transform_bind_group_key.unwrap();
        pass.set_bind_group(
            1,
            mesh_meta
                .into_inner()
                .mesh_transform_bind_group
                .get_value(transform_bindgroup_key)
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
