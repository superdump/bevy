use std::fs::File;
use std::io::Write;

use bevy::{
    asset::AssetPlugin,
    core::CorePlugin,
    diagnostic::DiagnosticsPlugin,
    input::{system::exit_on_esc_system, InputPlugin},
    log::LogPlugin,
    prelude::{shape, *},
    render::{
        camera::PerspectiveProjection,
        pass::{
            LoadOp, Operations, PassDescriptor, RenderPassColorAttachment,
            RenderPassDepthStencilAttachment, TextureAttachment,
        },
        pipeline::{
            BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWrite,
            CompareFunction, DepthBiasState, DepthStencilState, PipelineDescriptor, RenderPipeline,
            StencilFaceState, StencilState,
        },
        render_graph::{
            base::{self, camera, BaseRenderGraphConfig, MainPass},
            fullscreen_pass_node, FullscreenPassNode, PassNode, RenderGraph, RenderResourcesNode,
            WindowSwapChainNode, WindowTextureNode,
        },
        renderer::RenderResources,
        shader::{ShaderStage, ShaderStages},
        texture::{
            Extent3d, SamplerDescriptor, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsage,
        },
    },
    scene::ScenePlugin,
    transform::TransformSystem,
    window::{WindowId, WindowPlugin},
};
use bevy_fly_camera::{FlyCamera, FlyCameraPlugin};

mod node {
    // Resource bindings
    pub const WINDOW_TEXTURE_SIZE: &str = "WindowTextureSize";
    // Nodes
    pub const TRANSFORM: &str = "transform";
    pub const MODEL_INV_TRANS_3: &str = "model_inv_trans_3";
    pub const DEPTH_NORMAL_PRE_PASS: &str = "depth_normal_pre_pass_node";
    pub const DEPTH_RENDER_PASS: &str = "depth_render_pass";
    pub const NORMAL_RENDER_PASS: &str = "normal_render_pass";
    pub const SSAO_PASS: &str = "ssao_pass_node";
    pub const OCCLUSION_RENDER_PASS: &str = "occlusion_render_pass";
    pub const BLUR_X_PASS: &str = "blur_x_pass_node";
    pub const BLUR_Y_PASS: &str = "blur_y_pass_node";
    // Textures
    pub const DUMMY_SWAPCHAIN_TEXTURE: &str = "dummy_swapchain_texture";
    pub const SAMPLED_COLOR_ATTACHMENT: &str = "sampled_color_attachment";
    pub const DEPTH_TEXTURE: &str = "depth_texture";
    pub const NORMAL_TEXTURE: &str = "normal_texture";
    pub const SSAO_A_TEXTURE: &str = "ssao_a_texture";
    pub const SSAO_B_TEXTURE: &str = "ssao_b_texture";
}

#[derive(Debug, Default)]
struct SceneHandles {
    scene: Option<Handle<Scene>>,
    pipeline: Option<Handle<PipelineDescriptor>>,
    loaded: bool,
    scale: f32,
}

#[derive(Debug, RenderResources)]
struct ModelInvTrans3 {
    pub model_inv_trans_3: Mat4,
}

fn main() {
    env_logger::init();

    let mut app = App::build();

    app
        // Pbr
        // .insert_resource(AmbientLight {
        //     color: Color::WHITE,
        //     brightness: 1.0 / 5.0f32,
        // })
        .insert_resource(Msaa { samples: 1 })
        .insert_resource(WindowDescriptor {
            title: "SSAO demo".to_string(),
            width: 1600.,
            height: 900.,
            ..Default::default()
        });

    app.add_plugin(LogPlugin::default())
        .add_plugin(CorePlugin::default())
        .add_plugin(TransformPlugin::default())
        .add_plugin(DiagnosticsPlugin::default())
        .add_plugin(InputPlugin::default())
        .add_plugin(WindowPlugin::default())
        .add_plugin(AssetPlugin::default())
        .add_plugin(ScenePlugin::default());

    // cannot currently override config for a plugin as part of DefaultPlugins
    app.add_plugin(bevy::render::RenderPlugin {
        base_render_graph_config: Some(BaseRenderGraphConfig {
            add_2d_camera: true,
            add_3d_camera: true,
            add_main_depth_texture: false,
            add_main_pass: false,
            connect_main_pass_to_swapchain: false,
            connect_main_pass_to_main_depth_texture: false,
        }),
    });

    // app.add_plugin(bevy::pbr::PbrPlugin::default());

    app.add_plugin(bevy::gltf::GltfPlugin::default());

    app.add_plugin(bevy::winit::WinitPlugin::default());

    app.add_plugin(bevy::wgpu::WgpuPlugin::default());

    app.add_asset::<StandardMaterial>()
        .insert_resource(SceneHandles::default())
        .add_plugin(FlyCameraPlugin)
        .add_system(toggle_fly_camera.system())
        .add_startup_system(setup.system().label("setup"))
        .add_system(rotator_system.system())
        .add_startup_system(
            debug_render_graph
                .system()
                .label("debugdump")
                .after("setup"),
        )
        .add_system(exit_on_esc_system.system())
        .add_system(scene_loaded.system())
        .add_system_to_stage(
            CoreStage::PostUpdate,
            inverse_transpose_transform
                .system()
                .label(node::MODEL_INV_TRANS_3)
                .after(TransformSystem::ParentUpdate)
                .after(TransformSystem::TransformPropagate),
        )
        .run();
}

fn inverse_transpose_transform(
    mut commands: Commands,
    mut queries: QuerySet<(
        Query<(&GlobalTransform, &mut ModelInvTrans3), Changed<GlobalTransform>>,
        Query<(Entity, &GlobalTransform), (With<Handle<Mesh>>, Without<ModelInvTrans3>)>,
    )>,
) {
    for (transform, mut model_inv_trans_3) in queries.q0_mut().iter_mut() {
        model_inv_trans_3.model_inv_trans_3 = inverse_transpose_3(transform);
    }
    for (entity, transform) in queries.q1().iter() {
        commands.entity(entity).insert(ModelInvTrans3 {
            model_inv_trans_3: inverse_transpose_3(transform),
        });
    }
}

fn inverse_transpose_3(transform: &GlobalTransform) -> Mat4 {
    let model = transform.compute_matrix();
    let temp = mat4_to_mat3(&model);
    let inv_trans_3 = temp.inverse().transpose();
    mat3_to_mat4(&inv_trans_3)
}

fn mat4_to_mat3(mat4: &Mat4) -> Mat3 {
    let m = mat4.to_cols_array();
    Mat3::from_cols_array_2d(&[[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]])
}

fn mat3_to_mat4(mat3: &Mat3) -> Mat4 {
    let m = mat3.to_cols_array();
    Mat4::from_cols_array_2d(&[
        [m[0], m[1], m[2], 0.0],
        [m[3], m[4], m[5], 0.0],
        [m[6], m[7], m[8], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn toggle_fly_camera(keyboard_input: Res<Input<KeyCode>>, mut fly_camera: Query<&mut FlyCamera>) {
    if keyboard_input.just_pressed(KeyCode::C) {
        for mut fc in fly_camera.iter_mut() {
            fc.enabled = !fc.enabled;
        }
    }
}

fn debug_render_graph(render_graph: Res<RenderGraph>) {
    let dot = bevy_mod_debugdump::render_graph::render_graph_dot(&*render_graph);
    let mut file = File::create("render_graph.dot").unwrap();
    write!(file, "{}", dot).unwrap();
    println!("*** Updated render_graph.dot");
}

fn setup(
    mut commands: Commands,
    mut render_graph: ResMut<RenderGraph>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut meshes: ResMut<Assets<Mesh>>,
    // mut materials: ResMut<Assets<StandardMaterial>>,
    msaa: Res<Msaa>,
    asset_server: Res<AssetServer>,
) {
    setup_render_graph(
        &mut *render_graph,
        &mut *pipelines,
        &mut *shaders,
        &*msaa,
        &*asset_server,
    );

    asset_server.watch_for_changes().unwrap();

    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
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
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::Bgra8Unorm,
            blend: Some(BlendState {
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: asset_server.load::<Shader, _>("shaders/depth_normal_prepass.vert"),
            fragment: Some(asset_server.load::<Shader, _>("shaders/depth_normal_prepass.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // set_up_scene(&mut commands, &mut meshes, &pipeline_handle);
    // set_up_quad_scene(&mut commands, &mut meshes, &pipeline_handle);
    let scene_handle: Handle<Scene> =
        asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0");
    commands.insert_resource(SceneHandles {
        scene: Some(scene_handle),
        pipeline: Some(pipeline_handle),
        loaded: false,
        scale: 1.0,
    });

    commands
        .spawn_bundle(PerspectiveCameraBundle {
            perspective_projection: PerspectiveProjection {
                near: 0.1,
                ..Default::default()
            },
            transform: Transform::from_xyz(0.0, 1.7, 2.0)
                .looking_at(Vec3::new(0.0, 1.7, 0.0), Vec3::Y),
            ..Default::default()
        })
        .insert(FlyCamera::default());
    // .insert(Rotates);
    commands
        .spawn_bundle(PointLightBundle {
            transform: Transform::from_xyz(3.0, 5.0, 3.0),
            ..Default::default()
        })
        .insert(Rotates);
}

fn scene_loaded(
    mut commands: Commands,
    mut scene_handles: ResMut<SceneHandles>,
    mut scenes: ResMut<Assets<Scene>>,
) {
    if scene_handles.loaded || scene_handles.scene.is_none() || scene_handles.pipeline.is_none() {
        return;
    }
    if let Some(scene) = scenes.get_mut(scene_handles.scene.as_ref().unwrap()) {
        let pipeline_handle = scene_handles.pipeline.as_ref().unwrap();
        let scale = scene_handles.scale;
        commands
            .spawn_bundle((
                Transform::from_matrix(Mat4::from_scale_rotation_translation(
                    Vec3::new(scale, scale, scale),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 1.0, 0.0),
                )),
                GlobalTransform::default(),
            ))
            .with_children(|child_builder| {
                let mut query = scene.world.query::<(&Handle<Mesh>, &Transform)>();
                for (mesh, transform) in query.iter(&mut scene.world) {
                    child_builder.spawn_bundle(MeshBundle {
                        transform: transform.clone(),
                        mesh: mesh.clone(),
                        render_pipelines: RenderPipelines::from_pipelines(vec![
                            RenderPipeline::new(pipeline_handle.clone()),
                        ]),
                        ..Default::default()
                    });
                }
            });
        println!("SCENE LOADED!");
        scene_handles.loaded = true;
    }
}

fn set_up_scene(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    pipeline_handle: &Handle<PipelineDescriptor>,
) {
    let cube_handle = meshes.add(shape::Cube { size: 1.0 }.into());
    let sphere_handle = meshes.add(
        shape::Icosphere {
            radius: 0.5,
            subdivisions: 5,
        }
        .into(),
    );

    commands.spawn_bundle(MeshBundle {
        transform: Transform::from_xyz(-1.0, 0.0, 0.0),
        mesh: cube_handle.clone(),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        transform: Transform::from_xyz(0.0, 0.0, -1.0),
        mesh: cube_handle.clone(),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        transform: Transform::from_xyz(0.0, -1.0, 0.0),
        mesh: cube_handle,
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        mesh: sphere_handle,
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        ..Default::default()
    });
}

fn set_up_quad_scene(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    pipeline_handle: &Handle<PipelineDescriptor>,
) {
    let quad_handle = meshes.add(
        shape::Quad {
            size: Vec2::splat(2.0),
            flip: false,
        }
        .into(),
    );

    commands.spawn_bundle(MeshBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        transform: Transform::from_matrix(Mat4::from_scale_rotation_translation(
            Vec3::new(0.5, 0.5, 0.1),
            Quat::IDENTITY,
            Vec3::new(0.0, 2.0, 0.0),
        )),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        mesh: quad_handle.clone(),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        transform: Transform::from_matrix(Mat4::from_rotation_translation(
            Quat::from_rotation_y(90.0f32.to_radians()),
            Vec3::new(-1.0, 2.0, 0.0),
        )),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        mesh: quad_handle.clone(),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        transform: Transform::from_matrix(Mat4::from_rotation_translation(
            Quat::from_rotation_y(-90.0f32.to_radians()),
            Vec3::new(1.0, 2.0, 0.0),
        )),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        mesh: quad_handle.clone(),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        transform: Transform::from_xyz(0.0, 2.0, -1.0),
        ..Default::default()
    });
}

fn setup_render_graph(
    render_graph: &mut RenderGraph,
    pipelines: &mut Assets<PipelineDescriptor>,
    shaders: &mut Assets<Shader>,
    msaa: &Msaa,
    asset_server: &AssetServer,
) {
    // Set up the additional textures
    render_graph.add_node(
        node::DEPTH_TEXTURE,
        WindowTextureNode::new(
            WindowId::primary(),
            TextureDescriptor {
                size: Extent3d::new(1, 1, 1),
                mip_level_count: 1,
                sample_count: msaa.samples,
                dimension: bevy::render::texture::TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
            },
            Some(SamplerDescriptor::default()),
            None,
        ),
    );
    render_graph.add_node(
        node::NORMAL_TEXTURE,
        WindowTextureNode::new(
            WindowId::primary(),
            TextureDescriptor {
                size: Extent3d::new(1, 1, 1),
                mip_level_count: 1,
                sample_count: msaa.samples,
                dimension: bevy::render::texture::TextureDimension::D2,
                format: TextureFormat::Bgra8Unorm,
                usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
            },
            Some(SamplerDescriptor::default()),
            None,
        ),
    );
    render_graph.add_node(
        node::SSAO_A_TEXTURE,
        WindowTextureNode::new(
            WindowId::primary(),
            TextureDescriptor {
                size: Extent3d::new(1, 1, 1),
                mip_level_count: 1,
                sample_count: msaa.samples,
                dimension: bevy::render::texture::TextureDimension::D2,
                format: TextureFormat::R8Unorm,
                usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
            },
            Some(SamplerDescriptor::default()),
            None,
        ),
    );
    render_graph.add_node(
        node::SSAO_B_TEXTURE,
        WindowTextureNode::new(
            WindowId::primary(),
            TextureDescriptor {
                size: Extent3d::new(1, 1, 1),
                mip_level_count: 1,
                sample_count: msaa.samples,
                dimension: bevy::render::texture::TextureDimension::D2,
                format: TextureFormat::R8Unorm,
                usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
            },
            Some(SamplerDescriptor::default()),
            None,
        ),
    );

    // NOTE: Usage of the below is to only have one of the _render_pass enabled to render the
    // corresponding texture
    // For the main passes of interest (depth normal, ssao, blur) enable up to the one you want to render

    // Set up depth normal pre-pass pipeline
    set_up_depth_normal_pre_pass(msaa, render_graph);

    // // Render the depth texture
    // set_up_depth_render_pass(shaders, pipelines, msaa, render_graph, asset_server);
    // // Render the normal texture
    // set_up_normal_render_pass(shaders, pipelines, msaa, render_graph, asset_server);

    // Set up SSAO pass pipeline
    set_up_ssao_pass(shaders, pipelines, msaa, render_graph, asset_server);
    // Render the occlusion texture after the ssao pass
    set_up_occlusion_render_pass(shaders, pipelines, msaa, render_graph, asset_server);

    // // Set up blur X pass pipeline
    // set_up_blur_x_pass(shaders, pipelines, msaa, render_graph);
    // // Set up blur Y pass pipeline
    // set_up_blur_y_pass(shaders, pipelines, msaa, render_graph);

    // // Set up modified main pass
    // set_up_modified_main_pass(shaders, pipelines, msaa, render_graph);
}

fn set_up_depth_normal_pre_pass(msaa: &Msaa, render_graph: &mut RenderGraph) {
    // Set up pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![RenderPassColorAttachment {
            attachment: TextureAttachment::Input(node::NORMAL_TEXTURE.to_string()),
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::rgb(0.0, 0.0, 1.0)),
                store: true,
            },
        }],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            attachment: TextureAttachment::Input(node::DEPTH_TEXTURE.to_string()),
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: true,
            }),
            stencil_ops: None,
        }),
        sample_count: msaa.samples,
    };

    // Create the pass node
    let mut depth_normal_pass_node = PassNode::<MainPass, &MainPass>::new(pass_descriptor);
    depth_normal_pass_node.add_camera(camera::CAMERA_3D);
    render_graph.add_node(node::DEPTH_NORMAL_PRE_PASS, depth_normal_pass_node);

    render_graph.add_system_node(
        node::TRANSFORM,
        RenderResourcesNode::<GlobalTransform, MainPass>::new(true),
    );
    render_graph.add_system_node(
        node::MODEL_INV_TRANS_3,
        RenderResourcesNode::<ModelInvTrans3, MainPass>::new(true),
    );
    render_graph
        .add_node_edge(node::TRANSFORM, node::DEPTH_NORMAL_PRE_PASS)
        .unwrap();
    render_graph
        .add_node_edge(node::MODEL_INV_TRANS_3, node::DEPTH_NORMAL_PRE_PASS)
        .unwrap();

    render_graph
        .add_node_edge(base::node::TEXTURE_COPY, node::DEPTH_NORMAL_PRE_PASS)
        .unwrap();
    render_graph
        .add_node_edge(base::node::SHARED_BUFFERS, node::DEPTH_NORMAL_PRE_PASS)
        .unwrap();

    render_graph
        .add_node_edge(base::node::CAMERA_3D, node::DEPTH_NORMAL_PRE_PASS)
        .unwrap();

    render_graph
        .add_slot_edge(
            node::NORMAL_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::DEPTH_NORMAL_PRE_PASS,
            node::NORMAL_TEXTURE,
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::DEPTH_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::DEPTH_NORMAL_PRE_PASS,
            node::DEPTH_TEXTURE,
        )
        .unwrap();
}

fn set_up_ssao_pass(
    shaders: &mut Assets<Shader>,
    pipelines: &mut Assets<PipelineDescriptor>,
    msaa: &Msaa,
    render_graph: &mut RenderGraph,
    asset_server: &AssetServer,
) {
    asset_server.watch_for_changes().unwrap();

    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::R8Unorm,
            blend: Some(BlendState {
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                color: BlendComponent {
                    src_factor: BlendFactor::Src,
                    dst_factor: BlendFactor::OneMinusSrc,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                fullscreen_pass_node::shaders::VERTEX_SHADER, // Provides v_Uv
            )),
            fragment: Some(asset_server.load::<Shader, _>("shaders/ssao.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // Setup pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![RenderPassColorAttachment {
            attachment: TextureAttachment::Input("occlusion_texture".to_string()),
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
        sample_count: msaa.samples,
    };

    // Create the pass node
    let mut pass_node = FullscreenPassNode::new(
        pass_descriptor,
        pipeline_handle,
        vec![node::DEPTH_TEXTURE.into(), node::NORMAL_TEXTURE.into()],
    );
    pass_node.add_camera(base::camera::CAMERA_3D);
    render_graph.add_node(node::SSAO_PASS, pass_node);

    render_graph
        .add_node_edge(base::node::CAMERA_3D, node::SSAO_PASS)
        .unwrap();
    render_graph
        .add_node_edge(node::DEPTH_NORMAL_PRE_PASS, node::SSAO_PASS)
        .unwrap();

    render_graph
        .add_slot_edge(
            node::NORMAL_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::SSAO_PASS,
            node::NORMAL_TEXTURE,
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::NORMAL_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::SSAO_PASS,
            format!("{}_sampler", node::NORMAL_TEXTURE),
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::DEPTH_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::SSAO_PASS,
            node::DEPTH_TEXTURE,
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::DEPTH_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::SSAO_PASS,
            format!("{}_sampler", node::DEPTH_TEXTURE),
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::SSAO_PASS,
            "occlusion_texture",
        )
        .unwrap();
}

fn set_up_blur_x_pass(
    shaders: &mut Assets<Shader>,
    pipelines: &mut Assets<PipelineDescriptor>,
    msaa: &Msaa,
    render_graph: &mut RenderGraph,
    asset_server: &AssetServer,
) {
    asset_server.watch_for_changes().unwrap();

    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::R8Unorm,
            blend: Some(BlendState {
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: asset_server.load::<Shader, _>("shaders/blur_x.vert"),
            fragment: Some(asset_server.load::<Shader, _>("shaders/blur_x.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // Setup pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![RenderPassColorAttachment {
            attachment: TextureAttachment::Input(node::SSAO_B_TEXTURE.to_string()),
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::WHITE),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
        sample_count: msaa.samples,
    };

    // Create the pass node
    let pass_node = FullscreenPassNode::new(
        pass_descriptor,
        pipeline_handle,
        vec![node::SSAO_A_TEXTURE.into()],
    );
    render_graph.add_node(node::BLUR_X_PASS, pass_node);

    // NOTE: The blur X pass will read from A and write to B
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::BLUR_X_PASS,
            "blur_input_texture",
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::BLUR_X_PASS,
            "blur_input_texture_sampler",
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_B_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::BLUR_X_PASS,
            "blur_output_texture",
        )
        .unwrap();
}

fn set_up_blur_y_pass(
    shaders: &mut Assets<Shader>,
    pipelines: &mut Assets<PipelineDescriptor>,
    msaa: &Msaa,
    render_graph: &mut RenderGraph,
    asset_server: &AssetServer,
) {
    asset_server.watch_for_changes().unwrap();

    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::R8Unorm,
            blend: Some(BlendState {
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: asset_server.load::<Shader, _>("shaders/blur_y.vert"),
            fragment: Some(asset_server.load::<Shader, _>("shaders/blur_y.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // Setup pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![RenderPassColorAttachment {
            attachment: TextureAttachment::Input(node::SSAO_A_TEXTURE.to_string()),
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::WHITE),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
        sample_count: msaa.samples,
    };

    // Create the pass node
    let pass_node = FullscreenPassNode::new(
        pass_descriptor,
        pipeline_handle,
        vec![node::SSAO_B_TEXTURE.into()],
    );
    render_graph.add_node(node::BLUR_Y_PASS, pass_node);

    // NOTE: The blur Y pass will read from B and write to A
    render_graph
        .add_slot_edge(
            node::SSAO_B_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::BLUR_Y_PASS,
            "blur_input_texture",
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_B_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::BLUR_Y_PASS,
            "blur_input_texture_sampler",
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::BLUR_Y_PASS,
            "blur_output_texture",
        )
        .unwrap();
}

fn set_up_modified_main_pass(
    shaders: &mut Assets<Shader>,
    pipelines: &mut Assets<PipelineDescriptor>,
    msaa: &Msaa,
    render_graph: &mut RenderGraph,
) {
    // // Set up the main pass
    // render_graph
    // .add_slot_edge(
    //     node::SSAO_A_TEXTURE,
    //     WindowTextureNode::OUT_TEXTURE,
    //     base::node::MAIN_PASS,
    //     "ssao".into(),
    // )
    // .unwrap();
    // render_graph
    //     .add_slot_edge(
    //         node::SSAO_A_TEXTURE,
    //         WindowTextureNode::OUT_SAMPLER,
    //         base::node::MAIN_PASS,
    //         "ssao".into(),
    //     )
    //     .unwrap();

    // let pipeline_descriptor = PipelineDescriptor {
    //     depth_stencil: None,
    //     color_target_states: vec![ColorTargetState {
    //         format: TextureFormat::Bgra8UnormSrgb,
    //         color_blend: BlendState {
    //             src_factor: BlendFactor::SrcAlpha,
    //             dst_factor: BlendFactor::OneMinusSrcAlpha,
    //             operation: BlendOperation::Add,
    //         },
    //         alpha_blend: BlendState {
    //             src_factor: BlendFactor::One,
    //             dst_factor: BlendFactor::One,
    //             operation: BlendOperation::Add,
    //         },
    //         write_mask: ColorWrite::ALL,
    //     }],
    //     ..PipelineDescriptor::new(ShaderStages {
    //         vertex: shaders.add(Shader::from_glsl(
    //             ShaderStage::Vertex,
    //             fullscreen_pass_node::shaders::VERTEX_SHADER,
    //         )),
    //         fragment: Some(shaders.add(Shader::from_glsl(
    //             ShaderStage::Fragment,
    //             "#version 450

    //             layout(location=0) in vec2 v_Uv;

    //             layout(set = 0, binding = 0) uniform texture2D color_texture;
    //             layout(set = 0, binding = 1) uniform sampler color_texture_sampler;

    //             layout(location=0) out vec4 o_Target;

    //             void main() {
    //                 o_Target = texture(sampler2D(color_texture, color_texture_sampler), v_Uv);
    //             }
    //             ",
    //         ))),
    //     })
    // };

    // let pipeline_handle = pipelines.add(pipeline_descriptor);

    // // Setup post processing pass
    // let pass_descriptor = PassDescriptor {
    //     color_attachments: vec![RenderPassColorAttachmentDescriptor {
    //         attachment: TextureAttachment::Input("color_attachment".to_string()),
    //         resolve_target: None,
    //         ops: Operations {
    //             load: LoadOp::Clear(Color::rgb(0.1, 0.2, 0.3)),
    //             store: true,
    //         },
    //     }],
    //     depth_stencil_attachment: None,
    //     sample_count: 1,
    // };

    // // Create the pass node
    // let post_pass_node = FullscreenPassNode::new(
    //     pass_descriptor,
    //     pipeline_handle,
    //     vec!["color_texture".into()],
    // );
    // render_graph.add_node(node::POST_PASS, post_pass_node);

    // // Run after main pass
    // render_graph
    //     .add_node_edge(base::node::MAIN_PASS, node::POST_PASS)
    //     .unwrap();

    // // Connect color_attachment
    // render_graph
    //     .add_slot_edge(
    //         base::node::PRIMARY_SWAP_CHAIN,
    //         WindowTextureNode::OUT_TEXTURE,
    //         node::POST_PASS,
    //         "color_attachment",
    //     )
    //     .unwrap();

    // // Connect extra texture and sampler input
    // render_graph
    //     .add_slot_edge(
    //         node::MAIN_COLOR_TEXTURE,
    //         WindowTextureNode::OUT_TEXTURE,
    //         node::POST_PASS,
    //         "color_texture",
    //     )
    //     .unwrap();

    // render_graph
    //     .add_slot_edge(
    //         node::MAIN_COLOR_TEXTURE,
    //         WindowTextureNode::OUT_SAMPLER,
    //         node::POST_PASS,
    //         "color_texture_sampler",
    //     )
    //     .unwrap();
}

fn set_up_depth_render_pass(
    shaders: &mut Assets<Shader>,
    pipelines: &mut Assets<PipelineDescriptor>,
    msaa: &Msaa,
    render_graph: &mut RenderGraph,
    asset_server: &AssetServer,
) {
    asset_server.watch_for_changes().unwrap();

    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::Bgra8UnormSrgb,
            blend: Some(BlendState {
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                fullscreen_pass_node::shaders::VERTEX_SHADER,
            )),
            fragment: Some(asset_server.load::<Shader, _>("shaders/depth.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // Setup pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![msaa.color_attachment(
            TextureAttachment::Input("color_attachment".to_string()),
            TextureAttachment::Input("color_resolve_target".to_string()),
            Operations {
                load: LoadOp::Load,
                store: true,
            },
        )],
        depth_stencil_attachment: None,
        sample_count: msaa.samples,
    };

    // Create the pass node
    let pass_node = FullscreenPassNode::new(
        pass_descriptor,
        pipeline_handle,
        vec![node::DEPTH_TEXTURE.into()],
    );
    render_graph.add_node(node::DEPTH_RENDER_PASS, pass_node);

    // NOTE: The blur Y pass will read from B and write to A
    render_graph
        .add_slot_edge(
            node::DEPTH_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::DEPTH_RENDER_PASS,
            node::DEPTH_TEXTURE,
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::DEPTH_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::DEPTH_RENDER_PASS,
            format!("{}_sampler", node::DEPTH_TEXTURE),
        )
        .unwrap();
    if msaa.samples > 1 {
        render_graph.add_node(
            node::SAMPLED_COLOR_ATTACHMENT,
            WindowTextureNode::new(
                WindowId::primary(),
                TextureDescriptor {
                    size: Extent3d {
                        depth_or_array_layers: 1,
                        width: 1,
                        height: 1,
                    },
                    mip_level_count: 1,
                    sample_count: msaa.samples,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::default(),
                    usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
                },
                None,
                None,
            ),
        );
        render_graph
            .add_slot_edge(
                node::SAMPLED_COLOR_ATTACHMENT,
                WindowTextureNode::OUT_TEXTURE,
                node::DEPTH_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::DEPTH_RENDER_PASS,
                "color_resolve_target",
            )
            .unwrap();
    } else {
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::DEPTH_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
    }

    // // Hack to fill all main pass input slots
    // render_graph.add_node(
    //     node::DUMMY_SWAPCHAIN_TEXTURE,
    //     WindowTextureNode::new(
    //         WindowId::primary(),
    //         TextureDescriptor {
    //             size: Extent3d {
    //                 depth: 1,
    //                 width: 1,
    //                 height: 1,
    //             },
    //             mip_level_count: 1,
    //             sample_count: 1,
    //             dimension: TextureDimension::D2,
    //             format: TextureFormat::default(),
    //             usage: TextureUsage::OUTPUT_ATTACHMENT,
    //         },
    //         None,
    //         None,
    //     ),
    // );
    // render_graph
    //     .add_slot_edge(
    //         node::DUMMY_SWAPCHAIN_TEXTURE,
    //         WindowTextureNode::OUT_TEXTURE,
    //         base::node::MAIN_PASS,
    //         "color_resolve_target",
    //     )
    //     .unwrap();

    render_graph
        .add_node_edge(node::DEPTH_NORMAL_PRE_PASS, node::DEPTH_RENDER_PASS)
        .unwrap();
}

fn set_up_normal_render_pass(
    shaders: &mut Assets<Shader>,
    pipelines: &mut Assets<PipelineDescriptor>,
    msaa: &Msaa,
    render_graph: &mut RenderGraph,
    asset_server: &AssetServer,
) {
    asset_server.watch_for_changes().unwrap();

    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::Bgra8UnormSrgb,
            blend: Some(BlendState {
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                fullscreen_pass_node::shaders::VERTEX_SHADER,
            )),
            fragment: Some(asset_server.load::<Shader, _>("shaders/normal.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // Setup pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![msaa.color_attachment(
            TextureAttachment::Input("color_attachment".to_string()),
            TextureAttachment::Input("color_resolve_target".to_string()),
            Operations {
                load: LoadOp::Load,
                store: true,
            },
        )],
        depth_stencil_attachment: None,
        sample_count: msaa.samples,
    };

    // Create the pass node
    let pass_node = FullscreenPassNode::new(
        pass_descriptor,
        pipeline_handle,
        vec![node::NORMAL_TEXTURE.into()],
    );
    render_graph.add_node(node::NORMAL_RENDER_PASS, pass_node);

    // NOTE: The blur Y pass will read from B and write to A
    render_graph
        .add_slot_edge(
            node::NORMAL_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::NORMAL_RENDER_PASS,
            node::NORMAL_TEXTURE,
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::NORMAL_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::NORMAL_RENDER_PASS,
            format!("{}_sampler", node::NORMAL_TEXTURE),
        )
        .unwrap();
    if msaa.samples > 1 {
        render_graph.add_node(
            node::SAMPLED_COLOR_ATTACHMENT,
            WindowTextureNode::new(
                WindowId::primary(),
                TextureDescriptor {
                    size: Extent3d {
                        depth_or_array_layers: 1,
                        width: 1,
                        height: 1,
                    },
                    mip_level_count: 1,
                    sample_count: msaa.samples,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::default(),
                    usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
                },
                None,
                None,
            ),
        );
        render_graph
            .add_slot_edge(
                node::SAMPLED_COLOR_ATTACHMENT,
                WindowTextureNode::OUT_TEXTURE,
                node::NORMAL_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::NORMAL_RENDER_PASS,
                "color_resolve_target",
            )
            .unwrap();
    } else {
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::NORMAL_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
    }

    // // Hack to fill all main pass input slots
    // render_graph.add_node(
    //     node::DUMMY_SWAPCHAIN_TEXTURE,
    //     WindowTextureNode::new(
    //         WindowId::primary(),
    //         TextureDescriptor {
    //             size: Extent3d {
    //                 depth: 1,
    //                 width: 1,
    //                 height: 1,
    //             },
    //             mip_level_count: 1,
    //             sample_count: 1,
    //             dimension: TextureDimension::D2,
    //             format: TextureFormat::default(),
    //             usage: TextureUsage::OUTPUT_ATTACHMENT,
    //         },
    //         None,
    //         None,
    //     ),
    // );
    // render_graph
    //     .add_slot_edge(
    //         node::DUMMY_SWAPCHAIN_TEXTURE,
    //         WindowTextureNode::OUT_TEXTURE,
    //         base::node::MAIN_PASS,
    //         "color_resolve_target",
    //     )
    //     .unwrap();

    render_graph
        .add_node_edge(node::DEPTH_NORMAL_PRE_PASS, node::NORMAL_RENDER_PASS)
        .unwrap();
}

fn set_up_occlusion_render_pass(
    shaders: &mut Assets<Shader>,
    pipelines: &mut Assets<PipelineDescriptor>,
    msaa: &Msaa,
    render_graph: &mut RenderGraph,
    asset_server: &AssetServer,
) {
    asset_server.watch_for_changes().unwrap();

    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::Bgra8UnormSrgb,
            blend: Some(BlendState {
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                fullscreen_pass_node::shaders::VERTEX_SHADER,
            )),
            fragment: Some(asset_server.load::<Shader, _>("shaders/occlusion.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // Setup pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![msaa.color_attachment(
            TextureAttachment::Input("color_attachment".to_string()),
            TextureAttachment::Input("color_resolve_target".to_string()),
            Operations {
                load: LoadOp::Load,
                store: true,
            },
        )],
        depth_stencil_attachment: None,
        sample_count: msaa.samples,
    };

    // Create the pass node
    let pass_node = FullscreenPassNode::new(
        pass_descriptor,
        pipeline_handle,
        vec!["occlusion_texture".into()],
    );
    render_graph.add_node(node::OCCLUSION_RENDER_PASS, pass_node);

    // NOTE: The blur Y pass will read from B and write to A
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::OCCLUSION_RENDER_PASS,
            "occlusion_texture",
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::OCCLUSION_RENDER_PASS,
            format!("{}_sampler", "occlusion_texture"),
        )
        .unwrap();
    if msaa.samples > 1 {
        render_graph.add_node(
            node::SAMPLED_COLOR_ATTACHMENT,
            WindowTextureNode::new(
                WindowId::primary(),
                TextureDescriptor {
                    size: Extent3d {
                        depth_or_array_layers: 1,
                        width: 1,
                        height: 1,
                    },
                    mip_level_count: 1,
                    sample_count: msaa.samples,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::default(),
                    usage: TextureUsage::OUTPUT_ATTACHMENT | TextureUsage::SAMPLED,
                },
                None,
                None,
            ),
        );
        render_graph
            .add_slot_edge(
                node::SAMPLED_COLOR_ATTACHMENT,
                WindowTextureNode::OUT_TEXTURE,
                node::OCCLUSION_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::OCCLUSION_RENDER_PASS,
                "color_resolve_target",
            )
            .unwrap();
    } else {
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::OCCLUSION_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
    }

    // // Hack to fill all main pass input slots
    // render_graph.add_node(
    //     node::DUMMY_SWAPCHAIN_TEXTURE,
    //     WindowTextureNode::new(
    //         WindowId::primary(),
    //         TextureDescriptor {
    //             size: Extent3d {
    //                 depth: 1,
    //                 width: 1,
    //                 height: 1,
    //             },
    //             mip_level_count: 1,
    //             sample_count: 1,
    //             dimension: TextureDimension::D2,
    //             format: TextureFormat::default(),
    //             usage: TextureUsage::OUTPUT_ATTACHMENT,
    //         },
    //         None,
    //         None,
    //     ),
    // );
    // render_graph
    //     .add_slot_edge(
    //         node::DUMMY_SWAPCHAIN_TEXTURE,
    //         WindowTextureNode::OUT_TEXTURE,
    //         base::node::MAIN_PASS,
    //         "color_resolve_target",
    //     )
    //     .unwrap();

    render_graph
        .add_node_edge(node::SSAO_PASS, node::OCCLUSION_RENDER_PASS)
        .unwrap();
}

/// this component indicates what entities should rotate
struct Rotates;

fn rotator_system(time: Res<Time>, mut query: Query<&mut Transform, With<Rotates>>) {
    for mut transform in query.iter_mut() {
        *transform = Transform::from_rotation(Quat::from_rotation_y(
            (4.0 * std::f32::consts::PI / 20.0) * time.delta_seconds(),
        )) * *transform;
    }
}
