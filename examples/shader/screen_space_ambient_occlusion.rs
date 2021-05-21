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
        camera::{Camera, PerspectiveProjection},
        pass::{
            LoadOp, Operations, PassDescriptor, RenderPassColorAttachmentDescriptor,
            RenderPassDepthStencilAttachmentDescriptor, TextureAttachment,
        },
        pipeline::{
            BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWrite, CompareFunction,
            DepthBiasState, DepthStencilState, PipelineDescriptor, RenderPipeline,
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
    window::{WindowId, WindowPlugin},
};

mod node {
    // Resource bindings
    pub const CAMERA_INV_PROJ: &str = "CameraInvProj";
    pub const WINDOW_TEXTURE_SIZE: &str = "WindowTextureSize";
    // Nodes
    pub const TRANSFORM: &str = "transform";
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

    app.add_startup_system(update_camera_inverse_projection.system())
        .add_startup_system(setup.system().label("setup"))
        .add_system_to_stage(
            CoreStage::PostUpdate,
            update_camera_inverse_projection.system(),
        )
        .add_system(rotator_system.system())
        .add_startup_system(
            debug_render_graph
                .system()
                .label("debugdump")
                .after("setup"),
        )
        .add_system(exit_on_esc_system.system())
        .run();
}

fn debug_render_graph(render_graph: Res<RenderGraph>) {
    let dot = bevy_mod_debugdump::render_graph::render_graph_dot(&*render_graph);
    let mut file = File::create("render_graph.dot").unwrap();
    write!(file, "{}", dot).unwrap();
    println!("*** Updated render_graph.dot");
}
#[derive(Debug, RenderResources)]
pub struct CameraInvProj {
    pub inverse_projection: Mat4,
}

fn update_camera_inverse_projection(
    mut commands: Commands,
    to_init: Query<(Entity, &Camera), (With<PerspectiveProjection>, Without<CameraInvProj>)>,
    mut to_update: Query<
        (&Camera, &mut CameraInvProj),
        (With<PerspectiveProjection>, Changed<Camera>),
    >,
) {
    for (entity, camera) in to_init.iter() {
        // If the determinant is 0, then the matrix is not invertible
        debug_assert!(camera.projection_matrix.determinant().abs() > 1e-5);
        commands.entity(entity).insert(CameraInvProj {
            inverse_projection: camera.projection_matrix.inverse(),
        });
    }
    for (camera, mut camera_inverse_projection) in to_update.iter_mut() {
        // If the determinant is 0, then the matrix is not invertible
        debug_assert!(camera.projection_matrix.determinant().abs() > 1e-5);
        camera_inverse_projection.inverse_projection = camera.projection_matrix.inverse();
    }
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
    // render_graph.add_system_node(
    //     node::CAMERA_INV_PROJ,
    //     RenderResourcesNode::<CameraInvProj>::new(true),
    // );

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
            clamp_depth: false,
        }),
        color_target_states: vec![ColorTargetState {
            format: TextureFormat::Bgra8Unorm,
            color_blend: BlendState {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: asset_server.load::<Shader, _>("shaders/depth_normal_prepass.vert"),
            fragment: Some(asset_server.load::<Shader, _>("shaders/depth_normal_prepass.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

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
        // material: materials.add(Color::PINK.into()),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        transform: Transform::from_xyz(0.0, 0.0, -1.0),
        mesh: cube_handle.clone(),
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        // material: materials.add(Color::PURPLE.into()),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        transform: Transform::from_xyz(0.0, -1.0, 0.0),
        mesh: cube_handle,
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle.clone(),
        )]),
        // material: materials.add(Color::TEAL.into()),
        ..Default::default()
    });
    commands.spawn_bundle(MeshBundle {
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        mesh: sphere_handle,
        render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
            pipeline_handle,
        )]),
        // material: materials.add(Color::YELLOW.into()),
        ..Default::default()
    });
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_xyz(2.0, 2.0, 2.0)
                .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
            ..Default::default()
        })
        .insert(Rotates);
    commands
        .spawn_bundle(PointLightBundle {
            transform: Transform::from_xyz(3.0, 5.0, 3.0),
            ..Default::default()
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
        color_attachments: vec![RenderPassColorAttachmentDescriptor {
            attachment: TextureAttachment::Input(node::NORMAL_TEXTURE.to_string()),
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::rgb(0.0, 0.0, 1.0)),
                store: true,
            },
        }],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
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
    let mut depth_normal_pass_node = PassNode::<&MainPass>::new(pass_descriptor);
    depth_normal_pass_node.add_camera(camera::CAMERA_3D);
    render_graph.add_node(node::DEPTH_NORMAL_PRE_PASS, depth_normal_pass_node);

    render_graph.add_system_node(
        node::TRANSFORM,
        RenderResourcesNode::<GlobalTransform>::new(true),
    );
    render_graph
        .add_node_edge(node::TRANSFORM, node::DEPTH_NORMAL_PRE_PASS)
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
            color_blend: BlendState {
                src_factor: BlendFactor::SrcColor,
                dst_factor: BlendFactor::OneMinusSrcColor,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            write_mask: ColorWrite::ALL,
        }],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,                          // FIXME: Fix the shaders!
                fullscreen_pass_node::shaders::VERTEX_SHADER, // Provides v_Uv
            )),
            fragment: Some(asset_server.load::<Shader, _>("shaders/ssao.frag")),
        })
    };

    let pipeline_handle = pipelines.add(pipeline_descriptor);

    // Setup pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![RenderPassColorAttachmentDescriptor {
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
    let pass_node = FullscreenPassNode::new(
        pass_descriptor,
        pipeline_handle,
        vec![node::DEPTH_TEXTURE.into(), node::NORMAL_TEXTURE.into()],
    );
    render_graph.add_node(node::SSAO_PASS, pass_node);

    // render_graph
    //     .add_node_edge(base::node::SHARED_BUFFERS, node::SSAO_PASS)
    //     .unwrap();
    // render_graph
    //     .add_node_edge(base::node::CAMERA_3D, node::SSAO_PASS)
    //     .unwrap();
    // render_graph
    //     .add_node_edge(node::CAMERA_INV_PROJ, node::SSAO_PASS)
    //     .unwrap();
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
            color_blend: BlendState {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
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
        color_attachments: vec![RenderPassColorAttachmentDescriptor {
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
            color_blend: BlendState {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
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
        color_attachments: vec![RenderPassColorAttachmentDescriptor {
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
            color_blend: BlendState {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
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
        color_attachments: vec![msaa.color_attachment_descriptor(
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
                        depth: 1,
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
            color_blend: BlendState {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
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
        color_attachments: vec![msaa.color_attachment_descriptor(
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
                        depth: 1,
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
            color_blend: BlendState {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendState {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
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
        color_attachments: vec![msaa.color_attachment_descriptor(
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
                        depth: 1,
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
