use std::fs::File;
use std::io::Write;

use bevy::{
    asset::{AssetPlugin, AssetServerSettings},
    core::{Bytes, CorePlugin},
    diagnostic::DiagnosticsPlugin,
    input::{system::exit_on_esc_system, InputPlugin},
    log::LogPlugin,
    pbr::AmbientLight,
    prelude::{shape, *},
    reflect::TypeUuid,
    render::{
        camera::PerspectiveProjection,
        draw, mesh,
        pass::{
            LoadOp, Operations, PassDescriptor, RenderPassColorAttachment,
            RenderPassDepthStencilAttachment, TextureAttachment,
        },
        pipeline::{
            self, BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState,
            ColorWrite, CompareFunction, DepthBiasState, DepthStencilState, PipelineDescriptor,
            RenderPipeline, StencilFaceState, StencilState,
        },
        render_graph::{
            base::{self, camera, BaseRenderGraphConfig, MainPass},
            fullscreen_pass_node, AssetRenderResourcesNode, FullscreenPassNode,
            GlobalRenderResourcesNode, PassNode, RenderGraph, RenderResourcesNode,
            WindowSwapChainNode, WindowTextureNode,
        },
        renderer::{RenderResource, RenderResources},
        shader::{self, ShaderStage, ShaderStages},
        texture::{
            Extent3d, SamplerDescriptor, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsage,
        },
        RenderStage,
    },
    scene::ScenePlugin,
    transform::TransformSystem,
    window::{WindowId, WindowPlugin},
};
use bevy_fly_camera::{FlyCamera, FlyCameraPlugin};

mod node {
    pub const WINDOW_TEXTURE_SIZE: &str = "WindowTextureSize";
    pub const SSAO_CONFIG: &str = "ssao_config";
    pub const TRANSFORM: &str = "transform";
    pub const MODEL_INV_TRANS_3: &str = "model_inv_trans_3";
    pub const STANDARD_MATERIAL: &str = "standard_material";
    pub const DEPTH_NORMAL_PRE_PASS: &str = "depth_normal_pre_pass_node";
    pub const DEPTH_RENDER_PASS: &str = "depth_render_pass";
    pub const NORMAL_RENDER_PASS: &str = "normal_render_pass";
    pub const SSAO_PASS: &str = "ssao_pass_node";
    pub const SSAO_RENDER_PASS: &str = "ssao_render_pass";
    pub const BLUR_X_PASS: &str = "blur_x_pass_node";
    pub const BLUR_Y_PASS: &str = "blur_y_pass_node";
    pub const DUMMY_SWAPCHAIN_TEXTURE: &str = "dummy_swapchain_texture";
    pub const DUMMY_COLOR_ATTACHMENT: &str = "dummy_color_attachment";
    pub const SAMPLED_COLOR_ATTACHMENT: &str = "sampled_color_attachment";
    pub const DEPTH_TEXTURE: &str = "depth_texture";
    pub const NORMAL_TEXTURE: &str = "normal_texture";
    pub const SSAO_TEXTURE: &str = "ssao_texture";
    pub const SSAO_A_TEXTURE: &str = "ssao_a_texture";
    pub const SSAO_B_TEXTURE: &str = "ssao_b_texture";
}

pub const SSAO_TEXTURE_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Texture::TYPE_UUID, 9247677681468533886);
pub const DEPTH_NORMAL_PIPELINE_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(PipelineDescriptor::TYPE_UUID, 12322817479103657807);

#[derive(Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct DepthNormalPass;

#[derive(Bundle)]
pub struct DepthNormalBundle {
    render_pipelines: RenderPipelines<DepthNormalPass>,
    depth_normal_pass: DepthNormalPass,
    draw: Draw<DepthNormalPass>,
}

impl Default for DepthNormalBundle {
    fn default() -> Self {
        Self {
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                DEPTH_NORMAL_PIPELINE_HANDLE.typed(),
            )]),
            depth_normal_pass: Default::default(),
            draw: Default::default(),
        }
    }
}

#[derive(Debug, Bytes, RenderResources, RenderResource)]
#[render_resources(from_self)]
pub struct SsaoConfig {
    kernel: [Vec4; 32],
    kernel_size: u32,
    radius: f32,
    bias: f32,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            kernel: [
                // precalculated hemisphere kernel (low discrepancy noiser)
                Vec4::new(-0.668154, -0.084296, 0.219458, 0.0),
                Vec4::new(-0.092521, 0.141327, 0.505343, 0.0),
                Vec4::new(-0.041960, 0.700333, 0.365754, 0.0),
                Vec4::new(0.722389, -0.015338, 0.084357, 0.0),
                Vec4::new(-0.815016, 0.253065, 0.465702, 0.0),
                Vec4::new(0.018993, -0.397084, 0.136878, 0.0),
                Vec4::new(0.617953, -0.234334, 0.513754, 0.0),
                Vec4::new(-0.281008, -0.697906, 0.240010, 0.0),
                Vec4::new(0.303332, -0.443484, 0.588136, 0.0),
                Vec4::new(-0.477513, 0.559972, 0.310942, 0.0),
                Vec4::new(0.307240, 0.076276, 0.324207, 0.0),
                Vec4::new(-0.404343, -0.615461, 0.098425, 0.0),
                Vec4::new(0.152483, -0.326314, 0.399277, 0.0),
                Vec4::new(0.435708, 0.630501, 0.169620, 0.0),
                Vec4::new(0.878907, 0.179609, 0.266964, 0.0),
                Vec4::new(-0.049752, -0.232228, 0.264012, 0.0),
                Vec4::new(0.537254, -0.047783, 0.693834, 0.0),
                Vec4::new(0.001000, 0.177300, 0.096643, 0.0),
                Vec4::new(0.626400, 0.524401, 0.492467, 0.0),
                Vec4::new(-0.708714, -0.223893, 0.182458, 0.0),
                Vec4::new(-0.106760, 0.020965, 0.451976, 0.0),
                Vec4::new(-0.285181, -0.388014, 0.241756, 0.0),
                Vec4::new(0.241154, -0.174978, 0.574671, 0.0),
                Vec4::new(-0.405747, 0.080275, 0.055816, 0.0),
                Vec4::new(0.079375, 0.289697, 0.348373, 0.0),
                Vec4::new(0.298047, -0.309351, 0.114787, 0.0),
                Vec4::new(-0.616434, -0.117369, 0.475924, 0.0),
                Vec4::new(-0.035249, 0.134591, 0.840251, 0.0),
                Vec4::new(0.175849, 0.971033, 0.211778, 0.0),
                Vec4::new(0.024805, 0.348056, 0.240006, 0.0),
                Vec4::new(-0.267123, 0.204885, 0.688595, 0.0),
                Vec4::new(-0.077639, -0.753205, 0.070938, 0.0),
            ],
            kernel_size: 8,
            radius: 0.2,
            bias: 0.025,
        }
    }
}

#[derive(Debug, Default)]
struct SceneHandles {
    scene: Option<Handle<Scene>>,
    loaded: bool,
    transform: Transform,
}

#[derive(Debug, RenderResources)]
struct ModelInvTrans3 {
    pub model_inv_trans_3: Mat4,
}

fn main() {
    env_logger::init();

    let mut app = App::build();

    app
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 0.75,
        })
        .insert_resource(Msaa { samples: 1 })
        .insert_resource(WindowDescriptor {
            title: "SSAO demo".to_string(),
            width: 1600.,
            height: 900.,
            ..Default::default()
        });

    app.insert_resource(AssetServerSettings {
        asset_folder: format!("{}/assets", env!("CARGO_MANIFEST_DIR")).to_string(),
    })
    .add_plugin(LogPlugin::default())
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
            add_main_depth_texture: true,
            add_main_pass: true,
            connect_main_pass_to_swapchain: true,
            connect_main_pass_to_main_depth_texture: true,
        }),
    });

    app.add_plugin(bevy::pbr::PbrPlugin::default());

    app.add_plugin(bevy::gltf::GltfPlugin::default());

    app.add_plugin(bevy::winit::WinitPlugin::default());

    app.add_plugin(bevy::wgpu::WgpuPlugin::default());

    // Needed for the DepthNormalPass
    app.register_type::<Draw<DepthNormalPass>>()
        .register_type::<RenderPipelines<DepthNormalPass>>()
        .register_type::<DepthNormalPass>()
        .add_system_to_stage(
            CoreStage::PreUpdate,
            draw::clear_draw_system::<DepthNormalPass>.system(),
        )
        .add_system_to_stage(
            RenderStage::RenderResource,
            mesh::mesh_resource_provider_system::<DepthNormalPass>.system(),
        )
        .add_system_to_stage(
            RenderStage::Draw,
            pipeline::draw_render_pipelines_system::<DepthNormalPass>.system(),
        )
        .add_system_to_stage(
            RenderStage::PostRender,
            shader::clear_shader_defs_system::<DepthNormalPass>.system(),
        )
        .add_system_to_stage(
            CoreStage::PostUpdate,
            shader::asset_shader_defs_system::<StandardMaterial, DepthNormalPass>.system(),
        );

    app.insert_resource(SceneHandles::default())
        .insert_resource(SsaoConfig::default())
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
    mut materials: ResMut<Assets<StandardMaterial>>,
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

    pipelines.set_untracked(DEPTH_NORMAL_PIPELINE_HANDLE, pipeline_descriptor);

    // set_up_scene(&mut commands, &mut materials, &mut meshes);
    // set_up_quad_scene(&mut commands, &mut materials, &mut meshes);
    let scene_handle: Handle<Scene> =
        asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0");
    commands.insert_resource(SceneHandles {
        scene: Some(scene_handle),
        loaded: false,
        transform: Transform::from_matrix(Mat4::from_scale_rotation_translation(
            Vec3::splat(1.0),
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 0.0),
        )),
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
        .insert(FlyCamera {
            enabled: false,
            ..Default::default()
        });
    // .insert(Rotates);
    commands
        .spawn_bundle(PointLightBundle {
            point_light: PointLight {
                intensity: 20.0,
                ..Default::default()
            },
            transform: Transform::from_xyz(3.0, 5.0, 3.0),
            ..Default::default()
        })
        .insert(Rotates);
}

fn scene_loaded(
    mut commands: Commands,
    mut scene_handles: ResMut<SceneHandles>,
    mut scenes: ResMut<Assets<Scene>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if scene_handles.loaded || scene_handles.scene.is_none() {
        return;
    }
    if let Some(scene_handle) = scene_handles.scene.as_ref() {
        if let Some(scene) = scenes.get_mut(scene_handle) {
            let mut query = scene.world.query::<(Entity, &MainPass)>();
            let entities = query
                .iter(&scene.world)
                .map(|(entity, _main_pass)| entity)
                .collect::<Vec<_>>();
            for entity in entities {
                scene
                    .world
                    .entity_mut(entity)
                    .insert_bundle(DepthNormalBundle::default());
                let material_handle = scene
                    .world
                    .entity(entity)
                    .get::<Handle<StandardMaterial>>()
                    .expect("MainPass entity does not have a Handle<StandardMaterial>")
                    .clone();
                let mut material = materials
                    .get_mut(material_handle)
                    .expect("Failed to get material");
                material.ssao_texture = Some(SSAO_TEXTURE_HANDLE.typed());
            }
            let transform = scene_handles.transform;
            commands
                .spawn_bundle((transform, GlobalTransform::default()))
                .with_children(|child_builder| {
                    child_builder.spawn_scene(scene_handle.clone());
                });
            println!("SCENE LOADED!");
            scene_handles.loaded = true;
        }
    }
}

fn set_up_scene(
    commands: &mut Commands,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    meshes: &mut ResMut<Assets<Mesh>>,
) {
    let white_handle = materials.add(Color::WHITE.into());
    let cube_handle = meshes.add(shape::Cube { size: 1.0 }.into());
    let sphere_handle = meshes.add(
        shape::Icosphere {
            radius: 0.5,
            subdivisions: 5,
        }
        .into(),
    );

    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: cube_handle.clone(),
            transform: Transform::from_xyz(-1.0, 0.0, 0.0),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: cube_handle.clone(),
            transform: Transform::from_xyz(0.0, 0.0, -1.0),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: cube_handle,
            transform: Transform::from_xyz(0.0, -1.0, 0.0),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: sphere_handle,
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
}

fn set_up_quad_scene(
    commands: &mut Commands,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    meshes: &mut ResMut<Assets<Mesh>>,
) {
    let white_handle = materials.add(Color::WHITE.into());
    let quad_handle = meshes.add(
        shape::Quad {
            size: Vec2::splat(2.0),
            flip: false,
        }
        .into(),
    );

    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            transform: Transform::from_matrix(Mat4::from_scale_rotation_translation(
                Vec3::new(0.5, 0.5, 0.1),
                Quat::IDENTITY,
                Vec3::new(0.0, 2.0, 0.0),
            )),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: quad_handle.clone(),
            transform: Transform::from_matrix(Mat4::from_rotation_translation(
                Quat::from_rotation_y(90.0f32.to_radians()),
                Vec3::new(-1.0, 2.0, 0.0),
            )),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: quad_handle.clone(),
            transform: Transform::from_matrix(Mat4::from_rotation_translation(
                Quat::from_rotation_y(-90.0f32.to_radians()),
                Vec3::new(1.0, 2.0, 0.0),
            )),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
    commands
        .spawn_bundle(PbrBundle {
            material: white_handle.clone(),
            mesh: quad_handle.clone(),
            transform: Transform::from_xyz(0.0, 2.0, -1.0),
            ..Default::default()
        })
        .insert_bundle(DepthNormalBundle::default());
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
            Some(SSAO_TEXTURE_HANDLE),
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
    // set_up_ssao_render_pass(shaders, pipelines, msaa, render_graph, asset_server);

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
    let mut depth_normal_pass_node =
        PassNode::<DepthNormalPass, &DepthNormalPass>::new(pass_descriptor);
    depth_normal_pass_node.add_camera(camera::CAMERA_3D);
    render_graph.add_node(node::DEPTH_NORMAL_PRE_PASS, depth_normal_pass_node);

    render_graph.add_system_node(
        node::TRANSFORM,
        RenderResourcesNode::<GlobalTransform, DepthNormalPass>::new(true),
    );
    render_graph.add_system_node(
        node::MODEL_INV_TRANS_3,
        RenderResourcesNode::<ModelInvTrans3, DepthNormalPass>::new(true),
    );
    render_graph.add_system_node(
        node::STANDARD_MATERIAL,
        AssetRenderResourcesNode::<StandardMaterial, DepthNormalPass>::new(true),
    );
    render_graph
        .add_node_edge(node::TRANSFORM, node::DEPTH_NORMAL_PRE_PASS)
        .unwrap();
    render_graph
        .add_node_edge(node::MODEL_INV_TRANS_3, node::DEPTH_NORMAL_PRE_PASS)
        .unwrap();
    render_graph
        .add_node_edge(node::STANDARD_MATERIAL, node::DEPTH_NORMAL_PRE_PASS)
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
            attachment: TextureAttachment::Input(node::SSAO_TEXTURE.to_string()),
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

    render_graph.add_system_node(
        node::SSAO_CONFIG,
        GlobalRenderResourcesNode::<SsaoConfig>::new(),
    );
    render_graph
        .add_node_edge(node::SSAO_CONFIG, node::SSAO_PASS)
        .unwrap();

    render_graph
        .add_node_edge(base::node::CAMERA_3D, node::SSAO_PASS)
        .unwrap();
    render_graph
        .add_node_edge(node::DEPTH_NORMAL_PRE_PASS, node::SSAO_PASS)
        .unwrap();
    render_graph
        .add_node_edge(node::SSAO_PASS, base::node::MAIN_PASS)
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
            node::SSAO_TEXTURE,
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
    //         format: TextureFormat::default(),
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
            format: TextureFormat::default(),
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
            fragment: Some(asset_server.load::<Shader, _>("shaders/depth_render.frag")),
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

    set_up_dummy_main_pass(render_graph, msaa);

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
            format: TextureFormat::default(),
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
            fragment: Some(asset_server.load::<Shader, _>("shaders/normal_render.frag")),
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

    set_up_dummy_main_pass(render_graph, msaa);

    render_graph
        .add_node_edge(node::DEPTH_NORMAL_PRE_PASS, node::NORMAL_RENDER_PASS)
        .unwrap();
}

fn set_up_ssao_render_pass(
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
            format: TextureFormat::default(),
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
            fragment: Some(asset_server.load::<Shader, _>("shaders/ssao_render.frag")),
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
        vec![node::SSAO_TEXTURE.into()],
    );
    render_graph.add_node(node::SSAO_RENDER_PASS, pass_node);

    // NOTE: The blur Y pass will read from B and write to A
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::SSAO_RENDER_PASS,
            node::SSAO_TEXTURE,
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_SAMPLER,
            node::SSAO_RENDER_PASS,
            format!("{}_sampler", node::SSAO_TEXTURE),
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
                node::SSAO_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::SSAO_RENDER_PASS,
                "color_resolve_target",
            )
            .unwrap();
    } else {
        render_graph
            .add_slot_edge(
                base::node::PRIMARY_SWAP_CHAIN,
                WindowSwapChainNode::OUT_TEXTURE,
                node::SSAO_RENDER_PASS,
                "color_attachment",
            )
            .unwrap();
    }

    set_up_dummy_main_pass(render_graph, msaa);

    render_graph
        .add_node_edge(node::SSAO_PASS, node::SSAO_RENDER_PASS)
        .unwrap();
}

fn set_up_dummy_main_pass(render_graph: &mut RenderGraph, msaa: &Msaa) {
    // Hack to fill all main pass input slots
    render_graph.add_node(
        node::DUMMY_SWAPCHAIN_TEXTURE,
        WindowTextureNode::new(
            WindowId::primary(),
            TextureDescriptor {
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: 1,
                    height: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::default(),
                usage: TextureUsage::OUTPUT_ATTACHMENT,
            },
            None,
            None,
        ),
    );

    if msaa.samples > 1 {
        render_graph.add_node(
            node::DUMMY_COLOR_ATTACHMENT,
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
                node::DUMMY_COLOR_ATTACHMENT,
                WindowTextureNode::OUT_TEXTURE,
                base::node::MAIN_PASS,
                "color_attachment",
            )
            .unwrap();
        render_graph
            .add_slot_edge(
                node::DUMMY_SWAPCHAIN_TEXTURE,
                WindowTextureNode::OUT_TEXTURE,
                base::node::MAIN_PASS,
                "color_resolve_target",
            )
            .unwrap();
    } else {
        render_graph
            .add_slot_edge(
                node::DUMMY_SWAPCHAIN_TEXTURE,
                WindowTextureNode::OUT_TEXTURE,
                base::node::MAIN_PASS,
                "color_attachment",
            )
            .unwrap();
    }
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
