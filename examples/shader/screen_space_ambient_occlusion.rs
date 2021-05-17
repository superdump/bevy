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
) {
    // render_graph.add_system_node(
    //     node::CAMERA_INV_PROJ,
    //     RenderResourcesNode::<CameraInvProj>::new(true),
    // );

    setup_render_graph(&mut *render_graph, &mut *pipelines, &mut *shaders, &*msaa);

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
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                "#version 450

                layout(location = 0) in vec3 Vertex_Position;
                layout(location = 1) in vec3 Vertex_Normal;
                
                layout(location = 0) out vec3 v_ViewNormal;
                
                layout(set = 0, binding = 0) uniform CameraViewProj {
                    mat4 ViewProj;
                };
                layout(set = 1, binding = 0) uniform CameraView {
                    mat4 View;
                };
                
                layout(set = 2, binding = 0) uniform Transform {
                    mat4 Model;
                };
                
                void main() {
                    v_ViewNormal = mat3(inverse(View)) * mat3(Model) * Vertex_Normal;
                    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
                }
                ",
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                layout(location = 0) in vec3 v_ViewNormal;

                layout(location = 0) out vec4 o_Target;

                void main() {
                    o_Target = vec4(v_ViewNormal * 0.5 + 0.5, 1.0);
                }
                ",
            ))),
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
        mesh: cube_handle.clone(),
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
            pipeline_handle.clone(),
        )]),
        // material: materials.add(Color::YELLOW.into()),
        ..Default::default()
    });
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_xyz(2.0, 2.0, 2.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        ..Default::default()
    });
    // .insert(Rotates);
    commands
        .spawn_bundle(PointLightBundle {
            transform: Transform::from_xyz(3.0, 5.0, 3.0),
            ..Default::default()
        })
        .insert(Rotates);
}

fn set_up_depth_normal_pre_pass(msaa: &Msaa, render_graph: &mut RenderGraph) {
    // Set up pass
    let pass_descriptor = PassDescriptor {
        color_attachments: vec![RenderPassColorAttachmentDescriptor {
            attachment: TextureAttachment::Input(node::NORMAL_TEXTURE.to_string()),
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::WHITE),
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
) {
    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![
            ColorTargetState {
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
            },
        ],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex, // FIXME: Fix the shaders!
                fullscreen_pass_node::shaders::VERTEX_SHADER, // Provides v_Uv
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                // FIXME: Move to a uniform!
                const float kernel[32][3] = // precalculated hemisphere kernel (low discrepancy noiser)
                {
                    {-0.668154f, -0.084296f, 0.219458f},
                    {-0.092521f,  0.141327f, 0.505343f},
                    {-0.041960f,  0.700333f, 0.365754f},
                    { 0.722389f, -0.015338f, 0.084357f},
                    {-0.815016f,  0.253065f, 0.465702f},
                    { 0.018993f, -0.397084f, 0.136878f},
                    { 0.617953f, -0.234334f, 0.513754f},
                    {-0.281008f, -0.697906f, 0.240010f},
                    { 0.303332f, -0.443484f, 0.588136f},
                    {-0.477513f,  0.559972f, 0.310942f},
                    { 0.307240f,  0.076276f, 0.324207f},
                    {-0.404343f, -0.615461f, 0.098425f},
                    { 0.152483f, -0.326314f, 0.399277f},
                    { 0.435708f,  0.630501f, 0.169620f},
                    { 0.878907f,  0.179609f, 0.266964f},
                    {-0.049752f, -0.232228f, 0.264012f},
                    { 0.537254f, -0.047783f, 0.693834f},
                    { 0.001000f,  0.177300f, 0.096643f},
                    { 0.626400f,  0.524401f, 0.492467f},
                    {-0.708714f, -0.223893f, 0.182458f},
                    {-0.106760f,  0.020965f, 0.451976f},
                    {-0.285181f, -0.388014f, 0.241756f},
                    { 0.241154f, -0.174978f, 0.574671f},
                    {-0.405747f,  0.080275f, 0.055816f},
                    { 0.079375f,  0.289697f, 0.348373f},
                    { 0.298047f, -0.309351f, 0.114787f},
                    {-0.616434f, -0.117369f, 0.475924f},
                    {-0.035249f,  0.134591f, 0.840251f},
                    { 0.175849f,  0.971033f, 0.211778f},
                    { 0.024805f,  0.348056f, 0.240006f},
                    {-0.267123f,  0.204885f, 0.688595f},
                    {-0.077639f, -0.753205f, 0.070938f}
                };

                layout(location = 0) in vec2 v_Uv;

                layout(location = 0) out float o_Target;

                // layout(set = 0, binding = 0) uniform CameraProj {
                //     mat4 Proj;
                // };
                // layout(set = 1, binding = 1) uniform CameraInvProj {
                //     mat4 InvProj;
                // };

                layout(set = 0, binding = 0) uniform texture2D depth_texture;
                layout(set = 0, binding = 1) uniform sampler depth_texture_sampler;
                layout(set = 0, binding = 2) uniform texture2D normal_texture;
                layout(set = 0, binding = 3) uniform sampler normal_texture_sampler;
                // layout(set = 2, binding = 4) uniform texture2D noise_texture;
                // layout(set = 2, binding = 5) uniform sampler noise_texture_sampler;

                // FIXME: Make these into uniforms
                const float RADIUS = 0.5;
                const float BIAS = 0.025;
                const int KERNEL_SIZE = 32;

                // From Matt Pettineo's article: https://mynameismjp.wordpress.com/2010/09/05/position-from-depth-3/
                // but I derived my own method
                // const float NearClipDistance = 1.0;
                // const float FarClipDistance = 1000.0;
                // const float ProjectionA = FarClipDistance / (FarClipDistance - NearClipDistance);
                // const float ProjectionB = (-FarClipDistance * NearClipDistance) / (FarClipDistance - NearClipDistance);

                float rand(vec2 co){
                    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
                }

                void main() {
                    // Hard=coded perspective_rh with fov_y_radians 45, aspect ratio 16/9, near 0.1, far 1000.0
                    // Can't yet bind camera projection in a fullscreen pass node and that may not make sense
                    mat4 Proj;
                    Proj[0] = vec4(1.357995, 0.0, 0.0, 0.0);
                    Proj[1] = vec4(0.0, 2.4142134, 0.0, 0.0);
                    Proj[2] = vec4(0.0, 0.0, -1.001001, -1.0);
                    Proj[3] = vec4(0.0, 0.0, -1.001001, 0.0);

                    mat4 InvProj;
                    InvProj[0] = vec4(0.7363797, 0.0, -0.0, 0.0);
                    InvProj[1] = vec4(0.0, 0.41421357, 0.0, -0.0);
                    InvProj[2] = vec4(-0.0, 0.0, -0.0, -0.99899995);
                    InvProj[3] = vec4(0.0, -0.0, -1.0, 1.0);

                    // TODO: For the 4x4 noise texture sampling
                    // tile noise texture over screen, based on screen dimensions divided by noise size
                    // const ivec2 size = textureSize(sampler2D(normal_texture, normal_texture_sampler), 0);
                    // const vec2 noiseScale = vec2(size) / 4.0;

                    // Calculate the fragment position from the depth texture
                    float depth = texture(sampler2D(depth_texture, depth_texture_sampler), v_Uv).x;
                    vec3 frag_ndc = vec3(v_Uv * 2.0 - 1.0, depth);
                    // FIXME: I feel like I should be multiplying the frag_ndc by the frag_clip.w. If frag_ndc.z at the far plane
                    // should be 1 and frag_ndc.z = frag_clip.z / frag_clip.w and frag_clip.w is -frag_view.z for perspective_rh projection
                    // then the below two lines should be correct, but they just make everything white.
                    // vec4 frag_clip = vec4(frag_ndc * -1000.0, -1000.0);
                    // vec4 frag_view = InvProj * frag_clip;
                    vec3 frag_view = mat3(InvProj) * frag_ndc;
                    vec3 normal = (texture(sampler2D(normal_texture, normal_texture_sampler), v_Uv).xyz - 0.5) * 2.0;
                    vec3 randomVec = normalize(vec3(rand(v_Uv), 0.0, rand(v_Uv + 1234.0)));
                    // TODO: Bind a 4x4 noise texture
                    // vec3 randomVec = texture(sampler2D(noise_texture, noise_texture_sampler), v_Uv * noiseScale).xyz;

                    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
                    vec3 bitangent = cross(normal, tangent);
                    mat3 TBN = mat3(tangent, bitangent, normal);

                    float occlusion = 0.0;
                    for (int i = 0; i < KERNEL_SIZE; ++i) {
                        vec3 sample_view = TBN * vec3(kernel[i][0], kernel[i][1], kernel[i][2]);
                        sample_view = frag_view.xyz + sample_view * RADIUS;

                        vec4 offset_view = vec4(sample_view, 1.0);
                        vec4 offset_clip = Proj * offset_view; // from view to clip space
                        vec3 offset_ndc = offset_clip.xyz / offset_clip.w; // perspective divide
                        vec2 uv = offset_ndc.xy * 0.5 + 0.5;
                        // offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 to 1.0

                        float sampleDepth = texture(sampler2D(depth_texture, depth_texture_sampler), uv).x;
                        // sampleDepth = dot(vec3(sampleDepth), InvProj[2].xyz);

                        // float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(frag_view.z - sampleDepth));
                        // occlusion += (sampleDepth >= sample_view.z + BIAS ? 1.0 : 0.0) * rangeCheck;
                        float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(frag_ndc.z - sampleDepth));
                        occlusion += (sampleDepth >= offset_ndc.z + BIAS ? 1.0 : 0.0) * rangeCheck;
                    }
                    occlusion = 1.0 - (occlusion / KERNEL_SIZE);

                    o_Target = occlusion;
                }
                ",
            ))),
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
) {
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
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex, // FIXME: Fix the shaders!
                "#version 450

                layout(location = 0) in vec3 Vertex_Position;
                layout(location = 1) in vec3 Vertex_Normal;
                
                layout(location = 0) out vec3 v_WorldNormal;
                
                layout(set = 0, binding = 0) uniform CameraViewProj {
                    mat4 ViewProj;
                };
                
                layout(set = 2, binding = 0) uniform Transform {
                    mat4 Model;
                };
                
                void main() {
                    v_WorldNormal = mat3(Model) * Vertex_Normal;
                    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
                }
                ",
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                layout(location = 0) in vec3 v_WorldNormal;

                layout(location = 0) out vec4 o_Target;

                void main() {
                    o_Target = vec4(v_WorldNormal, 0.0);
                }
                ",
            ))),
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
) {
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
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex, // FIXME: Fix the shaders!
                "#version 450

                layout(location = 0) in vec3 Vertex_Position;
                layout(location = 1) in vec3 Vertex_Normal;
                
                layout(location = 0) out vec3 v_WorldNormal;
                
                layout(set = 0, binding = 0) uniform CameraViewProj {
                    mat4 ViewProj;
                };
                
                layout(set = 2, binding = 0) uniform Transform {
                    mat4 Model;
                };
                
                void main() {
                    v_WorldNormal = mat3(Model) * Vertex_Normal;
                    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
                }
                ",
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                layout(location = 0) in vec3 v_WorldNormal;

                layout(location = 0) out vec4 o_Target;

                void main() {
                    o_Target = vec4(v_WorldNormal, 0.0);
                }
                ",
            ))),
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
) {
    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![
            ColorTargetState {
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
            },
        ],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                fullscreen_pass_node::shaders::VERTEX_SHADER,
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                layout(location = 0) in vec2 v_Uv;

                layout(set = 0, binding = 0) uniform texture2D depth_texture;
                layout(set = 0, binding = 1) uniform sampler depth_texture_sampler;

                layout(location = 0) out vec4 o_Target;

                void main() {
                    o_Target = vec4(vec3(texture(sampler2D(depth_texture, depth_texture_sampler), v_Uv).r), 1.0);
                }
                ",
            ))),
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
                usage: TextureUsage::OUTPUT_ATTACHMENT,
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
) {
    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![
            ColorTargetState {
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
            },
        ],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                fullscreen_pass_node::shaders::VERTEX_SHADER,
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                layout(location = 0) in vec2 v_Uv;

                layout(set = 0, binding = 0) uniform texture2D normal_texture;
                layout(set = 0, binding = 1) uniform sampler normal_texture_sampler;

                layout(location = 0) out vec4 o_Target;

                void main() {
                    o_Target = vec4((texture(sampler2D(normal_texture, normal_texture_sampler), v_Uv).rgb - 0.5) * 2.0, 1.0);
                }
                ",
            ))),
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
                usage: TextureUsage::OUTPUT_ATTACHMENT,
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
) {
    let pipeline_descriptor = PipelineDescriptor {
        depth_stencil: None,
        color_target_states: vec![
            ColorTargetState {
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
            },
        ],
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                fullscreen_pass_node::shaders::VERTEX_SHADER,
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                layout(location = 0) in vec2 v_Uv;

                layout(set = 0, binding = 0) uniform texture2D occlusion_texture;
                layout(set = 0, binding = 1) uniform sampler occlusion_texture_sampler;

                layout(location = 0) out vec4 o_Target;

                void main() {
                    o_Target = vec4(vec3(texture(sampler2D(occlusion_texture, occlusion_texture_sampler), v_Uv).r), 1.0);
                }
                ",
            ))),
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
                usage: TextureUsage::OUTPUT_ATTACHMENT,
            },
            None,
            None,
        ),
    );
    if msaa.samples > 1 {
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

fn setup_render_graph(
    render_graph: &mut RenderGraph,
    pipelines: &mut Assets<PipelineDescriptor>,
    shaders: &mut Assets<Shader>,
    msaa: &Msaa,
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
    // set_up_depth_render_pass(shaders, pipelines, msaa, render_graph);
    // // Render the normal texture
    // set_up_normal_render_pass(shaders, pipelines, msaa, render_graph);

    // Set up SSAO pass pipeline
    set_up_ssao_pass(shaders, pipelines, msaa, render_graph);
    // Render the occlusion texture after the ssao pass
    set_up_occlusion_render_pass(shaders, pipelines, msaa, render_graph);

    // // Set up blur X pass pipeline
    // set_up_blur_x_pass(shaders, pipelines, msaa, render_graph);
    // // Set up blur Y pass pipeline
    // set_up_blur_y_pass(shaders, pipelines, msaa, render_graph);

    // // Set up modified main pass
    // set_up_modified_main_pass(shaders, pipelines, msaa, render_graph);
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
