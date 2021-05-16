use bevy::{
    asset::AssetPlugin,
    core::CorePlugin,
    diagnostic::DiagnosticsPlugin,
    input::InputPlugin,
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
    pub const CAMERA_INVERSE_PROJECTION: &str = "CameraInverseProjection";
    pub const WINDOW_TEXTURE_SIZE: &str = "WindowTextureSize";
    // Nodes
    pub const TRANSFORM: &str = "transform";
    pub const DEPTH_NORMAL_PRE_PASS: &str = "depth_normal_pre_pass_node";
    pub const DEPTH_RENDER_PASS: &str = "depth_render_pass";
    pub const SSAO_PASS: &str = "ssao_pass_node";
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
    let mut app = App::build();

    app
        // Pbr
        // .insert_resource(AmbientLight {
        //     color: Color::WHITE,
        //     brightness: 1.0 / 5.0f32,
        // })
        .insert_resource(Msaa { samples: 4 })
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
        .add_system(update_camera_inverse_projection.system())
        .add_system(rotator_system.system())
        .add_startup_system(
            bevy_mod_debugdump::print_render_graph
                .system()
                .label("debugdump")
                .after("setup"),
        )
        .run();
}

#[derive(Debug, RenderResources)]
pub struct CameraInverseProjection {
    pub inverse_projection: Mat4,
}

fn update_camera_inverse_projection(
    mut commands: Commands,
    to_init: Query<
        (Entity, &Camera),
        (
            With<PerspectiveProjection>,
            Without<CameraInverseProjection>,
        ),
    >,
    mut to_update: Query<
        (&Camera, &mut CameraInverseProjection),
        (With<PerspectiveProjection>, Changed<Camera>),
    >,
) {
    for (entity, camera) in to_init.iter() {
        // If the determinant is 0, then the matrix is not invertible
        debug_assert!(camera.projection_matrix.determinant().abs() > 1e-5);
        commands.entity(entity).insert(CameraInverseProjection {
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
    render_graph.add_system_node(
        node::CAMERA_INVERSE_PROJECTION,
        RenderResourcesNode::<CameraInverseProjection>::new(true),
    );

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
                    v_ViewNormal = mat3(View) * mat3(Model) * Vertex_Normal;
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
                    o_Target = vec4(v_ViewNormal, 1.0);
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
                load: LoadOp::Clear(Color::rgb(0.1, 0.1, 0.1)),
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
                ShaderStage::Vertex, // FIXME: Fix the shaders!
                fullscreen_pass_node::shaders::VERTEX_SHADER, // Provides v_Uv
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                "#version 450

                // FIXME: Move to a uniform!
                const vec3 kernel[32] = // precalculated hemisphere kernel (low discrepancy noiser)
                {
                    vec3(-0.668154f, -0.084296f, 0.219458f),
                    vec3(-0.092521f,  0.141327f, 0.505343f),
                    vec3(-0.041960f,  0.700333f, 0.365754f),
                    vec3( 0.722389f, -0.015338f, 0.084357f),
                    vec3(-0.815016f,  0.253065f, 0.465702f),
                    vec3( 0.018993f, -0.397084f, 0.136878f),
                    vec3( 0.617953f, -0.234334f, 0.513754f),
                    vec3(-0.281008f, -0.697906f, 0.240010f),
                    vec3( 0.303332f, -0.443484f, 0.588136f),
                    vec3(-0.477513f,  0.559972f, 0.310942f),
                    vec3( 0.307240f,  0.076276f, 0.324207f),
                    vec3(-0.404343f, -0.615461f, 0.098425f),
                    vec3( 0.152483f, -0.326314f, 0.399277f),
                    vec3( 0.435708f,  0.630501f, 0.169620f),
                    vec3( 0.878907f,  0.179609f, 0.266964f),
                    vec3(-0.049752f, -0.232228f, 0.264012f),
                    vec3( 0.537254f, -0.047783f, 0.693834f),
                    vec3( 0.001000f,  0.177300f, 0.096643f),
                    vec3( 0.626400f,  0.524401f, 0.492467f),
                    vec3(-0.708714f, -0.223893f, 0.182458f),
                    vec3(-0.106760f,  0.020965f, 0.451976f),
                    vec3(-0.285181f, -0.388014f, 0.241756f),
                    vec3( 0.241154f, -0.174978f, 0.574671f),
                    vec3(-0.405747f,  0.080275f, 0.055816f),
                    vec3( 0.079375f,  0.289697f, 0.348373f),
                    vec3( 0.298047f, -0.309351f, 0.114787f),
                    vec3(-0.616434f, -0.117369f, 0.475924f),
                    vec3(-0.035249f,  0.134591f, 0.840251f),
                    vec3( 0.175849f,  0.971033f, 0.211778f),
                    vec3( 0.024805f,  0.348056f, 0.240006f),
                    vec3(-0.267123f,  0.204885f, 0.688595f),
                    vec3(-0.077639f, -0.753205f, 0.070938f)
                };

                layout(location = 0) in vec2 v_Uv;

                layout(location = 0) out float o_Target;

                layout(set = 0, binding = 0) uniform CameraProjection {
                    mat4 Projection;
                };
                layout(set = 0, binding = 1) uniform CameraInverseProjection {
                    mat4 InverseProjection;
                };

                layout(set = 2, binding = 0) uniform texture2D depth_texture;
                layout(set = 2, binding = 1) uniform sampler depth_texture_sampler;
                layout(set = 2, binding = 2) uniform texture2D normal_texture;
                layout(set = 2, binding = 3) uniform sampler normal_texture_sampler;
                // layout(set = 2, binding = 4) uniform texture2D noise_texture;
                // layout(set = 2, binding = 5) uniform sampler noise_texture_sampler;

                // tile noise texture over screen, based on screen dimensions divided by noise size
                const ivec2 size = textureSize(sampler2D(normal_texture, normal_texture_sampler), 0);
                const vec2 noiseScale = vec2(size) / 4.0;
                // FIXME: Make this into a uniform
                const float RADIUS = 0.5;
                const float BIAS = 0.025;
                const int KERNEL_SIZE = 32;

                const float NearClipDistance = 1.0;
                const float FarClipDistance = 1000.0;
                const float ProjectionA = FarClipDistance / (FarClipDistance - NearClipDistance);
                const float ProjectionB = (-FarClipDistance * NearClipDistance) / (FarClipDistance - NearClipDistance);

                void main() {
                    // Calculate the fragment position from the depth texture
                    vec3 viewRay =
                    float depth = texture(sampler2D(depth_texture, depth_texture_sampler), v_Uv).x;
                    float linearDepth = ProjectionB / (depth - ProjectionA);
                    vec3 fragPos = vec3(0.0);
                    vec3 normal = texture(sampler2D(normal_texture, normal_texture_sampler), v_Uv).xyz;
                    vec3 randomVec = texture(sampler2D(noise_texture, noise_texture_sampler), v_Uv * noiseScale).xyz;

                    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
                    vec3 bitangent = cross(normal, tangent);
                    mat3 TBN = mat3(tangent, bitangent, normal);

                    float occlusion = 0.0;
                    for (int i = 0; i < KERNEL_SIZE; ++i) {
                        vec3 samplePos = TBN * kernel[i];
                        samplePos = fragPos + samplePos * RADIUS;

                        vec4 offset = vec4(samplePos, 1.0);
                        offset = Projection * offset; // from view to clip space
                        offset.xyz /= offset.w; // perspective divide
                        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 to 1.0

                        float sampleDepth = texture(sampler2D(depth_texture, depth_texture_sampler), offset.xy).x;

                        float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(fragPos.z - sampleDepth));
                        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;  
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
            attachment: TextureAttachment::Input(node::SSAO_A_TEXTURE.to_string()),
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

    render_graph
        .add_node_edge(node::CAMERA_INVERSE_PROJECTION, node::SSAO_PASS)
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
            node::NORMAL_TEXTURE,
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
            node::DEPTH_TEXTURE,
        )
        .unwrap();
    render_graph
        .add_slot_edge(
            node::SSAO_A_TEXTURE,
            WindowTextureNode::OUT_TEXTURE,
            node::SSAO_PASS,
            "ssao_output_texture",
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

fn set_up_depth_display_pass(
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

fn setup_render_graph(
    render_graph: &mut RenderGraph,
    pipelines: &mut Assets<PipelineDescriptor>,
    shaders: &mut Assets<Shader>,
    msaa: &Msaa,
) {
    // Set up the additional textures
    // FIXME: Make a new depth texture with OUTPUT_ATTACHMENT | SAMPLED
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

    // Set up depth normal pre-pass pipeline
    set_up_depth_normal_pre_pass(msaa, render_graph);

    // Render the depth texture
    set_up_depth_display_pass(shaders, pipelines, msaa, render_graph);

    // // Set up SSAO pass pipeline
    // set_up_ssao_pass(shaders, pipelines, msaa, render_graph);
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
