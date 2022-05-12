use bevy::{
    core_pipeline::{self, AlphaMask3d, Opaque3d, Transparent3d},
    input::mouse::MouseMotion,
    pbr::{DebugBounds, NotShadowCaster},
    prelude::*,
    render::{
        camera::{ActiveCamera, Camera3d, CameraTypePlugin, RenderTarget},
        render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext, SlotValue},
        render_phase::RenderPhase,
        renderer::RenderContext,
        RenderApp, RenderStage,
    },
    window::{CreateWindow, WindowId},
};

fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(SecondWindowCameraPlugin)
        .add_startup_system(setup)
        // .add_system_to_stage(CoreStage::PreUpdate, toggle_camera)
        .add_system(camera_controller)
        .run();
}

#[derive(Component, Default)]
struct SecondWindowCamera3d;

struct Cameras {
    cameras: [Entity; 2],
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut create_window_events: EventWriter<CreateWindow>,
    mut active_camera: ResMut<ActiveCamera<Camera3d>>,
) {
    // cube
    let cube_mesh_handle = meshes.add(Mesh::from(shape::Cube { size: 0.1 }));
    let rgb_handles = [
        materials.add(Color::RED.into()),
        materials.add(Color::GREEN.into()),
        materials.add(Color::BLUE.into()),
    ];
    commands
        .spawn_bundle(PbrBundle {
            mesh: cube_mesh_handle.clone(),
            material: materials.add(Color::BLACK.into()),
            ..default()
        })
        .insert(DebugBounds);
    commands
        .spawn_bundle(PbrBundle {
            mesh: cube_mesh_handle.clone(),
            material: rgb_handles[0].clone(),
            transform: Transform::from_xyz(2.0, 0.0, 0.0),
            ..default()
        })
        .insert(DebugBounds);
    commands
        .spawn_bundle(PbrBundle {
            mesh: cube_mesh_handle.clone(),
            material: rgb_handles[1].clone(),
            transform: Transform::from_xyz(0.0, 2.0, 0.0),
            ..default()
        })
        .insert(DebugBounds);
    commands
        .spawn_bundle(PbrBundle {
            mesh: cube_mesh_handle.clone(),
            material: rgb_handles[2].clone(),
            transform: Transform::from_xyz(0.0, 0.0, 2.0),
            ..default()
        })
        .insert(DebugBounds);
    // light
    commands.spawn_bundle(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // second window
    let window_id = WindowId::new();

    // sends out a "CreateWindow" event, which will be received by the windowing backend
    create_window_events.send(CreateWindow {
        id: window_id,
        descriptor: WindowDescriptor {
            width: 800.,
            height: 600.,
            title: "Second window".to_string(),
            ..default()
        },
    });

    // second window camera
    commands.spawn_bundle(PerspectiveCameraBundle {
        camera: Camera {
            target: RenderTarget::Window(window_id),
            ..default()
        },
        transform: Transform::from_xyz(10.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
        marker: SecondWindowCamera3d,
        ..PerspectiveCameraBundle::new()
    });

    // cameras
    // let cameras = [
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            // transform: Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        })
        .insert_bundle((DebugBounds, CameraController::default()))
        .with_children(|builder| {
            builder
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Icosphere {
                        radius: 0.05,
                        ..default()
                    })),
                    material: materials.add(Color::RED.into()),
                    ..default()
                })
                .insert(NotShadowCaster);
        });
    // .id(),
    //     commands
    //         .spawn_bundle(PerspectiveCameraBundle {
    //             transform: Transform::from_xyz(2.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
    //             ..default()
    //         })
    //         // .insert_bundle((DebugBounds, CameraController::default()))
    //         .with_children(|builder| {
    //             builder
    //                 .spawn_bundle(PbrBundle {
    //                     mesh: meshes.add(Mesh::from(shape::Icosphere {
    //                         radius: 0.05,
    //                         ..default()
    //                     })),
    //                     material: materials.add(Color::GREEN.into()),
    //                     ..default()
    //                 })
    //                 .insert(NotShadowCaster);
    //         })
    //         .id(),
    // ];
    // active_camera.set(cameras[0]);
    // commands.insert_resource(Cameras { cameras });
}

fn toggle_camera(
    key_input: Res<Input<KeyCode>>,
    cameras: Res<Cameras>,
    mut active_camera: ResMut<ActiveCamera<Camera3d>>,
) {
    if key_input.just_pressed(KeyCode::C) {
        let next_camera = match active_camera.get() {
            Some(camera) if camera == cameras.cameras[0] => cameras.cameras[1],
            Some(camera) if camera == cameras.cameras[1] => cameras.cameras[0],
            _ => return,
        };
        active_camera.set(next_camera);
    }
}

struct SecondWindowCameraPlugin;
impl Plugin for SecondWindowCameraPlugin {
    fn build(&self, app: &mut App) {
        // adds the `ActiveCamera<SecondWindowCamera3d>` resource and extracts the camera into the render world
        app.add_plugin(CameraTypePlugin::<SecondWindowCamera3d>::default());

        let render_app = app.sub_app_mut(RenderApp);

        // add `RenderPhase<Opaque3d>`, `RenderPhase<AlphaMask3d>` and `RenderPhase<Transparent3d>` camera phases
        render_app.add_system_to_stage(RenderStage::Extract, extract_second_camera_phases);

        // add a render graph node that executes the 3d subgraph
        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        let second_window_node = render_graph.add_node("second_window_cam", SecondWindowDriverNode);
        render_graph
            .add_node_edge(
                core_pipeline::node::MAIN_PASS_DEPENDENCIES,
                second_window_node,
            )
            .unwrap();
        render_graph
            .add_node_edge(core_pipeline::node::CLEAR_PASS_DRIVER, second_window_node)
            .unwrap();
    }
}

struct SecondWindowDriverNode;
impl render_graph::Node for SecondWindowDriverNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        _: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if let Some(camera) = world.resource::<ActiveCamera<SecondWindowCamera3d>>().get() {
            graph.run_sub_graph(
                core_pipeline::draw_3d_graph::NAME,
                vec![SlotValue::Entity(camera)],
            )?;
        }

        Ok(())
    }
}

fn extract_second_camera_phases(
    mut commands: Commands,
    active: Res<ActiveCamera<SecondWindowCamera3d>>,
) {
    if let Some(entity) = active.get() {
        commands.get_or_spawn(entity).insert_bundle((
            RenderPhase::<Opaque3d>::default(),
            RenderPhase::<AlphaMask3d>::default(),
            RenderPhase::<Transparent3d>::default(),
        ));
    }
}

#[derive(Clone, Component)]
struct CameraController {
    pub enabled: bool,
    pub initialized: bool,
    pub sensitivity: f32,
    pub key_forward: KeyCode,
    pub key_back: KeyCode,
    pub key_left: KeyCode,
    pub key_right: KeyCode,
    pub key_up: KeyCode,
    pub key_down: KeyCode,
    pub key_run: KeyCode,
    pub key_enable_mouse: MouseButton,
    pub walk_speed: f32,
    pub run_speed: f32,
    pub friction: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub velocity: Vec3,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            enabled: true,
            initialized: false,
            sensitivity: 0.5,
            key_forward: KeyCode::W,
            key_back: KeyCode::S,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_up: KeyCode::E,
            key_down: KeyCode::Q,
            key_run: KeyCode::LShift,
            key_enable_mouse: MouseButton::Left,
            walk_speed: 5.0,
            run_speed: 15.0,
            friction: 0.5,
            pitch: 0.0,
            yaw: 0.0,
            velocity: Vec3::ZERO,
        }
    }
}

fn camera_controller(
    time: Res<Time>,
    mut mouse_events: EventReader<MouseMotion>,
    mouse_button_input: Res<Input<MouseButton>>,
    key_input: Res<Input<KeyCode>>,
    mut query: Query<(&mut Transform, &mut CameraController), With<Camera>>,
    active_camera: Res<ActiveCamera<Camera3d>>,
) {
    let dt = time.delta_seconds();

    let active_camera = match active_camera.get() {
        Some(active_camera) => active_camera,
        None => return,
    };
    if let Ok((mut transform, mut options)) = query.get_mut(active_camera) {
        if !options.initialized {
            let (yaw, pitch, _roll) = transform.rotation.to_euler(EulerRot::YXZ);
            options.yaw = yaw;
            options.pitch = pitch;
            options.initialized = true;
        }
        if !options.enabled {
            return;
        }

        // Handle key input
        let mut axis_input = Vec3::ZERO;
        if key_input.pressed(options.key_forward) {
            axis_input.z += 1.0;
        }
        if key_input.pressed(options.key_back) {
            axis_input.z -= 1.0;
        }
        if key_input.pressed(options.key_right) {
            axis_input.x += 1.0;
        }
        if key_input.pressed(options.key_left) {
            axis_input.x -= 1.0;
        }
        if key_input.pressed(options.key_up) {
            axis_input.y += 1.0;
        }
        if key_input.pressed(options.key_down) {
            axis_input.y -= 1.0;
        }

        // Apply movement update
        if axis_input != Vec3::ZERO {
            let max_speed = if key_input.pressed(options.key_run) {
                options.run_speed
            } else {
                options.walk_speed
            };
            options.velocity = axis_input.normalize() * max_speed;
        } else {
            let friction = options.friction.clamp(0.0, 1.0);
            options.velocity *= 1.0 - friction;
            if options.velocity.length_squared() < 1e-6 {
                options.velocity = Vec3::ZERO;
            }
        }
        let forward = transform.forward();
        let right = transform.right();
        transform.translation += options.velocity.x * dt * right
            + options.velocity.y * dt * Vec3::Y
            + options.velocity.z * dt * forward;

        // Handle mouse input
        let mut mouse_delta = Vec2::ZERO;
        if mouse_button_input.pressed(options.key_enable_mouse) {
            for mouse_event in mouse_events.iter() {
                mouse_delta += mouse_event.delta;
            }
        }

        if mouse_delta != Vec2::ZERO {
            // Apply look update
            let (pitch, yaw) = (
                (options.pitch - mouse_delta.y * 0.5 * options.sensitivity * dt).clamp(
                    -0.99 * std::f32::consts::FRAC_PI_2,
                    0.99 * std::f32::consts::FRAC_PI_2,
                ),
                options.yaw - mouse_delta.x * options.sensitivity * dt,
            );
            transform.rotation = Quat::from_euler(EulerRot::ZYX, 0.0, yaw, pitch);
            options.pitch = pitch;
            options.yaw = yaw;
        }
    }
}
