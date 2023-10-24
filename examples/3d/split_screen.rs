//! Renders two cameras to the same window to accomplish "split screen".

use std::{f32::consts::PI, fmt};

use bevy::{core_pipeline::clear_color::ClearColorConfig, prelude::*, render::camera::Viewport};
use bevy_internal::{
    input::mouse::MouseMotion,
    pbr::{check_light_mesh_visibility, CascadesVisibleEntities},
    render::view::VisibleEntities,
    utils::HashSet,
    window::{CursorGrabMode, WindowResolution},
};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(1920.0, 1080.0)
                        .with_scale_factor_override(1.0),
                    resizable: false,
                    ..default()
                }),
                ..default()
            }),
            CameraControllerPlugin,
        ))
        .insert_resource(GizmoConfig {
            frustum: FrustumGizmoConfig {
                draw_all: true,
                default_color: Some(Color::WHITE),
            },
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            PostUpdate,
            (
                apply_deferred.after(check_light_mesh_visibility),
                update_colors,
            )
                .chain(),
        )
        .run();
}

#[derive(Resource)]
struct CubeMaterials {
    cube: Handle<Mesh>,
    red: Handle<StandardMaterial>,
    green: Handle<StandardMaterial>,
    blue: Handle<StandardMaterial>,
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cube = meshes.add(Mesh::from(shape::Cube::new(10.0)));
    let red = materials.add(Color::rgba(1.0, 0.0, 0.0, 0.75).into());
    let green = materials.add(Color::rgba(0.0, 1.0, 0.0, 0.2).into());
    let blue = materials.add(Color::rgba(0.0, 0.0, 1.0, 0.2).into());

    let mut batch = Vec::with_capacity(32 * 32 * 32);
    for z in 0..32 {
        for y in 0..32 {
            for x in 0..32 {
                batch.push(PbrBundle {
                    mesh: cube.clone_weak(),
                    material: green.clone_weak(),
                    transform: Transform::from_xyz(
                        (32 * x) as f32,
                        (32 * y) as f32,
                        (-32 * z) as f32,
                    ),
                    ..default()
                });
            }
        }
    }
    commands.spawn_batch(batch);
    commands.insert_resource(CubeMaterials {
        cube,
        red,
        green,
        blue,
    });

    // Light
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, 1.0, -PI / 4.)),
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        ..default()
    });

    // Main Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(30.0 * Vec3::new(14.5, 14.5, 0.0)),
            camera: Camera {
                viewport: Some(Viewport {
                    physical_size: 2 * UVec2::new(192, 108),
                    ..default()
                }),
                ..default()
            },
            ..default()
        },
        MainCamera,
    ));

    let camera_controller = CameraController::default();
    // Display the controls of the scene viewer
    info!("{}", camera_controller);

    // Observer Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(30.0 * Vec3::new(-30.0, 14.5, -14.5))
                .with_rotation(Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2)),
            camera: Camera {
                // Renders the right camera after the left camera, which has a default priority of 0
                order: 1,
                ..default()
            },
            camera_3d: Camera3d {
                // don't clear on the second camera because the first camera already cleared the window
                clear_color: ClearColorConfig::None,
                ..default()
            },
            // projection: OrthographicProjection::default().into(),
            ..default()
        },
        ObserverCamera,
        camera_controller,
    ));
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct ObserverCamera;

fn update_colors(
    main_camera: Query<(Entity, &VisibleEntities), With<MainCamera>>,
    light: Query<&CascadesVisibleEntities>,
    cube_materials: Res<CubeMaterials>,
    mut meshes: Query<(Entity, &mut Handle<StandardMaterial>)>,
    mut logged: Local<bool>,
) {
    let mut culled = meshes
        .iter()
        .map(|(entity, _)| entity)
        .collect::<HashSet<Entity>>();
    let total = culled.len();
    let (main_camera_entity, main_camera_visible) = main_camera.single();
    let mut visible = main_camera_visible
        .entities
        .iter()
        .copied()
        .collect::<HashSet<Entity>>();
    culled = culled.difference(&visible).copied().collect::<HashSet<_>>();

    let mut casters = HashSet::new();
    if let Some(cascade_visible_entities) = light.single().entities.get(&main_camera_entity) {
        casters = cascade_visible_entities
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect::<HashSet<_>>();
        casters = casters
            .difference(&visible)
            .copied()
            .collect::<HashSet<_>>();
        culled = culled.difference(&casters).copied().collect::<HashSet<_>>();
    }

    if !*logged {
        info!(
            "{} visible, {} casters, {} culled, {} total",
            visible.len(),
            casters.len(),
            culled.len(),
            total
        );
        *logged = true;
    }

    for entity in culled.drain() {
        meshes
            .get_mut(entity)
            .map(|(_, mut handle)| *handle = cube_materials.red.clone_weak())
            .ok();
    }
    for entity in casters.drain() {
        meshes
            .get_mut(entity)
            .map(|(_, mut handle)| *handle = cube_materials.green.clone_weak())
            .ok();
    }
    for entity in visible.drain() {
        meshes
            .get_mut(entity)
            .map(|(_, mut handle)| *handle = cube_materials.blue.clone_weak())
            .ok();
    }
}

/// Based on Valorant's default sensitivity, not entirely sure why it is exactly 1.0 / 180.0,
/// but I'm guessing it is a misunderstanding between degrees/radians and then sticking with
/// it because it felt nice.
pub const RADIANS_PER_DOT: f32 = 1.0 / 180.0;

#[derive(Component)]
pub struct CameraController {
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
    pub mouse_key_enable_mouse: MouseButton,
    pub keyboard_key_enable_mouse: KeyCode,
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
            sensitivity: 1.0,
            key_forward: KeyCode::W,
            key_back: KeyCode::S,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_up: KeyCode::E,
            key_down: KeyCode::Q,
            key_run: KeyCode::ShiftLeft,
            mouse_key_enable_mouse: MouseButton::Left,
            keyboard_key_enable_mouse: KeyCode::M,
            walk_speed: 50.0,
            run_speed: 150.0,
            friction: 0.5,
            pitch: 0.0,
            yaw: 0.0,
            velocity: Vec3::ZERO,
        }
    }
}

impl fmt::Display for CameraController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "
Freecam Controls:
    MOUSE\t- Move camera orientation
    {:?}/{:?}\t- Enable mouse movement
    {:?}{:?}\t- forward/backward
    {:?}{:?}\t- strafe left/right
    {:?}\t- 'run'
    {:?}\t- up
    {:?}\t- down",
            self.mouse_key_enable_mouse,
            self.keyboard_key_enable_mouse,
            self.key_forward,
            self.key_back,
            self.key_left,
            self.key_right,
            self.key_run,
            self.key_up,
            self.key_down
        )
    }
}

pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, camera_controller);
    }
}

fn camera_controller(
    time: Res<Time>,
    mut windows: Query<&mut Window>,
    mut mouse_events: EventReader<MouseMotion>,
    mouse_button_input: Res<Input<MouseButton>>,
    key_input: Res<Input<KeyCode>>,
    mut move_toggled: Local<bool>,
    mut query: Query<(&mut Transform, &mut CameraController), With<Camera>>,
) {
    let dt = time.delta_seconds();

    if let Ok((mut transform, mut options)) = query.get_single_mut() {
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
        if key_input.just_pressed(options.keyboard_key_enable_mouse) {
            *move_toggled = !*move_toggled;
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
        if mouse_button_input.pressed(options.mouse_key_enable_mouse) || *move_toggled {
            for mut window in &mut windows {
                if !window.focused {
                    continue;
                }

                window.cursor.grab_mode = CursorGrabMode::Locked;
                window.cursor.visible = false;
            }

            for mouse_event in mouse_events.read() {
                mouse_delta += mouse_event.delta;
            }
        }
        if mouse_button_input.just_released(options.mouse_key_enable_mouse) {
            for mut window in &mut windows {
                window.cursor.grab_mode = CursorGrabMode::None;
                window.cursor.visible = true;
            }
        }

        if mouse_delta != Vec2::ZERO {
            // Apply look update
            options.pitch = (options.pitch - mouse_delta.y * RADIANS_PER_DOT * options.sensitivity)
                .clamp(-PI / 2., PI / 2.);
            options.yaw -= mouse_delta.x * RADIANS_PER_DOT * options.sensitivity;
            transform.rotation = Quat::from_euler(EulerRot::ZYX, 0.0, options.yaw, options.pitch);
        }
    }
}
