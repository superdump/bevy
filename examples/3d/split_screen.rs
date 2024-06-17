//! Renders two cameras to the same window to accomplish "split screen".

use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::pbr::{check_light_mesh_visibility, CascadesVisibleEntities};
use bevy::prelude::*;
use bevy::render::camera::Viewport;
use bevy::render::view::{VisibleEntities, WithMesh};
use bevy::utils::HashSet;
use bevy::window::{CursorGrabMode, WindowResized, WindowResolution};
use std::{f32::consts::*, fmt};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(1920.0, 1080.0)
                        .with_scale_factor_override(1.0),
                    ..default()
                }),
                ..default()
            }),
            CameraControllerPlugin,
        ))
        // .insert_resource(GizmoConfig {
        //         frustum: FrustumGizmoConfig {
        //             draw_all: true,
        //         default_color: Some(Color::WHITE),
        //     },
        //     ..default()
        // })
        .add_systems(Startup, setup)
        .add_systems(Update, (set_camera_viewports, button_system))
        .add_systems(PostUpdate, update_colors.after(check_light_mesh_visibility))
        .run();
}

#[derive(Resource)]
struct CubeMaterials {
    cube: Handle<Mesh>,
    red: Handle<StandardMaterial>,
    green: Handle<StandardMaterial>,
    blue: Handle<StandardMaterial>,
}

#[derive(Component)]
struct MainCamera;

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut gizmo_config: ResMut<GizmoConfigStore>,
) {
    let cube = meshes.add(Mesh::from(Cuboid {
        half_size: Vec3::splat(10.0),
    }));
    let red = materials.add(StandardMaterial::from_color(Color::srgba(
        1.0, 0.0, 0.0, 0.75,
    )));
    let green = materials.add(StandardMaterial::from_color(Color::srgba(
        0.0, 1.0, 0.0, 0.2,
    )));
    let blue = materials.add(StandardMaterial::from_color(Color::srgba(
        0.0, 0.0, 1.0, 0.2,
    )));

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
    commands.spawn((
        DirectionalLightBundle {
            transform: Transform::from_rotation(Quat::from_euler(
                EulerRot::ZYX,
                0.0,
                1.0,
                -PI / 4.,
            )),
            directional_light: DirectionalLight {
                shadows_enabled: true,
                ..default()
            },
            ..default()
        },
        ShowLightGizmo::default(),
    ));

    // Show light gizmos
    let (_, light_config) = gizmo_config.config_mut::<LightGizmoConfigGroup>();
    // light_config.draw_all = true;
    light_config.color = LightGizmoColor::MatchLightColor;

    // Cameras and their dedicated UI
    for (index, (camera_name, camera_transform)) in [
        (
            "Player 1",
            Transform::from_translation(30.0 * Vec3::new(14.5, 14.5, 0.0)),
        ),
        (
            "Player 2",
            Transform::from_translation(30.0 * Vec3::new(-30.0, 14.5, -14.5))
                .with_rotation(Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2)),
        ),
    ]
    .iter()
    .enumerate()
    {
        let camera = commands
            .spawn((
                Camera3dBundle {
                    transform: *camera_transform,
                    camera: Camera {
                        // Renders cameras with different priorities to prevent ambiguities
                        order: index as isize,
                        // Don't clear after the first camera because the first camera already cleared the entire window
                        clear_color: if index > 0 {
                            ClearColorConfig::None
                        } else {
                            ClearColorConfig::default()
                        },
                        ..default()
                    },
                    ..default()
                },
                CameraPosition {
                    pos: UVec2::new((index % 2) as u32, 0), //(index / 2) as u32),
                },
            ))
            .id();

        if index == 0 {
            commands
                .entity(camera)
                .insert((MainCamera, ShowLightGizmo::default()));
        } else {
            commands.entity(camera).insert(CameraController::default());
        }

        // Set up UI
        commands
            .spawn((
                TargetCamera(camera),
                NodeBundle {
                    style: Style {
                        width: Val::Percent(100.),
                        height: Val::Percent(100.),
                        padding: UiRect::all(Val::Px(20.)),
                        ..default()
                    },
                    ..default()
                },
            ))
            .with_children(|parent| {
                parent.spawn(TextBundle::from_section(*camera_name, TextStyle::default()));
                buttons_panel(parent);
            });
    }

    fn buttons_panel(parent: &mut ChildBuilder) {
        parent
            .spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    padding: UiRect::all(Val::Px(20.)),
                    ..default()
                },
                ..default()
            })
            .with_children(|parent| {
                rotate_button(parent, "<", Direction::Left);
                rotate_button(parent, ">", Direction::Right);
            });
    }

    fn rotate_button(parent: &mut ChildBuilder, caption: &str, direction: Direction) {
        parent
            .spawn((
                RotateCamera(direction),
                ButtonBundle {
                    style: Style {
                        width: Val::Px(40.),
                        height: Val::Px(40.),
                        border: UiRect::all(Val::Px(2.)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    border_color: Color::WHITE.into(),
                    image: UiImage::default().with_color(Color::srgb(0.25, 0.25, 0.25)),
                    ..default()
                },
            ))
            .with_children(|parent| {
                parent.spawn(TextBundle::from_section(caption, TextStyle::default()));
            });
    }
}

#[derive(Component)]
struct CameraPosition {
    pos: UVec2,
}

#[derive(Component)]
struct RotateCamera(Direction);

enum Direction {
    Left,
    Right,
}

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
        .get::<WithMesh>()
        .iter()
        .copied()
        .collect::<HashSet<Entity>>();
    culled = culled.difference(&visible).copied().collect::<HashSet<_>>();

    let mut casters = HashSet::new();
    if let Some(cascade_visible_entities) = light.single().entities.get(&main_camera_entity) {
        casters = cascade_visible_entities
            .iter()
            .flat_map(|v| v.get::<WithMesh>().iter().copied())
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

fn set_camera_viewports(
    windows: Query<&Window>,
    mut resize_events: EventReader<WindowResized>,
    mut query: Query<(&CameraPosition, &mut Camera)>,
) {
    // We need to dynamically resize the camera's viewports whenever the window size changes
    // so then each camera always takes up half the screen.
    // A resize_event is sent when the window is first created, allowing us to reuse this system for initial setup.
    for resize_event in resize_events.read() {
        let window = windows.get(resize_event.window).unwrap();
        let size = window.physical_size() / UVec2::new(2, 1);

        for (camera_position, mut camera) in &mut query {
            camera.viewport = Some(Viewport {
                physical_position: camera_position.pos * size,
                physical_size: size,
                ..default()
            });
        }
    }
}

#[allow(clippy::type_complexity)]
fn button_system(
    interaction_query: Query<
        (&Interaction, &TargetCamera, &RotateCamera),
        (Changed<Interaction>, With<Button>),
    >,
    mut camera_query: Query<&mut Transform, With<Camera>>,
) {
    for (interaction, target_camera, RotateCamera(direction)) in &interaction_query {
        if let Interaction::Pressed = *interaction {
            // Since TargetCamera propagates to the children, we can use it to find
            // which side of the screen the button is on.
            if let Ok(mut camera_transform) = camera_query.get_mut(target_camera.entity()) {
                let angle = match direction {
                    Direction::Left => -0.1,
                    Direction::Right => 0.1,
                };
                camera_transform.rotate_around(Vec3::ZERO, Quat::from_axis_angle(Vec3::Y, angle));
            }
        }
    }
}

pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_camera_controller);
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
    pub mouse_key_cursor_grab: MouseButton,
    pub keyboard_key_toggle_cursor_grab: KeyCode,
    pub walk_speed: f32,
    pub run_speed: f32,
    pub scroll_factor: f32,
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
            key_forward: KeyCode::KeyW,
            key_back: KeyCode::KeyS,
            key_left: KeyCode::KeyA,
            key_right: KeyCode::KeyD,
            key_up: KeyCode::KeyE,
            key_down: KeyCode::KeyQ,
            key_run: KeyCode::ShiftLeft,
            mouse_key_cursor_grab: MouseButton::Left,
            keyboard_key_toggle_cursor_grab: KeyCode::KeyM,
            walk_speed: 250.0,
            run_speed: 1000.0,
            scroll_factor: 0.1,
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
    Mouse\t- Move camera orientation
    Scroll\t- Adjust movement speed
    {:?}\t- Hold to grab cursor
    {:?}\t- Toggle cursor grab
    {:?} & {:?}\t- Fly forward & backwards
    {:?} & {:?}\t- Fly sideways left & right
    {:?} & {:?}\t- Fly up & down
    {:?}\t- Fly faster while held",
            self.mouse_key_cursor_grab,
            self.keyboard_key_toggle_cursor_grab,
            self.key_forward,
            self.key_back,
            self.key_left,
            self.key_right,
            self.key_up,
            self.key_down,
            self.key_run,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_camera_controller(
    time: Res<Time>,
    mut windows: Query<&mut Window>,
    mut mouse_events: EventReader<MouseMotion>,
    mut scroll_events: EventReader<MouseWheel>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    key_input: Res<ButtonInput<KeyCode>>,
    mut toggle_cursor_grab: Local<bool>,
    mut mouse_cursor_grab: Local<bool>,
    mut query: Query<(&mut Transform, &mut CameraController), With<Camera>>,
) {
    let dt = time.delta_seconds();

    if let Ok((mut transform, mut controller)) = query.get_single_mut() {
        if !controller.initialized {
            let (yaw, pitch, _roll) = transform.rotation.to_euler(EulerRot::YXZ);
            controller.yaw = yaw;
            controller.pitch = pitch;
            controller.initialized = true;
            info!("{}", *controller);
        }
        if !controller.enabled {
            mouse_events.clear();
            return;
        }

        let mut scroll = 0.0;
        for scroll_event in scroll_events.read() {
            let amount = match scroll_event.unit {
                MouseScrollUnit::Line => scroll_event.y,
                MouseScrollUnit::Pixel => scroll_event.y / 16.0,
            };
            scroll += amount;
        }
        controller.walk_speed += scroll * controller.scroll_factor * controller.walk_speed;
        controller.run_speed = controller.walk_speed * 3.0;

        // Handle key input
        let mut axis_input = Vec3::ZERO;
        if key_input.pressed(controller.key_forward) {
            axis_input.z += 1.0;
        }
        if key_input.pressed(controller.key_back) {
            axis_input.z -= 1.0;
        }
        if key_input.pressed(controller.key_right) {
            axis_input.x += 1.0;
        }
        if key_input.pressed(controller.key_left) {
            axis_input.x -= 1.0;
        }
        if key_input.pressed(controller.key_up) {
            axis_input.y += 1.0;
        }
        if key_input.pressed(controller.key_down) {
            axis_input.y -= 1.0;
        }

        let mut cursor_grab_change = false;
        if key_input.just_pressed(controller.keyboard_key_toggle_cursor_grab) {
            *toggle_cursor_grab = !*toggle_cursor_grab;
            cursor_grab_change = true;
        }
        if mouse_button_input.just_pressed(controller.mouse_key_cursor_grab) {
            *mouse_cursor_grab = true;
            cursor_grab_change = true;
        }
        if mouse_button_input.just_released(controller.mouse_key_cursor_grab) {
            *mouse_cursor_grab = false;
            cursor_grab_change = true;
        }
        let cursor_grab = *mouse_cursor_grab || *toggle_cursor_grab;

        // Apply movement update
        if axis_input != Vec3::ZERO {
            let max_speed = if key_input.pressed(controller.key_run) {
                controller.run_speed
            } else {
                controller.walk_speed
            };
            controller.velocity = axis_input.normalize() * max_speed;
        } else {
            let friction = controller.friction.clamp(0.0, 1.0);
            controller.velocity *= 1.0 - friction;
            if controller.velocity.length_squared() < 1e-6 {
                controller.velocity = Vec3::ZERO;
            }
        }
        let forward = *transform.forward();
        let right = *transform.right();
        transform.translation += controller.velocity.x * dt * right
            + controller.velocity.y * dt * Vec3::Y
            + controller.velocity.z * dt * forward;

        // Handle cursor grab
        if cursor_grab_change {
            if cursor_grab {
                for mut window in &mut windows {
                    if !window.focused {
                        continue;
                    }

                    window.cursor.grab_mode = CursorGrabMode::Locked;
                    window.cursor.visible = false;
                }
            } else {
                for mut window in &mut windows {
                    window.cursor.grab_mode = CursorGrabMode::None;
                    window.cursor.visible = true;
                }
            }
        }

        // Handle mouse input
        let mut mouse_delta = Vec2::ZERO;
        if cursor_grab {
            for mouse_event in mouse_events.read() {
                mouse_delta += mouse_event.delta;
            }
        } else {
            mouse_events.clear();
        }

        if mouse_delta != Vec2::ZERO {
            // Apply look update
            controller.pitch = (controller.pitch
                - mouse_delta.y * RADIANS_PER_DOT * controller.sensitivity)
                .clamp(-PI / 2., PI / 2.);
            controller.yaw -= mouse_delta.x * RADIANS_PER_DOT * controller.sensitivity;
            transform.rotation =
                Quat::from_euler(EulerRot::ZYX, 0.0, controller.yaw, controller.pitch);
        }
    }
}
