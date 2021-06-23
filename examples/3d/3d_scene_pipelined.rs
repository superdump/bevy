use bevy::{
    core::Time,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::prelude::*,
    input::Input,
    math::Vec3,
    pbr2::{
        AmbientLight, DirectionalLight, DirectionalLightBundle, OmniLight, OmniLightBundle,
        PbrBundle, StandardMaterial,
    },
    prelude::{App, Assets, KeyCode, Transform},
    render2::{
        camera::{OrthographicProjection, PerspectiveCameraBundle},
        color::Color,
        mesh::{shape, Mesh},
    },
    wgpu2::diagnostic::WgpuResourceDiagnosticsPlugin,
    PipelinedDefaultPlugins,
};

fn main() {
    App::new()
        .add_plugins(PipelinedDefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(WgpuResourceDiagnosticsPlugin::default())
        .add_startup_system(setup.system())
        .add_system(movement.system())
        .add_system(animate_light_direction.system())
        .run();
}

struct Movable;

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(AmbientLight {
        color: Color::ORANGE_RED,
        brightness: 0.02,
    });
    // plane
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 10.0 })),
        material: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..Default::default()
        }),
        ..Default::default()
    });
    // cube
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(StandardMaterial {
                base_color: Color::PINK,
                ..Default::default()
            }),
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..Default::default()
        })
        .insert(Movable);
    // sphere
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 0.5,
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::LIME_GREEN,
                ..Default::default()
            }),
            transform: Transform::from_xyz(1.5, 1.0, 1.5),
            ..Default::default()
        })
        .insert(Movable);
    // light
    commands.spawn_bundle(OmniLightBundle {
        omni_light: OmniLight {
            color: Color::RED,
            ..Default::default()
        },
        transform: Transform::from_xyz(5.0, 8.0, 2.0),
        ..Default::default()
    });
    commands.spawn_bundle(OmniLightBundle {
        omni_light: OmniLight {
            color: Color::GREEN,
            ..Default::default()
        },
        transform: Transform::from_xyz(5.0, 8.0, -2.0),
        ..Default::default()
    });
    const HALF_SIZE: f32 = 5.0;
    let mut directional_light = DirectionalLight::default();
    directional_light.color = Color::BLUE;
    directional_light.shadow_projection = OrthographicProjection {
        left: -HALF_SIZE,
        right: HALF_SIZE,
        bottom: -HALF_SIZE,
        top: HALF_SIZE,
        near: -10.0 * HALF_SIZE,
        far: 10.0 * HALF_SIZE,
        ..Default::default()
    };
    commands
        .spawn_bundle(DirectionalLightBundle {
            directional_light,
            ..Default::default()
        })
        .id();
    // camera
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .id();
}

fn animate_light_direction(time: Res<Time>, mut query: Query<&mut DirectionalLight>) {
    for mut light in query.iter_mut() {
        let (s, c) = (time.seconds_since_startup() as f32 * std::f32::consts::TAU / 10.0).sin_cos();
        light.set_direction(Vec3::new(2.0 * s, -1.0, c));
    }
}

fn movement(
    input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Movable>>,
) {
    for mut transform in query.iter_mut() {
        let mut direction = Vec3::ZERO;
        if input.pressed(KeyCode::Up) {
            direction.y += 1.0;
        }
        if input.pressed(KeyCode::Down) {
            direction.y -= 1.0;
        }
        if input.pressed(KeyCode::Left) {
            direction.x -= 1.0;
        }
        if input.pressed(KeyCode::Right) {
            direction.x += 1.0;
        }

        transform.translation += time.delta_seconds() * 2.0 * direction;
    }
}
