use bevy::{
    core::Time,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::prelude::*,
    math::Vec3,
    pbr2::{
        PbrBundle,
        StandardMaterial,
    },
    prelude::{App, Assets, Transform},
    render2::{
        camera::PerspectiveCameraBundle,
        color::Color,
        mesh::{shape, Mesh},
    },
    PipelinedDefaultPlugins,
};

fn main() {
    App::new()
        .add_plugins(PipelinedDefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_startup_system(setup)
        .add_system(bounce)
        .run();
}

struct Bouncing;

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(
        shape::Icosphere {
            radius: 0.5,
            subdivisions: 5,
        }
        .into(),
    );

    let material = materials.add(StandardMaterial {
        emissive: Color::rgb_linear(1.0, 0.3, 0.2) * 4.0,
        ..Default::default()
    });

    for x in -10..10 {
        for z in -10..10 {
            commands
                .spawn_bundle(PbrBundle {
                    mesh: mesh.clone(),
                    material: material.clone(),
                    transform: Transform::from_xyz(x as f32 * 2.0, 0.0, z as f32 * 2.0),
                    ..Default::default()
                })
                .insert(Bouncing);
        }
    }

    // camera
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

fn bounce(time: Res<Time>, mut query: Query<&mut Transform, With<Bouncing>>) {
    for mut transform in query.iter_mut() {
        transform.translation.y = (transform.translation.x
            + transform.translation.z
            + time.seconds_since_startup() as f32)
            .sin();
    }
}
