use bevy::{
    core_pipeline::Msaa,
    ecs::prelude::*,
    input::Input,
    math::Vec3,
    pbr2::{PbrBundle, PointLightBundle, StandardMaterial},
    prelude::{App, Assets, KeyCode, Transform},
    render2::{
        camera::PerspectiveCameraBundle,
        color::Color,
        mesh::{shape, Mesh},
    },
    PipelinedDefaultPlugins,
};

/// This example shows how to configure Multi-Sample Anti-Aliasing. Setting the sample count higher
/// will result in smoother edges, but it will also increase the cost to render those edges. The
/// range should generally be somewhere between 1 (no multi sampling, but cheap) to 8 (crisp but
/// expensive)
fn main() {
    println!("Press 'm' to toggle MSAA");
    println!("Using 4x MSAA");
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(PipelinedDefaultPlugins)
        .add_startup_system(setup.system())
        .add_system(cycle_msaa.system())
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // cube
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 2.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        ..Default::default()
    });
    // light
    commands.spawn_bundle(PointLightBundle {
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..Default::default()
    });
    // camera
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_xyz(-3.0, 3.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

fn cycle_msaa(input: Res<Input<KeyCode>>, mut msaa: ResMut<Msaa>) {
    if input.just_pressed(KeyCode::M) {
        msaa.samples = if msaa.samples == 4 {
            println!("Not using MSAA");
            1
        } else {
            println!("Using 4x MSAA");
            4
        };
    }
}
