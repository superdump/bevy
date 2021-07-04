use bevy::{
    core::Time,
    ecs::prelude::*,
    math::Vec3,
    pbr2::{AmbientLight, DirectionalLight, DirectionalLightBundle},
    prelude::{App, AssetServer, SpawnSceneCommands, Transform},
    render2::{
        camera::{OrthographicProjection, PerspectiveCameraBundle},
        color::Color,
    },
    PipelinedDefaultPlugins,
};

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 1.0 / 5.0f32,
        })
        .add_plugins(PipelinedDefaultPlugins)
        .add_startup_system(setup.system())
        .add_system(animate_light_direction.system())
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn_scene(asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"));
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_xyz(0.7, 0.7, 1.0).looking_at(Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
        ..Default::default()
    });
    const HALF_SIZE: f32 = 1.0;
    let mut directional_light = DirectionalLight::default();
    directional_light.color = Color::WHITE;
    directional_light.shadow_projection = OrthographicProjection {
        left: -HALF_SIZE,
        right: HALF_SIZE,
        bottom: -HALF_SIZE,
        top: HALF_SIZE,
        near: -10.0 * HALF_SIZE,
        far: 10.0 * HALF_SIZE,
        ..Default::default()
    };
    directional_light.shadow_bias_min = 0.0001;
    directional_light.shadow_bias_max = 0.001;
    commands.spawn_bundle(DirectionalLightBundle {
        directional_light,
        ..Default::default()
    });
}

fn animate_light_direction(time: Res<Time>, mut query: Query<&mut DirectionalLight>) {
    for mut light in query.iter_mut() {
        let (s, c) = (time.seconds_since_startup() as f32 * std::f32::consts::TAU / 10.0).sin_cos();
        light.set_direction(Vec3::new(2.0 * s, -1.0, c));
    }
}
