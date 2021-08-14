use bevy::prelude::{App, AssetServer, Commands, Res, Vec2};
use bevy::render2::camera::OrthographicCameraBundle;
use bevy::sprite2::{PipelinedSpriteBundle, Sprite};
use bevy::PipelinedDefaultPlugins;

fn main() {
    let mut app = App::new();
    app.add_plugins(PipelinedDefaultPlugins)
        .add_startup_system(setup);
    app.run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    // mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let texture = asset_server.load("branding/icon.png");
    commands.spawn_bundle(OrthographicCameraBundle::new_2d());

    commands.spawn_bundle(PipelinedSpriteBundle {
        // material: materials.add(texture_handle.into()),
        texture,
        ..Default::default()
    });
}
