//! A simple 3D scene with light shining over a cube sitting on a plane.

use bevy::{asset::LoadState, prelude::*, render::view::EnvironmentMap};

mod camera_controller;
mod cubemap_materials;

use camera_controller::*;
use cubemap_materials::*;

const CUBEMAPS: [&str; 5] = [
    "textures/Storforsen4.ktx2",
    "textures/Storforsen4_basisu.ktx2",
    "textures/Storforsen4_basisu_mipmaps.ktx2",
    "textures/Storforsen4_toktx_mipmaps.ktx2",
    "textures/cubemap_kram_mipmaps_bc1_sanfrancisco.ktx2",
];

const CUBEMAP_ARRAYS: [&str; 3] = [
    "textures/cubemap_array_kram_mipmaps.ktx2",
    "textures/cubemap_array_kram_mipmaps_uncompressed.ktx2",
    "textures/cubemap_array_toktx_mipmaps.ktx2",
];

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(MaterialPlugin::<CubemapMaterial>::default())
        .add_plugin(MaterialPlugin::<CubemapArrayMaterial>::default())
        .add_startup_system(setup)
        .add_system(asset_loaded)
        .add_system(camera_controller)
        .add_system(animate_light_direction)
        .run();
}

struct Cubemap {
    is_loaded: bool,
    image_handle: Handle<Image>,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // a grid of spheres with different metallicity and roughness values
    let mesh_handle = meshes.add(Mesh::from(shape::Icosphere {
        radius: 0.45,
        subdivisions: 32,
    }));
    for y in -2..=2 {
        for x in -5..=5 {
            let x01 = (x + 5) as f32 / 10.0;
            let y01 = (y + 2) as f32 / 4.0;
            commands.spawn_bundle(PbrBundle {
                mesh: mesh_handle.clone(),
                material: materials.add(StandardMaterial {
                    base_color: Color::hex("ffd891").unwrap(),
                    metallic: y01,
                    perceptual_roughness: x01,
                    ..default()
                }),
                transform: Transform::from_xyz(x as f32, y as f32 + 0.5, 0.0),
                ..default()
            });
        }
    }
    // unlit sphere
    commands.spawn_bundle(PbrBundle {
        mesh: mesh_handle,
        material: materials.add(StandardMaterial {
            base_color: Color::hex("ffd891").unwrap(),
            unlit: true,
            ..default()
        }),
        transform: Transform::from_xyz(-5.0, -2.5, 0.0),
        ..default()
    });

    // directional 'sun' light
    commands.spawn_bundle(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 32000.0,
            ..default()
        },
        transform: Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4),
            ..default()
        },
        ..default()
    });

    let skybox_handle = asset_server.load(CUBEMAPS[4]);
    // camera
    commands
        .spawn_bundle(Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 8.0).looking_at(Vec3::default(), Vec3::Y),
            ..default()
        })
        .insert_bundle((
            CameraController::default(),
            EnvironmentMap {
                handle: asset_server.load(CUBEMAPS[4]),
            },
        ));

    commands.insert_resource(Cubemap {
        is_loaded: false,
        image_handle: skybox_handle,
    });
}

fn asset_loaded(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cubemap_materials: ResMut<Assets<CubemapMaterial>>,
    mut cubemap_array_materials: ResMut<Assets<CubemapArrayMaterial>>,
    mut cubemap: ResMut<Cubemap>,
) {
    if !cubemap.is_loaded
        && asset_server.get_load_state(cubemap.image_handle.clone_weak()) == LoadState::Loaded
    {
        println!("LOADED!");
        let is_array = images
            .get_mut(&cubemap.image_handle)
            .unwrap()
            .texture_descriptor
            .array_layer_count()
            > 6;

        // spawn cube
        if is_array {
            commands.spawn_bundle(MaterialMeshBundle::<CubemapArrayMaterial> {
                mesh: meshes.add(Mesh::from(shape::Cube { size: 10000.0 })),
                material: cubemap_array_materials.add(CubemapArrayMaterial {
                    base_color_texture: Some(cubemap.image_handle.clone_weak()),
                }),
                ..default()
            });
        } else {
            commands.spawn_bundle(MaterialMeshBundle::<CubemapMaterial> {
                mesh: meshes.add(Mesh::from(shape::Cube { size: 10000.0 })),
                material: cubemap_materials.add(CubemapMaterial {
                    base_color_texture: Some(cubemap.image_handle.clone_weak()),
                }),
                ..default()
            });
        }
        cubemap.is_loaded = true;
    }
}

fn animate_light_direction(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<DirectionalLight>>,
) {
    for mut transform in &mut query {
        transform.rotate_y(time.delta_seconds() * 0.5);
    }
}
