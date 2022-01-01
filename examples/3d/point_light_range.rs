use bevy::{
    input::mouse::MouseMotion,
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::{options::WgpuOptions, render_resource::WgpuFeatures},
};

fn main() {
    App::new()
        .insert_resource(WgpuOptions {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(WireframePlugin)
        .add_startup_system(setup)
        .add_system(update_light_range)
        .run();
}

const LUMINOUS_POWER: f32 = 1600.0;

pub struct Intensity {
    pub luminous_power: f32,
}

pub struct Illuminance {
    pub minimum: f32,
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // ground plane
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 10000.0 })),
        material: materials.add(StandardMaterial {
            base_color: Color::DARK_GRAY,
            perceptual_roughness: 1.0,
            ..Default::default()
        }),
        ..Default::default()
    });

    // disable the ambient light
    commands.insert_resource(AmbientLight {
        brightness: 0.0,
        ..Default::default()
    });

    let illuminance = Illuminance {
        minimum: PointLight::MINIMUM_ILLUMINANCE,
    };
    let intensity = Intensity {
        luminous_power: LUMINOUS_POWER,
    };
    let mut point_light = PointLight::from_luminous_power(intensity.luminous_power);
    point_light.range = PointLight::calculate_range(intensity.luminous_power, illuminance.minimum);
    commands.insert_resource(illuminance);
    commands.insert_resource(intensity);
    // white point light
    commands
        .spawn_bundle(PointLightBundle {
            transform: Transform::from_xyz(0.0, 0.1, 0.0),
            point_light,
            ..Default::default()
        })
        .with_children(|builder| {
            builder.spawn_bundle(PbrBundle {
                mesh: meshes.add(Mesh::from(shape::UVSphere {
                    radius: 0.1,
                    ..Default::default()
                })),
                material: materials.add(StandardMaterial {
                    emissive: Color::WHITE,
                    ..Default::default()
                }),
                ..Default::default()
            });
            builder
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::UVSphere {
                        radius: 1.0,
                        ..Default::default()
                    })),
                    material: materials.add(StandardMaterial {
                        base_color: Color::NONE,
                        alpha_mode: AlphaMode::Blend,
                        double_sided: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .insert(Wireframe);
            builder
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::UVSphere {
                        radius: 1.0,
                        ..Default::default()
                    })),
                    material: materials.add(StandardMaterial {
                        base_color: Color::NONE,
                        alpha_mode: AlphaMode::Blend,
                        double_sided: true,
                        ..Default::default()
                    }),
                    transform: Transform::from_scale(Vec3::splat(-1.0)),
                    ..Default::default()
                })
                .insert(Wireframe);
        });

    // camera
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_xyz(1.0, 0.1, 1.0).looking_at(Vec3::new(1.0, 0.0, 0.0), Vec3::Y),
        ..Default::default()
    });
}

const ILLUMINANCE_STEP: f32 = 0.1;
const LUMINOUS_POWER_FACTOR: f32 = 2.0;

fn update_light_range(
    time: Res<Time>,
    key_input: Res<Input<KeyCode>>,
    mut illuminance: ResMut<Illuminance>,
    mut intensity: ResMut<Intensity>,
    mut lights: Query<&mut PointLight>,
    mut spheres: Query<
        &mut Transform,
        (
            With<Wireframe>,
            With<Handle<Mesh>>,
            Without<PerspectiveProjection>,
        ),
    >,
    mut cameras: Query<&mut Transform, With<PerspectiveProjection>>,
) {
    let mut changed = false;

    if key_input.just_pressed(KeyCode::Right) {
        changed = true;
        intensity.luminous_power *= LUMINOUS_POWER_FACTOR;
    } else if key_input.just_pressed(KeyCode::Left) {
        changed = true;
        intensity.luminous_power /= LUMINOUS_POWER_FACTOR;
    }
    if key_input.just_pressed(KeyCode::Up) {
        changed = true;
        illuminance.minimum += ILLUMINANCE_STEP;
    } else if key_input.just_pressed(KeyCode::Down) {
        changed = true;
        illuminance.minimum -= ILLUMINANCE_STEP;
    }
    illuminance.minimum = illuminance.minimum.max(ILLUMINANCE_STEP);
    if !changed {
        return;
    }

    let range = PointLight::calculate_range(intensity.luminous_power, illuminance.minimum);

    println!(
        "Luminous power: {} lumens, minimum illuminance: {} lumens / meter^2, range: {} meters",
        intensity.luminous_power, illuminance.minimum, range
    );

    for mut light in lights.iter_mut() {
        light.intensity = intensity.luminous_power;
        light.range =
            PointLight::calculate_range(intensity.luminous_power, PointLight::MINIMUM_ILLUMINANCE);
    }
    for mut transform in spheres.iter_mut() {
        transform.scale = Vec3::splat(transform.scale.x.signum() * range);
    }
    for mut transform in cameras.iter_mut() {
        *transform =
            Transform::from_xyz(range, 0.1, 1.0).looking_at(Vec3::new(range, 0.0, 0.0), Vec3::Y);
    }
}
