use bevy::{
    core_pipeline::clear_color::ClearColorConfig,
    input::mouse::MouseMotion,
    pbr::{MaterialPipeline, MaterialPipelineKey, StandardMaterialFlags},
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::RenderTarget,
        mesh::MeshVertexBufferLayout,
        render_asset::RenderAssets,
        render_resource::{
            AsBindGroup, AsBindGroupShaderType, Extent3d, Face, RenderPipelineDescriptor,
            ShaderRef, ShaderType, SpecializedMeshPipelineError, TextureDescriptor,
            TextureDimension, TextureFormat, TextureUsages,
        },
        view::RenderLayers,
    },
};

#[derive(Component)]
struct PlayerCamera;

#[derive(Component)]
struct PortalCamera;

#[derive(Component)]
struct PortalEntrance;

#[derive(Component)]
struct PortalExit;

struct PortalMaterials {
    is_active: bool,
    active: Handle<PortalMaterial>,
    inactive: Handle<StandardMaterial>,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(MaterialPlugin::<PortalMaterial>::default())
        .add_startup_system(setup)
        .add_system(camera_controller)
        .add_system(toggle_portal_texture)
        .add_system(sync_portal_camera_to_player_camera)
        .run();
}

fn toggle_portal_texture(
    mut commands: Commands,
    key_input: Res<Input<KeyCode>>,
    mut portal_materials: ResMut<PortalMaterials>,
    portal: Query<Entity, With<PortalEntrance>>,
) {
    if key_input.just_pressed(KeyCode::P) {
        portal_materials.is_active = !portal_materials.is_active;
        for entity in portal.iter() {
            if portal_materials.is_active {
                commands
                    .entity(entity)
                    .remove::<Handle<StandardMaterial>>()
                    .insert(portal_materials.active.clone_weak());
            } else {
                commands
                    .entity(entity)
                    .remove::<Handle<PortalMaterial>>()
                    .insert(portal_materials.inactive.clone_weak());
            }
        }
    }
}

fn sync_portal_camera_to_player_camera(
    player_cameras: Query<
        &Transform,
        (
            With<PlayerCamera>,
            Without<PortalEntrance>,
            Without<PortalCamera>,
            Without<PortalExit>,
        ),
    >,
    portal_entrances: Query<
        &Transform,
        (
            With<PortalEntrance>,
            Without<PlayerCamera>,
            Without<PortalCamera>,
            Without<PortalExit>,
        ),
    >,
    mut portal_cameras: Query<
        &mut Transform,
        (
            With<PortalCamera>,
            Without<PlayerCamera>,
            Without<PortalEntrance>,
            Without<PortalExit>,
        ),
    >,
    portal_exits: Query<
        &Transform,
        (
            With<PortalExit>,
            Without<PlayerCamera>,
            Without<PortalEntrance>,
            Without<PortalCamera>,
        ),
    >,
) {
    let player_camera = player_cameras.single();
    let portal_entrance = portal_entrances.single();
    let mut portal_camera = portal_cameras.single_mut();
    let portal_exit = portal_exits.single();

    let m = portal_exit.compute_matrix()
        * portal_entrance.compute_matrix().inverse()
        * player_camera.compute_matrix();
    let (_, rotation, translation) = m.to_scale_rotation_translation();
    portal_camera.translation = translation;
    portal_camera.rotation = rotation;
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut portal_materials: ResMut<Assets<PortalMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Set up the portal camera and render target
    let size = Extent3d {
        width: 1280,
        height: 720,
        ..default()
    };

    // This is the texture that will be rendered to.
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
        },
        ..default()
    };

    // fill image.data with zeroes
    image.resize(size);

    let image_handle = images.add(image);

    // This material has the texture that has been rendered.
    let portal_material = portal_materials.add(PortalMaterial {
        base_color_texture: Some(image_handle.clone()),
        reflectance: 0.02,
        unlit: false,
        ..default()
    });

    commands.insert_resource(PortalMaterials {
        is_active: true,
        active: portal_material.clone(),
        inactive: materials.add(StandardMaterial {
            base_color: Color::AQUAMARINE,
            reflectance: 0.02,
            unlit: false,
            ..default()
        }),
    });

    // This specifies the layer used for the first pass, which will be attached to the first pass camera and cube.
    let first_pass_layer = RenderLayers::layer(1);

    // Set up the player scene
    // ground plane
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 10.0 })),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });
    // portal
    commands
        .spawn_bundle(MaterialMeshBundle::<PortalMaterial> {
            mesh: meshes.add(Mesh::from(shape::Quad::default())),
            material: portal_material.clone_weak(),
            transform: Transform::from_xyz(0.0, 1.0, 0.0).with_scale(Vec3::new(1.0, 2.0, 1.0)),
            ..default()
        })
        .insert(PortalEntrance);
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
    // The main pass camera.
    commands
        .spawn_bundle(Camera3dBundle {
            transform: Transform::from_xyz(0.0, 1.0, 5.0)
                .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
            ..default()
        })
        .insert_bundle((PlayerCamera, CameraController::default()));

    // Set up the portal scene
    commands
        .spawn_bundle(SpatialBundle {
            transform: Transform::from_xyz(100.0, 0.0, 0.0),
            ..default()
        })
        .with_children(|parent| {
            // ground plane
            parent
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Plane { size: 10.0 })),
                    material: materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        perceptual_roughness: 1.0,
                        ..default()
                    }),
                    ..default()
                })
                .insert(first_pass_layer);

            // left wall
            let mut transform = Transform::from_xyz(2.5, 2.5, 0.0);
            transform.rotate_z(std::f32::consts::FRAC_PI_2);
            parent
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Box::new(5.0, 0.15, 5.0))),
                    transform,
                    material: materials.add(StandardMaterial {
                        base_color: Color::INDIGO,
                        perceptual_roughness: 1.0,
                        ..default()
                    }),
                    ..default()
                })
                .insert(first_pass_layer);
            // back (right) wall
            let mut transform = Transform::from_xyz(0.0, 2.5, -2.5);
            transform.rotate_x(std::f32::consts::FRAC_PI_2);
            parent
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Box::new(5.0, 0.15, 5.0))),
                    transform,
                    material: materials.add(StandardMaterial {
                        base_color: Color::INDIGO,
                        perceptual_roughness: 1.0,
                        ..default()
                    }),
                    ..default()
                })
                .insert(first_pass_layer);

            // cube
            parent
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
                    material: materials.add(StandardMaterial {
                        base_color: Color::PINK,
                        ..default()
                    }),
                    transform: Transform::from_xyz(0.0, 0.5, 0.0),
                    ..default()
                })
                .insert(first_pass_layer);
            // sphere
            parent
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::UVSphere {
                        radius: 0.5,
                        ..default()
                    })),
                    material: materials.add(StandardMaterial {
                        base_color: Color::LIME_GREEN,
                        ..default()
                    }),
                    transform: Transform::from_xyz(1.5, 1.0, 1.5),
                    ..default()
                })
                .insert(first_pass_layer);

            // red point light
            parent
                .spawn_bundle(PointLightBundle {
                    // transform: Transform::from_xyz(5.0, 8.0, 2.0),
                    transform: Transform::from_xyz(1.0, 2.0, 0.0),
                    point_light: PointLight {
                        intensity: 1600.0, // lumens - roughly a 100W non-halogen incandescent bulb
                        color: Color::RED,
                        shadows_enabled: true,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|builder| {
                    builder
                        .spawn_bundle(PbrBundle {
                            mesh: meshes.add(Mesh::from(shape::UVSphere {
                                radius: 0.1,
                                ..default()
                            })),
                            material: materials.add(StandardMaterial {
                                base_color: Color::RED,
                                emissive: Color::rgba_linear(100.0, 0.0, 0.0, 0.0),
                                ..default()
                            }),
                            ..default()
                        })
                        .insert(first_pass_layer);
                });

            // green spot light
            parent
                .spawn_bundle(SpotLightBundle {
                    transform: Transform::from_xyz(-1.0, 2.0, 0.0)
                        .looking_at(Vec3::new(-1.0, 0.0, 0.0), Vec3::Z),
                    spot_light: SpotLight {
                        intensity: 1600.0, // lumens - roughly a 100W non-halogen incandescent bulb
                        color: Color::GREEN,
                        shadows_enabled: true,
                        inner_angle: 0.6,
                        outer_angle: 0.8,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|builder| {
                    builder
                        .spawn_bundle(PbrBundle {
                            transform: Transform::from_rotation(Quat::from_rotation_x(
                                std::f32::consts::PI / 2.0,
                            )),
                            mesh: meshes.add(Mesh::from(shape::Capsule {
                                depth: 0.125,
                                radius: 0.1,
                                ..default()
                            })),
                            material: materials.add(StandardMaterial {
                                base_color: Color::GREEN,
                                emissive: Color::rgba_linear(0.0, 100.0, 0.0, 0.0),
                                ..default()
                            }),
                            ..default()
                        })
                        .insert(first_pass_layer);
                });

            // blue point light
            parent
                .spawn_bundle(PointLightBundle {
                    // transform: Transform::from_xyz(5.0, 8.0, 2.0),
                    transform: Transform::from_xyz(0.0, 4.0, 0.0),
                    point_light: PointLight {
                        intensity: 1600.0, // lumens - roughly a 100W non-halogen incandescent bulb
                        color: Color::BLUE,
                        shadows_enabled: true,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|builder| {
                    builder
                        .spawn_bundle(PbrBundle {
                            mesh: meshes.add(Mesh::from(shape::UVSphere {
                                radius: 0.1,
                                ..default()
                            })),
                            material: materials.add(StandardMaterial {
                                base_color: Color::BLUE,
                                emissive: Color::rgba_linear(0.0, 0.0, 100.0, 0.0),
                                ..default()
                            }),
                            ..default()
                        })
                        .insert(first_pass_layer);
                });

            // portal camera
            parent
                .spawn_bundle(Camera3dBundle {
                    camera_3d: Camera3d {
                        clear_color: ClearColorConfig::Custom(Color::WHITE),
                        ..default()
                    },
                    camera: Camera {
                        // render before the "main pass" camera
                        priority: -1,
                        target: RenderTarget::Image(image_handle.clone()),
                        ..default()
                    },
                    transform: Transform::from_xyz(-2.0, 2.5, 5.0)
                        .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
                    ..default()
                })
                .insert_bundle((first_pass_layer, PortalCamera));

            parent
                .spawn_bundle(TransformBundle {
                    local: Transform::from_xyz(-2.0, 1.0, 5.0)
                        .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
                    ..default()
                })
                .insert(PortalExit);
        });

    // ambient light
    commands.insert_resource(AmbientLight {
        color: Color::ORANGE_RED,
        brightness: 0.02,
    });

    // directional 'sun' light
    const HALF_SIZE: f32 = 10.0;
    commands.spawn_bundle(DirectionalLightBundle {
        directional_light: DirectionalLight {
            // Configure the projection to better fit the scene
            shadow_projection: OrthographicProjection {
                left: -HALF_SIZE,
                right: HALF_SIZE,
                bottom: -HALF_SIZE,
                top: HALF_SIZE,
                near: -10.0 * HALF_SIZE,
                far: 10.0 * HALF_SIZE,
                ..default()
            },
            shadows_enabled: true,
            ..default()
        },
        transform: Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4),
            ..default()
        },
        ..default()
    });
}

#[derive(Component)]
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
            sensitivity: 0.5,
            key_forward: KeyCode::W,
            key_back: KeyCode::S,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_up: KeyCode::E,
            key_down: KeyCode::Q,
            key_run: KeyCode::LShift,
            mouse_key_enable_mouse: MouseButton::Left,
            keyboard_key_enable_mouse: KeyCode::M,
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

/// A material with "standard" properties used in PBR lighting
/// Standard property values with pictures here
/// <https://google.github.io/filament/Material%20Properties.pdf>.
///
/// May be created directly from a [`Color`] or an [`Image`].
#[derive(AsBindGroup, Debug, Clone, TypeUuid)]
#[uuid = "9c5a0ddf-1eaf-41b4-9832-ed736fd26af3"]
#[bind_group_data(PortalMaterialKey)]
#[uniform(0, PortalMaterialUniform)]
pub struct PortalMaterial {
    /// Doubles as diffuse albedo for non-metallic, specular for metallic and a mix for everything
    /// in between. If used together with a base_color_texture, this is factored into the final
    /// base color as `base_color * base_color_texture_value`
    pub base_color: Color,
    #[texture(1)]
    #[sampler(2)]
    pub base_color_texture: Option<Handle<Image>>,
    // Use a color for user friendliness even though we technically don't use the alpha channel
    // Might be used in the future for exposure correction in HDR
    pub emissive: Color,
    #[texture(3)]
    #[sampler(4)]
    pub emissive_texture: Option<Handle<Image>>,
    /// Linear perceptual roughness, clamped to [0.089, 1.0] in the shader
    /// Defaults to minimum of 0.089
    /// If used together with a roughness/metallic texture, this is factored into the final base
    /// color as `roughness * roughness_texture_value`
    pub perceptual_roughness: f32,
    /// From [0.0, 1.0], dielectric to pure metallic
    /// If used together with a roughness/metallic texture, this is factored into the final base
    /// color as `metallic * metallic_texture_value`
    pub metallic: f32,
    #[texture(5)]
    #[sampler(6)]
    pub metallic_roughness_texture: Option<Handle<Image>>,
    /// Specular intensity for non-metals on a linear scale of [0.0, 1.0]
    /// defaults to 0.5 which is mapped to 4% reflectance in the shader
    pub reflectance: f32,
    #[texture(9)]
    #[sampler(10)]
    pub normal_map_texture: Option<Handle<Image>>,
    /// Normal map textures authored for DirectX have their y-component flipped. Set this to flip
    /// it to right-handed conventions.
    pub flip_normal_map_y: bool,
    #[texture(7)]
    #[sampler(8)]
    pub occlusion_texture: Option<Handle<Image>>,
    /// Support two-sided lighting by automatically flipping the normals for "back" faces
    /// within the PBR lighting shader.
    /// Defaults to false.
    /// This does not automatically configure backface culling, which can be done via
    /// `cull_mode`.
    pub double_sided: bool,
    /// Whether to cull the "front", "back" or neither side of a mesh
    /// defaults to `Face::Back`
    pub cull_mode: Option<Face>,
    pub unlit: bool,
    pub alpha_mode: AlphaMode,
    pub depth_bias: f32,
}

impl Default for PortalMaterial {
    fn default() -> Self {
        Self {
            base_color: Color::rgb(1.0, 1.0, 1.0),
            base_color_texture: None,
            emissive: Color::BLACK,
            emissive_texture: None,
            // This is the minimum the roughness is clamped to in shader code
            // See <https://google.github.io/filament/Filament.html#materialsystem/parameterization/>
            // It's the minimum floating point value that won't be rounded down to 0 in the
            // calculations used. Although technically for 32-bit floats, 0.045 could be
            // used.
            perceptual_roughness: 0.089,
            // Few materials are purely dielectric or metallic
            // This is just a default for mostly-dielectric
            metallic: 0.01,
            metallic_roughness_texture: None,
            // Minimum real-world reflectance is 2%, most materials between 2-5%
            // Expressed in a linear scale and equivalent to 4% reflectance see
            // <https://google.github.io/filament/Material%20Properties.pdf>
            reflectance: 0.5,
            occlusion_texture: None,
            normal_map_texture: None,
            flip_normal_map_y: false,
            double_sided: false,
            cull_mode: Some(Face::Back),
            unlit: false,
            alpha_mode: AlphaMode::Opaque,
            depth_bias: 0.0,
        }
    }
}

/// The GPU representation of the uniform data of a [`PortalMaterial`].
#[derive(Clone, Default, ShaderType)]
pub struct PortalMaterialUniform {
    /// Doubles as diffuse albedo for non-metallic, specular for metallic and a mix for everything
    /// in between.
    pub base_color: Vec4,
    // Use a color for user friendliness even though we technically don't use the alpha channel
    // Might be used in the future for exposure correction in HDR
    pub emissive: Vec4,
    /// Linear perceptual roughness, clamped to [0.089, 1.0] in the shader
    /// Defaults to minimum of 0.089
    pub roughness: f32,
    /// From [0.0, 1.0], dielectric to pure metallic
    pub metallic: f32,
    /// Specular intensity for non-metals on a linear scale of [0.0, 1.0]
    /// defaults to 0.5 which is mapped to 4% reflectance in the shader
    pub reflectance: f32,
    pub flags: u32,
    /// When the alpha mode mask flag is set, any base color alpha above this cutoff means fully opaque,
    /// and any below means fully transparent.
    pub alpha_cutoff: f32,
}

impl AsBindGroupShaderType<PortalMaterialUniform> for PortalMaterial {
    fn as_bind_group_shader_type(&self, images: &RenderAssets<Image>) -> PortalMaterialUniform {
        let mut flags = StandardMaterialFlags::NONE;
        if self.base_color_texture.is_some() {
            flags |= StandardMaterialFlags::BASE_COLOR_TEXTURE;
        }
        if self.emissive_texture.is_some() {
            flags |= StandardMaterialFlags::EMISSIVE_TEXTURE;
        }
        if self.metallic_roughness_texture.is_some() {
            flags |= StandardMaterialFlags::METALLIC_ROUGHNESS_TEXTURE;
        }
        if self.occlusion_texture.is_some() {
            flags |= StandardMaterialFlags::OCCLUSION_TEXTURE;
        }
        if self.double_sided {
            flags |= StandardMaterialFlags::DOUBLE_SIDED;
        }
        if self.unlit {
            flags |= StandardMaterialFlags::UNLIT;
        }
        let has_normal_map = self.normal_map_texture.is_some();
        if has_normal_map {
            match images
                .get(self.normal_map_texture.as_ref().unwrap())
                .unwrap()
                .texture_format
            {
                // All 2-component unorm formats
                TextureFormat::Rg8Unorm
                | TextureFormat::Rg16Unorm
                | TextureFormat::Bc5RgUnorm
                | TextureFormat::EacRg11Unorm => {
                    flags |= StandardMaterialFlags::TWO_COMPONENT_NORMAL_MAP;
                }
                _ => {}
            }
            if self.flip_normal_map_y {
                flags |= StandardMaterialFlags::FLIP_NORMAL_MAP_Y;
            }
        }
        // NOTE: 0.5 is from the glTF default - do we want this?
        let mut alpha_cutoff = 0.5;
        match self.alpha_mode {
            AlphaMode::Opaque => flags |= StandardMaterialFlags::ALPHA_MODE_OPAQUE,
            AlphaMode::Mask(c) => {
                alpha_cutoff = c;
                flags |= StandardMaterialFlags::ALPHA_MODE_MASK;
            }
            AlphaMode::Blend => flags |= StandardMaterialFlags::ALPHA_MODE_BLEND,
        };

        PortalMaterialUniform {
            base_color: self.base_color.as_linear_rgba_f32().into(),
            emissive: self.emissive.into(),
            roughness: self.perceptual_roughness,
            metallic: self.metallic,
            reflectance: self.reflectance,
            flags: flags.bits(),
            alpha_cutoff,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PortalMaterialKey {
    normal_map: bool,
    cull_mode: Option<Face>,
}

impl From<&PortalMaterial> for PortalMaterialKey {
    fn from(material: &PortalMaterial) -> Self {
        PortalMaterialKey {
            normal_map: material.normal_map_texture.is_some(),
            cull_mode: material.cull_mode,
        }
    }
}

impl Material for PortalMaterial {
    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if key.bind_group_data.normal_map {
            descriptor
                .fragment
                .as_mut()
                .unwrap()
                .shader_defs
                .push(String::from("STANDARDMATERIAL_NORMAL_MAP"));
        }
        descriptor.primitive.cull_mode = key.bind_group_data.cull_mode;
        if let Some(label) = &mut descriptor.label {
            *label = format!("pbr_{}", *label).into();
        }
        Ok(())
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/portal.wgsl".into()
    }

    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    #[inline]
    fn depth_bias(&self) -> f32 {
        self.depth_bias
    }
}
