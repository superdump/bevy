use crate::{
    AmbientLight, CubemapVisibleEntities, DirectionalLight, DirectionalLightShadowMap, MeshUniform,
    NotShadowCaster, PbrShaders, PointLight, PointLightShadowMap, TransformBindGroup,
};
use bevy_asset::Handle;
use bevy_core::FloatOrd;
use bevy_core_pipeline::Transparent3d;
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::*, SystemState},
};
use bevy_math::{
    const_vec3, Mat4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles,
};
use bevy_render2::{
    camera::{Camera, CameraProjection},
    color::Color,
    mesh::Mesh,
    primitives::{Aabb, CubemapFrusta, Frustum, Sphere},
    render_asset::RenderAssets,
    render_component::DynamicUniformIndex,
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_phase::{
        Draw, DrawFunctionId, DrawFunctions, PhaseItem, RenderPhase, TrackedRenderPass,
    },
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    shader::Shader,
    texture::*,
    view::{
        ComputedVisibility, ExtractedView, RenderLayers, ViewUniformOffset, ViewUniforms,
        Visibility, VisibleEntities,
    },
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{tracing::warn, HashMap};
use bevy_window::Windows;
use crevice::std140::AsStd140;
use std::{collections::HashSet, num::NonZeroU32};

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum LightSystems {
    AddClusters,
    UpdateClusters,
    AssignLightsToClusters,
    UpdateDirectionalLightFrusta,
    UpdatePointLightFrusta,
    CheckLightVisibility,
    ExtractClusters,
    ExtractLights,
    PrepareClusters,
    PrepareLights,
    QueueShadows,
}

pub struct ExtractedAmbientLight {
    color: Color,
    brightness: f32,
}

pub struct ExtractedPointLight {
    color: Color,
    /// luminous intensity in lumens per steradian
    intensity: f32,
    range: f32,
    radius: f32,
    transform: GlobalTransform,
    shadows_enabled: bool,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
}

pub type ExtractedPointLightShadowMap = PointLightShadowMap;

pub struct ExtractedDirectionalLight {
    color: Color,
    illuminance: f32,
    direction: Vec3,
    projection: Mat4,
    shadows_enabled: bool,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
    near: f32,
    far: f32,
}

pub type ExtractedDirectionalLightShadowMap = DirectionalLightShadowMap;

#[repr(C)]
#[derive(Copy, Clone, AsStd140, Default, Debug)]
pub struct GpuPointLight {
    projection: Mat4,
    color: Vec4,
    position: Vec3,
    inverse_square_range: f32,
    radius: f32,
    near: f32,
    far: f32,
    flags: u32,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
}

// NOTE: These must match the bit flags in bevy_pbr2/src/render/pbr.frag!
bitflags::bitflags! {
    #[repr(transparent)]
    struct PointLightFlags: u32 {
        const SHADOWS_ENABLED            = (1 << 0);
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

#[repr(C)]
#[derive(Copy, Clone, AsStd140, Default, Debug)]
pub struct GpuDirectionalLight {
    view_projection: Mat4,
    color: Vec4,
    dir_to_light: Vec3,
    flags: u32,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
}

// NOTE: These must match the bit flags in bevy_pbr2/src/render/pbr.frag!
bitflags::bitflags! {
    #[repr(transparent)]
    struct DirectionalLightFlags: u32 {
        const SHADOWS_ENABLED            = (1 << 0);
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, AsStd140)]
pub struct GpuLights {
    // TODO: this comes first to work around a WGSL alignment issue. We need to solve this issue before releasing the renderer rework
    directional_lights: [GpuDirectionalLight; MAX_DIRECTIONAL_LIGHTS],
    ambient_color: Vec4,
    cluster_dimensions: UVec4,
    n_directional_lights: u32,
}

// NOTE: this must be kept in sync with the same constants in pbr.frag
pub const MAX_POINT_LIGHTS: usize = 128;
pub const MAX_DIRECTIONAL_LIGHTS: usize = 1;
pub const POINT_SHADOW_LAYERS: u32 = (6 * MAX_POINT_LIGHTS) as u32;
pub const DIRECTIONAL_SHADOW_LAYERS: u32 = MAX_DIRECTIONAL_LIGHTS as u32;
pub const SHADOW_FORMAT: TextureFormat = TextureFormat::Depth32Float;

pub struct ShadowShaders {
    pub shader_module: ShaderModule,
    pub pipeline: RenderPipeline,
    pub view_layout: BindGroupLayout,
    pub point_light_sampler: Sampler,
    pub directional_light_sampler: Sampler,
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for ShadowShaders {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let pbr_shaders = world.get_resource::<PbrShaders>().unwrap();
        let shader = Shader::from_wgsl(include_str!("depth.wgsl"));
        let shader_module = render_device.create_shader_module(&shader);

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX | ShaderStage::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        // TODO: change this to ViewUniform::std140_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(144),
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            push_constant_ranges: &[],
            bind_group_layouts: &[&view_layout, &pbr_shaders.mesh_layout],
        });

        let pipeline = render_device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            vertex: VertexState {
                buffers: &[VertexBufferLayout {
                    array_stride: 32,
                    step_mode: InputStepMode::Vertex,
                    attributes: &[
                        // Position (GOTCHA! Vertex_Position isn't first in the buffer due to how Mesh sorts attributes (alphabetically))
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 0,
                        },
                        // Normal
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 1,
                        },
                        // Uv
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                }],
                module: &shader_module,
                entry_point: "vertex",
            },
            fragment: None,
            depth_stencil: Some(DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            layout: Some(&pipeline_layout),
            multisample: MultisampleState::default(),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                clamp_depth: false,
                conservative: false,
            },
        });

        ShadowShaders {
            shader_module,
            pipeline,
            view_layout,
            point_light_sampler: render_device.create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                compare: Some(CompareFunction::GreaterEqual),
                ..Default::default()
            }),
            directional_light_sampler: render_device.create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                compare: Some(CompareFunction::GreaterEqual),
                ..Default::default()
            }),
        }
    }
}

pub struct ExtractedClusterConfig {
    /// Tile size
    tile_size: UVec2,
    /// Number of clusters in x / y / z in the view frustum
    axis_slices: UVec3,
}

#[derive(Debug)]
pub struct Clusters {
    /// Tile size
    tile_size: UVec2,
    /// Number of clusters in x / y / z in the view frustum
    axis_slices: UVec3,
    aabbs: Vec<Aabb>,
    lights: Vec<VisiblePointLights>,
}

impl Clusters {
    fn new(tile_size: UVec2, screen_size: UVec2, z_slices: u32) -> Self {
        let mut clusters = Self {
            tile_size,
            axis_slices: Default::default(),
            aabbs: Default::default(),
            lights: Default::default(),
        };
        clusters.update(tile_size, screen_size, z_slices);
        clusters
    }

    fn update(&mut self, tile_size: UVec2, screen_size: UVec2, z_slices: u32) {
        self.tile_size = tile_size;
        self.axis_slices = UVec3::new(
            (screen_size.x + 1) / tile_size.x,
            (screen_size.y + 1) / tile_size.y,
            z_slices,
        );
    }
}

fn clip_to_view(inverse_projection: Mat4, clip: Vec4) -> Vec4 {
    let view = inverse_projection * clip;
    view / view.w
}

fn screen_to_view(screen_size: Vec2, inverse_projection: Mat4, screen: Vec2, ndc_z: f32) -> Vec4 {
    let tex_coord = screen / screen_size;
    let clip = Vec4::new(
        tex_coord.x * 2.0 - 1.0,
        (1.0 - tex_coord.y) * 2.0 - 1.0,
        ndc_z,
        1.0,
    );
    clip_to_view(inverse_projection, clip)
}

// Calculate the intersection of a ray from the eye through the view space position to a z plane
fn line_intersection_to_z_plane(origin: Vec3, p: Vec3, z: f32) -> Vec3 {
    let v = p - origin;
    let t = (z - Vec3::Z.dot(origin)) / Vec3::Z.dot(v);
    origin + t * v
}

fn compute_aabb_for_cluster(
    z_near: f32,
    z_far: f32,
    tile_size: Vec2,
    screen_size: Vec2,
    inverse_projection: Mat4,
    cluster_dimensions: UVec3,
    ijk: UVec3,
) -> Aabb {
    let ijk = ijk.as_f32();

    // Calculate the minimum and maximum points in screen space
    let p_min = ijk.xy() * tile_size;
    let p_max = p_min + tile_size;
    // dbg!(p_min);

    // Convert to view space at the near plane
    // NOTE: 1.0 is the near plane due to using reverse z projections
    let p_min = screen_to_view(screen_size, inverse_projection, p_min, 1.0);
    let p_max = screen_to_view(screen_size, inverse_projection, p_max, 1.0);
    // dbg!(p_min);

    // dbg!(z_near);
    // dbg!(z_far);
    let z_far_over_z_near = -z_far / -z_near;
    // dbg!(z_far_over_z_near);
    let cluster_near = -z_near * z_far_over_z_near.powf(ijk.z / cluster_dimensions.z as f32);
    // dbg!(cluster_near);
    // NOTE: This could be simplified to:
    // let cluster_far = cluster_near * z_far_over_z_near;
    let cluster_far = -z_near * z_far_over_z_near.powf((ijk.z + 1.0) / cluster_dimensions.z as f32);
    // dbg!(cluster_far);

    // Calculate the four intersection points of the min and max points with the cluster near and far planes
    let p_min_near = line_intersection_to_z_plane(Vec3::ZERO, p_min.xyz(), cluster_near);
    // dbg!(p_min_near);
    let p_min_far = line_intersection_to_z_plane(Vec3::ZERO, p_min.xyz(), cluster_far);
    // dbg!(p_min_far);
    let p_max_near = line_intersection_to_z_plane(Vec3::ZERO, p_max.xyz(), cluster_near);
    // dbg!(p_max_near);
    let p_max_far = line_intersection_to_z_plane(Vec3::ZERO, p_max.xyz(), cluster_far);
    // dbg!(p_max_far);

    let cluster_min = p_min_near.min(p_min_far).min(p_max_near.min(p_max_far));
    let cluster_max = p_min_near.max(p_min_far).max(p_max_near.max(p_max_far));

    // panic!("blerp");
    Aabb::from_min_max(cluster_min, cluster_max)
}

pub fn add_clusters(
    mut commands: Commands,
    windows: Res<Windows>,
    cameras: Query<(Entity, &Camera), Without<Clusters>>,
) {
    // println!("Running add_clusters with {} cameras", cameras.iter().count());
    for (entity, camera) in cameras.iter() {
        let window = windows.get(camera.window).unwrap();
        // let divisions = 8;
        let clusters = Clusters::new(
            UVec2::splat(window.physical_width() / 16),
            // UVec2::new(
            //     window.physical_width() / divisions,
            //     window.physical_height() / divisions,
            // ),
            // UVec2::new(window.physical_width() / 2, window.physical_height()),
            UVec2::new(window.physical_width(), window.physical_height()),
            24,
            // divisions,
            // 1,
        );
        // dbg!(&clusters);
        commands.entity(entity).insert(clusters);
    }
}

pub fn update_clusters(windows: Res<Windows>, mut views: Query<(&Camera, &mut Clusters)>) {
    // println!("Running add_clusters with {} cameras", cameras.iter().count());
    for (camera, mut clusters) in views.iter_mut() {
        let inverse_projection = camera.projection_matrix.inverse();
        let window = windows.get(camera.window).unwrap();
        let screen_size_u32 = UVec2::new(window.physical_width(), window.physical_height());
        let screen_size = screen_size_u32.as_f32();
        let tile_size_u32 = clusters.tile_size;
        let tile_size = tile_size_u32.as_f32();
        let z_slices = clusters.axis_slices.z;
        clusters.update(tile_size_u32, screen_size_u32, z_slices);

        // Calculate view space AABBs
        // NOTE: It is important that these are iterated in a specific order
        //       so that we can calculate the cluster index in the fragment shader!
        // I choose to scan along rows of tiles in x,y, and for each tile then scan
        // along z
        let mut aabbs = Vec::with_capacity(
            (clusters.axis_slices.y * clusters.axis_slices.x * clusters.axis_slices.z) as usize,
        );
        for y in 0..clusters.axis_slices.y {
            for x in 0..clusters.axis_slices.x {
                for z in 0..clusters.axis_slices.z {
                    // FIXME: Make independent of screen size by dropping tile size and just using i / dim.x?
                    aabbs.push(compute_aabb_for_cluster(
                        camera.near,
                        camera.far,
                        tile_size,
                        screen_size,
                        inverse_projection,
                        clusters.axis_slices,
                        UVec3::new(x, y, z),
                    ));
                }
            }
        }
        // dbg!(&aabbs);
        // panic!("blerp");
        clusters.aabbs = aabbs;
    }
}

#[derive(Clone, Debug, Default)]
pub struct VisiblePointLights {
    pub entities: Vec<Entity>,
}

impl VisiblePointLights {
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Entity> {
        self.entities.iter()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }
}

// NOTE: Run this before update_point_light_frusta!
pub fn assign_lights_to_clusters(
    mut commands: Commands,
    mut global_lights: ResMut<VisiblePointLights>,
    mut views: Query<(Entity, &GlobalTransform, &mut Clusters), With<Camera>>,
    lights: Query<(Entity, &GlobalTransform, &PointLight)>,
) {
    let light_count = lights.iter().count();
    let mut global_lights_set = HashSet::with_capacity(light_count);
    for (view_entity, view_transform, mut clusters) in views.iter_mut() {
        let view_transform = view_transform.compute_matrix();
        let cluster_count = clusters.aabbs.len();
        let mut clusters_lights = Vec::with_capacity(cluster_count);
        let mut visible_lights = HashSet::with_capacity(light_count);
        for (cluster_index, cluster_aabb) in clusters.aabbs.iter().enumerate() {
            let mut cluster_lights = Vec::with_capacity(light_count);
            for (light_index, (light_entity, transform, light)) in lights.iter().enumerate() {
                let light_sphere = Sphere {
                    center: transform.translation,
                    radius: light.range,
                };
                if light_sphere.intersects_obb(cluster_aabb, &view_transform) {
                    // println!("cluster {} assigned light {}", cluster_index, light_index);
                    // println!(
                    //     "Light {:?} intersects cluster at {:?} with {:?}",
                    //     light_sphere,
                    //     view_transform.transform_vector3(cluster_aabb.center),
                    //     cluster_aabb.half_extents,
                    // );
                    // panic!("blerp");
                    global_lights_set.insert(light_entity);
                    visible_lights.insert(light_entity);
                    cluster_lights.push(light_entity);
                }
            }
            cluster_lights.shrink_to_fit();
            clusters_lights.push(VisiblePointLights {
                entities: cluster_lights,
            });
        }
        clusters.lights = clusters_lights;
        commands.entity(view_entity).insert(VisiblePointLights {
            entities: visible_lights.into_iter().collect(),
        });
    }
    global_lights.entities = global_lights_set.into_iter().collect();
}

pub fn update_directional_light_frusta(
    mut views: Query<(&GlobalTransform, &DirectionalLight, &mut Frustum)>,
) {
    for (transform, directional_light, mut frustum) in views.iter_mut() {
        // The frustum is used for culling meshes to the light for shadow mapping
        // so if shadow mapping is disabled for this light, then the frustum is
        // not needed.
        if !directional_light.shadows_enabled {
            continue;
        }

        let view_projection = directional_light.shadow_projection.get_projection_matrix()
            * transform.compute_matrix().inverse();
        *frustum = Frustum::from_view_projection(
            &view_projection,
            &transform.translation,
            &transform.back(),
            directional_light.shadow_projection.far(),
        );
    }
}

// NOTE: Run this after assign_lights_to_clusters!
pub fn update_point_light_frusta(
    global_lights: Res<VisiblePointLights>,
    mut views: Query<(Entity, &GlobalTransform, &PointLight, &mut CubemapFrusta)>,
) {
    let projection =
        Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 1.0, POINT_LIGHT_NEAR_Z);
    let view_rotations = CUBE_MAP_FACES
        .iter()
        .map(|CubeMapFace { target, up }| GlobalTransform::identity().looking_at(*target, *up))
        .collect::<Vec<_>>();

    let global_lights_set = global_lights
        .entities
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    for (entity, transform, point_light, mut cubemap_frusta) in views.iter_mut() {
        // The frusta are used for culling meshes to the light for shadow mapping
        // so if shadow mapping is disabled for this light, then the frusta are
        // not needed.
        // Also, if the light is not relevant for any cluster, it will not be in the
        // global lights set and so there is no need to update its frusta.
        if !point_light.shadows_enabled || !global_lights_set.contains(&entity) {
            continue;
        }

        // ignore scale because we don't want to effectively scale light radius and range
        // by applying those as a view transform to shadow map rendering of objects
        // and ignore rotation because we want the shadow map projections to align with the axes
        let view_translation = GlobalTransform::from_translation(transform.translation);
        let view_backward = transform.back();

        for (view_rotation, frustum) in view_rotations.iter().zip(cubemap_frusta.iter_mut()) {
            let view = view_translation * *view_rotation;
            let view_projection = projection * view.compute_matrix().inverse();

            *frustum = Frustum::from_view_projection(
                &view_projection,
                &transform.translation,
                &view_backward,
                point_light.range,
            );
        }
    }
}

pub fn check_light_mesh_visibility(
    // NOTE: VisiblePointLights is an alias for VisibleEntities so the Without<DirectionalLight>
    //       is needed to avoid an unnecessary QuerySet
    visible_point_lights: Query<&VisiblePointLights, Without<DirectionalLight>>,
    mut point_lights: Query<(
        &PointLight,
        &GlobalTransform,
        &CubemapFrusta,
        &mut CubemapVisibleEntities,
        Option<&RenderLayers>,
    )>,
    mut directional_lights: Query<(
        &DirectionalLight,
        &Frustum,
        &mut VisibleEntities,
        Option<&RenderLayers>,
    )>,
    mut visible_entity_query: Query<
        (
            Entity,
            &Visibility,
            &mut ComputedVisibility,
            Option<&RenderLayers>,
            Option<&Aabb>,
            Option<&GlobalTransform>,
        ),
        Without<NotShadowCaster>,
    >,
) {
    // Directonal lights
    for (directional_light, frustum, mut visible_entities, maybe_view_mask) in
        directional_lights.iter_mut()
    {
        visible_entities.entities.clear();

        // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
        if !directional_light.shadows_enabled {
            continue;
        }

        let view_mask = maybe_view_mask.copied().unwrap_or_default();

        for (
            entity,
            visibility,
            mut computed_visibility,
            maybe_entity_mask,
            maybe_aabb,
            maybe_transform,
        ) in visible_entity_query.iter_mut()
        {
            if !visibility.is_visible {
                continue;
            }

            let entity_mask = maybe_entity_mask.copied().unwrap_or_default();
            if !view_mask.intersects(&entity_mask) {
                continue;
            }

            // If we have an aabb and transform, do frustum culling
            if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                if !frustum.intersects_obb(aabb, &transform.compute_matrix()) {
                    continue;
                }
            }

            computed_visibility.is_visible = true;
            visible_entities.entities.push(entity);
        }

        // TODO: check for big changes in visible entities len() vs capacity() (ex: 2x) and resize
        // to prevent holding unneeded memory
    }

    // Point lights
    for visible_lights in visible_point_lights.iter() {
        for light_entity in visible_lights.entities.iter().copied() {
            if let Ok((
                point_light,
                transform,
                cubemap_frusta,
                mut cubemap_visible_entities,
                maybe_view_mask,
            )) = point_lights.get_mut(light_entity)
            {
                for visible_entities in cubemap_visible_entities.iter_mut() {
                    visible_entities.entities.clear();
                }

                // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
                if !point_light.shadows_enabled {
                    continue;
                }

                let view_mask = maybe_view_mask.copied().unwrap_or_default();
                let light_sphere = Sphere {
                    center: transform.translation,
                    radius: point_light.range,
                };

                for (
                    entity,
                    visibility,
                    mut computed_visibility,
                    maybe_entity_mask,
                    maybe_aabb,
                    maybe_transform,
                ) in visible_entity_query.iter_mut()
                {
                    if !visibility.is_visible {
                        continue;
                    }

                    let entity_mask = maybe_entity_mask.copied().unwrap_or_default();
                    if !view_mask.intersects(&entity_mask) {
                        continue;
                    }

                    // If we have an aabb and transform, do frustum culling
                    if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                        let model_to_world = transform.compute_matrix();
                        // Do a cheap sphere vs obb test to prune out most meshes outside the sphere of the light
                        if !light_sphere.intersects_obb(aabb, &model_to_world) {
                            continue;
                        }
                        for (frustum, visible_entities) in cubemap_frusta
                            .iter()
                            .zip(cubemap_visible_entities.iter_mut())
                        {
                            if frustum.intersects_obb(aabb, &model_to_world) {
                                computed_visibility.is_visible = true;
                                visible_entities.entities.push(entity);
                            }
                        }
                    } else {
                        computed_visibility.is_visible = true;
                        for visible_entities in cubemap_visible_entities.iter_mut() {
                            visible_entities.entities.push(entity)
                        }
                    }
                }

                // TODO: check for big changes in visible entities len() vs capacity() (ex: 2x) and resize
                // to prevent holding unneeded memory
            }
        }
    }
}

pub type ExtractedClustersPointLights = Vec<VisiblePointLights>;
pub fn extract_clusters(mut commands: Commands, views: Query<(Entity, &Clusters), With<Camera>>) {
    for (entity, clusters) in views.iter() {
        commands.get_or_spawn(entity).insert_bundle((
            clusters.lights.clone(),
            ExtractedClusterConfig {
                tile_size: clusters.tile_size,
                axis_slices: clusters.axis_slices,
            },
        ));
    }
}

pub fn extract_lights(
    mut commands: Commands,
    ambient_light: Res<AmbientLight>,
    point_light_shadow_map: Res<PointLightShadowMap>,
    directional_light_shadow_map: Res<DirectionalLightShadowMap>,
    global_point_lights: Res<VisiblePointLights>,
    // visible_point_lights: Query<&VisiblePointLights>,
    point_lights: Query<(&PointLight, &CubemapVisibleEntities, &GlobalTransform)>,
    directional_lights: Query<(
        Entity,
        &DirectionalLight,
        &VisibleEntities,
        &GlobalTransform,
    )>,
) {
    commands.insert_resource(ExtractedAmbientLight {
        color: ambient_light.color,
        brightness: ambient_light.brightness,
    });
    commands.insert_resource::<ExtractedPointLightShadowMap>(point_light_shadow_map.clone());
    commands.insert_resource::<ExtractedDirectionalLightShadowMap>(
        directional_light_shadow_map.clone(),
    );
    // This is the point light shadow map texel size for one face of the cube as a distance of 1.0
    // world unit from the light.
    // point_light_texel_size = 2.0 * 1.0 * tan(PI / 4.0) / cube face width in texels
    // PI / 4.0 is half the cube face fov, tan(PI / 4.0) = 1.0, so this simplifies to:
    // point_light_texel_size = 2.0 / cube face width in texels
    // NOTE: When using various PCF kernel sizes, this will need to be adjusted, according to:
    // https://catlikecoding.com/unity/tutorials/custom-srp/point-and-spot-shadows/
    let point_light_texel_size = 2.0 / point_light_shadow_map.size as f32;

    for entity in global_point_lights.iter().copied() {
        if let Ok((point_light, cubemap_visible_entities, transform)) = point_lights.get(entity) {
            commands.get_or_spawn(entity).insert_bundle((
                ExtractedPointLight {
                    color: point_light.color,
                    // NOTE: Map from luminous power in lumens to luminous intensity in lumens per steradian
                    // for a point light. See https://google.github.io/filament/Filament.html#mjx-eqn-pointLightLuminousPower
                    // for details.
                    intensity: point_light.intensity / (4.0 * std::f32::consts::PI),
                    range: point_light.range,
                    radius: point_light.radius,
                    transform: *transform,
                    shadows_enabled: point_light.shadows_enabled,
                    shadow_depth_bias: point_light.shadow_depth_bias,
                    // The factor of SQRT_2 is for the worst-case diagonal offset
                    shadow_normal_bias: point_light.shadow_normal_bias
                        * point_light_texel_size
                        * std::f32::consts::SQRT_2,
                },
                cubemap_visible_entities.clone(),
            ));
        }
    }

    for (entity, directional_light, visible_entities, transform) in directional_lights.iter() {
        // Calulate the directional light shadow map texel size using the largest x,y dimension of
        // the orthographic projection divided by the shadow map resolution
        // NOTE: When using various PCF kernel sizes, this will need to be adjusted, according to:
        // https://catlikecoding.com/unity/tutorials/custom-srp/directional-shadows/
        let largest_dimension = (directional_light.shadow_projection.right
            - directional_light.shadow_projection.left)
            .max(
                directional_light.shadow_projection.top
                    - directional_light.shadow_projection.bottom,
            );
        let directional_light_texel_size =
            largest_dimension / directional_light_shadow_map.size as f32;
        commands.get_or_spawn(entity).insert_bundle((
            ExtractedDirectionalLight {
                color: directional_light.color,
                illuminance: directional_light.illuminance,
                direction: transform.forward(),
                projection: directional_light.shadow_projection.get_projection_matrix(),
                shadows_enabled: directional_light.shadows_enabled,
                shadow_depth_bias: directional_light.shadow_depth_bias,
                // The factor of SQRT_2 is for the worst-case diagonal offset
                shadow_normal_bias: directional_light.shadow_normal_bias
                    * directional_light_texel_size
                    * std::f32::consts::SQRT_2,
                near: directional_light.shadow_projection.near,
                far: directional_light.shadow_projection.far,
            },
            visible_entities.clone(),
        ));
    }
}

const POINT_LIGHT_NEAR_Z: f32 = 0.1f32;

// Can't do `Vec3::Y * -1.0` because mul isn't const
const NEGATIVE_X: Vec3 = const_vec3!([-1.0, 0.0, 0.0]);
const NEGATIVE_Y: Vec3 = const_vec3!([0.0, -1.0, 0.0]);
const NEGATIVE_Z: Vec3 = const_vec3!([0.0, 0.0, -1.0]);

struct CubeMapFace {
    target: Vec3,
    up: Vec3,
}

// see https://www.khronos.org/opengl/wiki/Cubemap_Texture
const CUBE_MAP_FACES: [CubeMapFace; 6] = [
    // 0 	GL_TEXTURE_CUBE_MAP_POSITIVE_X
    CubeMapFace {
        target: NEGATIVE_X,
        up: NEGATIVE_Y,
    },
    // 1 	GL_TEXTURE_CUBE_MAP_NEGATIVE_X
    CubeMapFace {
        target: Vec3::X,
        up: NEGATIVE_Y,
    },
    // 2 	GL_TEXTURE_CUBE_MAP_POSITIVE_Y
    CubeMapFace {
        target: NEGATIVE_Y,
        up: Vec3::Z,
    },
    // 3 	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
    CubeMapFace {
        target: Vec3::Y,
        up: NEGATIVE_Z,
    },
    // 4 	GL_TEXTURE_CUBE_MAP_POSITIVE_Z
    CubeMapFace {
        target: NEGATIVE_Z,
        up: NEGATIVE_Y,
    },
    // 5 	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    CubeMapFace {
        target: Vec3::Z,
        up: NEGATIVE_Y,
    },
];

fn face_index_to_name(face_index: usize) -> &'static str {
    match face_index {
        0 => "+x",
        1 => "-x",
        2 => "+y",
        3 => "-y",
        4 => "+z",
        5 => "-z",
        _ => "invalid",
    }
}

pub struct ShadowView {
    pub depth_texture_view: TextureView,
    pub pass_name: String,
}

pub struct ViewShadowBindings {
    pub point_light_depth_texture: Texture,
    pub point_light_depth_texture_view: TextureView,
    pub directional_light_depth_texture: Texture,
    pub directional_light_depth_texture_view: TextureView,
}

pub struct ViewLightEntities {
    pub lights: Vec<Entity>,
}

pub struct ViewLightsUniformOffset {
    pub offset: u32,
}

#[derive(Default)]
pub struct GlobalLightMeta {
    pub gpu_point_lights: UniformVec<GpuPointLight>,
    pub entity_to_index: HashMap<Entity, usize>,
}

#[derive(Default)]
pub struct LightMeta {
    pub view_gpu_lights: DynamicUniformVec<GpuLights>,
    pub shadow_view_bind_group: Option<BindGroup>,
}

pub enum LightEntity {
    Directional(Entity),
    Point((Entity, usize)),
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_lights(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut global_light_meta: ResMut<GlobalLightMeta>,
    mut light_meta: ResMut<LightMeta>,
    views: Query<(Entity, &ExtractedClusterConfig), With<RenderPhase<Transparent3d>>>,
    ambient_light: Res<ExtractedAmbientLight>,
    point_light_shadow_map: Res<ExtractedPointLightShadowMap>,
    directional_light_shadow_map: Res<ExtractedDirectionalLightShadowMap>,
    point_lights: Query<(Entity, &ExtractedPointLight)>,
    directional_lights: Query<(Entity, &ExtractedDirectionalLight)>,
) {
    // PERF: view.iter().count() could be views.iter().len() if we implemented ExactSizeIterator for archetype-only filters
    light_meta
        .view_gpu_lights
        .reserve_and_clear(views.iter().count(), &render_device);

    let ambient_color = ambient_light.color.as_rgba_linear() * ambient_light.brightness;
    // Pre-calculate for PointLights
    let cube_face_projection =
        Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 1.0, POINT_LIGHT_NEAR_Z);
    let cube_face_rotations = CUBE_MAP_FACES
        .iter()
        .map(|CubeMapFace { target, up }| GlobalTransform::identity().looking_at(*target, *up))
        .collect::<Vec<_>>();

    // NOTE: Because gpu_point_lights is a uniform buffer and we can have _up to_ MAX_POINT_LIGHTS
    //       but we don't know how many we will have, the buffer must always be MAX_POINT_LIGHTS large
    global_light_meta
        .gpu_point_lights
        .reserve_and_clear(MAX_POINT_LIGHTS, &render_device);
    let n_point_lights = point_lights.iter().count();
    global_light_meta.entity_to_index.clear();
    if global_light_meta.entity_to_index.capacity() < n_point_lights {
        global_light_meta.entity_to_index.reserve(n_point_lights);
    }
    for (index, (entity, light)) in point_lights.iter().enumerate() {
        let mut flags = PointLightFlags::NONE;
        if light.shadows_enabled {
            flags |= PointLightFlags::SHADOWS_ENABLED;
        }
        global_light_meta.gpu_point_lights.push(GpuPointLight {
            projection: cube_face_projection,
            // premultiply color by intensity
            // we don't use the alpha at all, so no reason to multiply only [0..3]
            color: (light.color.as_rgba_linear() * light.intensity).into(),
            radius: light.radius,
            position: light.transform.translation,
            inverse_square_range: 1.0 / (light.range * light.range),
            near: POINT_LIGHT_NEAR_Z,
            far: light.range,
            flags: flags.bits,
            shadow_depth_bias: light.shadow_depth_bias,
            shadow_normal_bias: light.shadow_normal_bias,
        });
        global_light_meta.entity_to_index.insert(entity, index);
    }
    // NOTE: Pad up to max point lights to meet the fixed size uniform buffer requirement
    for _ in n_point_lights..MAX_POINT_LIGHTS {
        global_light_meta
            .gpu_point_lights
            .push(GpuPointLight::default());
    }
    global_light_meta
        .gpu_point_lights
        .write_buffer(&render_queue);

    // set up light data for each view
    for (entity, clusters) in views.iter() {
        let point_light_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                size: Extent3d {
                    width: point_light_shadow_map.size as u32,
                    height: point_light_shadow_map.size as u32,
                    depth_or_array_layers: POINT_SHADOW_LAYERS,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SHADOW_FORMAT,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
                label: None,
            },
        );
        let directional_light_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                size: Extent3d {
                    width: directional_light_shadow_map.size as u32,
                    height: directional_light_shadow_map.size as u32,
                    depth_or_array_layers: DIRECTIONAL_SHADOW_LAYERS,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SHADOW_FORMAT,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
                label: None,
            },
        );
        let mut view_lights = Vec::new();

        let mut gpu_lights = GpuLights {
            directional_lights: [GpuDirectionalLight::default(); MAX_DIRECTIONAL_LIGHTS],
            ambient_color: ambient_color.into(),
            cluster_dimensions: clusters.axis_slices.extend(0),
            n_directional_lights: directional_lights.iter().len() as u32,
        };

        // TODO: this should select lights based on relevance to the view instead of the first ones that show up in a query
        for (light_entity, light) in point_lights.iter() {
            if !light.shadows_enabled {
                continue;
            }
            let light_index = *global_light_meta
                .entity_to_index
                .get(&light_entity)
                .unwrap();
            // ignore scale because we don't want to effectively scale light radius and range
            // by applying those as a view transform to shadow map rendering of objects
            // and ignore rotation because we want the shadow map projections to align with the axes
            let view_translation = GlobalTransform::from_translation(light.transform.translation);

            for (face_index, view_rotation) in cube_face_rotations.iter().enumerate() {
                let depth_texture_view =
                    point_light_depth_texture
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: None,
                            format: None,
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: None,
                            base_array_layer: (light_index * 6 + face_index) as u32,
                            array_layer_count: NonZeroU32::new(1),
                        });

                let view_light_entity = commands
                    .spawn()
                    .insert_bundle((
                        ShadowView {
                            depth_texture_view,
                            pass_name: format!(
                                "shadow pass point light {} {}",
                                light_index,
                                face_index_to_name(face_index)
                            ),
                        },
                        ExtractedView {
                            width: point_light_shadow_map.size as u32,
                            height: point_light_shadow_map.size as u32,
                            transform: view_translation * *view_rotation,
                            projection: cube_face_projection,
                            near: POINT_LIGHT_NEAR_Z,
                            far: light.range,
                        },
                        RenderPhase::<Shadow>::default(),
                        LightEntity::Point((light_entity, face_index)),
                    ))
                    .id();
                view_lights.push(view_light_entity);
            }
        }

        for (i, (light_entity, light)) in directional_lights
            .iter()
            .enumerate()
            .take(MAX_DIRECTIONAL_LIGHTS)
        {
            // direction is negated to be ready for N.L
            let dir_to_light = -light.direction;

            // convert from illuminance (lux) to candelas
            //
            // exposure is hard coded at the moment but should be replaced
            // by values coming from the camera
            // see: https://google.github.io/filament/Filament.html#imagingpipeline/physicallybasedcamera/exposuresettings
            const APERTURE: f32 = 4.0;
            const SHUTTER_SPEED: f32 = 1.0 / 250.0;
            const SENSITIVITY: f32 = 100.0;
            let ev100 =
                f32::log2(APERTURE * APERTURE / SHUTTER_SPEED) - f32::log2(SENSITIVITY / 100.0);
            let exposure = 1.0 / (f32::powf(2.0, ev100) * 1.2);
            let intensity = light.illuminance * exposure;

            // NOTE: A directional light seems to have to have an eye position on the line along the direction of the light
            //       through the world origin. I (Rob Swain) do not yet understand why it cannot be translated away from this.
            let view = Mat4::look_at_rh(Vec3::ZERO, light.direction, Vec3::Y);
            // NOTE: This orthographic projection defines the volume within which shadows from a directional light can be cast
            let projection = light.projection;

            let mut flags = DirectionalLightFlags::NONE;
            if light.shadows_enabled {
                flags |= DirectionalLightFlags::SHADOWS_ENABLED;
            }

            gpu_lights.directional_lights[i] = GpuDirectionalLight {
                // premultiply color by intensity
                // we don't use the alpha at all, so no reason to multiply only [0..3]
                color: (light.color.as_rgba_linear() * intensity).into(),
                dir_to_light,
                // NOTE: * view is correct, it should not be view.inverse() here
                view_projection: projection * view,
                flags: flags.bits,
                shadow_depth_bias: light.shadow_depth_bias,
                shadow_normal_bias: light.shadow_normal_bias,
            };

            let depth_texture_view =
                directional_light_depth_texture
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: None,
                        format: None,
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: None,
                        base_array_layer: i as u32,
                        array_layer_count: NonZeroU32::new(1),
                    });

            let view_light_entity = commands
                .spawn()
                .insert_bundle((
                    ShadowView {
                        depth_texture_view,
                        pass_name: format!("shadow pass directional light {}", i),
                    },
                    ExtractedView {
                        width: directional_light_shadow_map.size as u32,
                        height: directional_light_shadow_map.size as u32,
                        transform: GlobalTransform::from_matrix(view.inverse()),
                        projection,
                        near: light.near,
                        far: light.far,
                    },
                    RenderPhase::<Shadow>::default(),
                    LightEntity::Directional(light_entity),
                ))
                .id();
            view_lights.push(view_light_entity);
        }
        let point_light_depth_texture_view =
            point_light_depth_texture
                .texture
                .create_view(&TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: Some(TextureViewDimension::CubeArray),
                    aspect: TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                });
        let directional_light_depth_texture_view = directional_light_depth_texture
            .texture
            .create_view(&TextureViewDescriptor {
                label: None,
                format: None,
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            });

        commands.entity(entity).insert_bundle((
            ViewShadowBindings {
                point_light_depth_texture: point_light_depth_texture.texture,
                point_light_depth_texture_view,
                directional_light_depth_texture: directional_light_depth_texture.texture,
                directional_light_depth_texture_view,
            },
            ViewLightEntities {
                lights: view_lights,
            },
            ViewLightsUniformOffset {
                offset: light_meta.view_gpu_lights.push(gpu_lights),
            },
        ));
    }

    light_meta.view_gpu_lights.write_buffer(&render_queue);
}

const CLUSTER_OFFSET_MASK: u32 = (1 << 24) - 1;
const CLUSTER_COUNT_SIZE: u32 = 8;
const CLUSTER_COUNT_MASK: u32 = (1 << 8) - 1;
const POINT_LIGHT_INDEX_MASK: u32 = (1 << 8) - 1;

// NOTE: With uniform buffer max binding size as 16384 bytes
//       that means we can fit say 128 point lights in one uniform
//       buffer, which means the count can be at most 128 so it
//       needs 7 bits, use 8 for convenience.
//       The array of indices can also use u8 and that means the
//       offset in to the array of indices needs to be able to address
//       16384 values. lod2(16384) = 21 bits.
//       This means we can pack the offset into the upper 24 bits of a u32
//       and the count into the lower 8 bits.
// FIXME: Probably there are endianness concerns here????!!!!!
fn pack_offset_and_count(offset: usize, count: usize) -> u32 {
    ((offset as u32 & CLUSTER_OFFSET_MASK) << CLUSTER_COUNT_SIZE)
        | (count as u32 & CLUSTER_COUNT_MASK)
}

#[derive(Default)]
pub struct ViewClusterBindings {
    n_indices: usize,
    // NOTE: UVec4 is because all arrays in Std140 layout have 16-byte alignment
    pub cluster_light_index_lists: UniformVec<UVec4>,
    n_offsets: usize,
    // NOTE: UVec4 is because all arrays in Std140 layout have 16-byte alignment
    pub cluster_offsets_and_counts: UniformVec<UVec4>,
}

impl ViewClusterBindings {
    const MAX_UNIFORM_ITEMS: usize = 16384 / (4 * 4);
    const MAX_INDICES: usize = 16384;
    const MAX_CLUSTERS: usize = 4096;

    pub fn reserve(&mut self, render_device: &RenderDevice) {
        self.cluster_light_index_lists
            .reserve(Self::MAX_UNIFORM_ITEMS, render_device);
        self.cluster_offsets_and_counts
            .reserve(Self::MAX_UNIFORM_ITEMS, render_device);
    }

    pub fn push_offset_and_count(&mut self, offset: usize, count: usize) {
        let packed = pack_offset_and_count(offset, count);

        let component = self.n_offsets & ((1 << 2) - 1);
        if component == 0 {
            self.cluster_offsets_and_counts
                .push(UVec4::new(packed, 0, 0, 0));
        } else {
            let array_index = self.n_offsets >> 2; // >> 2 is equivalent to / 4
            let array_value = self.cluster_offsets_and_counts.get_mut(array_index);
            array_value[component] = packed;
        }

        self.n_offsets += 1;
    }

    pub fn n_indices(&self) -> usize {
        self.n_indices
    }

    pub fn push_index(&mut self, index: usize) {
        // NOTE: Packing four u8s into a u32 and four u32s into a UVec4 so we need to check
        //       whether to add a new UVec4 or to bitwise-or the value into a position in an existing u32
        let index = index as u32 & POINT_LIGHT_INDEX_MASK;

        // If n_indices % 16 == 0 then we need to add a new value, else we get the current
        // one
        let sub_indices = self.n_indices & ((1 << 4) - 1);
        if sub_indices == 0 {
            self.cluster_light_index_lists
                .push(UVec4::new(index, 0, 0, 0));
        } else {
            let array_index = self.n_indices >> 4; // >> 4 is equivalent to / 16
            let array_value = self.cluster_light_index_lists.get_mut(array_index);
            let component = (sub_indices >> 2) & ((1 << 2) - 1);
            let sub_index = sub_indices & ((1 << 2) - 1);
            array_value[component] |= index << (8 * sub_index);
        }

        self.n_indices += 1;
    }

    pub fn pad_uniform_buffers(&mut self) {
        // NOTE: We want to allow 'up to' MAX_CLUSTER_LIGHT_INDEX_LISTS_ITEMS * 4
        //       light indices and MAX_CLUSTERS clusters and we must always use
        //       full bindings
        // println!(
        //     "Padding {} index items (i.e. indices / 4) from {}",
        //     ViewClusterBindings::MAX_CLUSTER_LIGHT_INDEX_LISTS_ITEMS
        //         - self.cluster_light_index_lists.len(),
        //     self.cluster_light_index_lists.len()
        // );
        for _ in self.cluster_light_index_lists.len()..ViewClusterBindings::MAX_UNIFORM_ITEMS {
            self.cluster_light_index_lists.push(UVec4::ZERO);
        }
        // println!(
        //     "Padding {} cluster offsets and counts from {}",
        //     ViewClusterBindings::MAX_CLUSTERS - self.cluster_offsets_and_counts.len(),
        //     self.cluster_offsets_and_counts.len()
        // );
        for _ in self.cluster_offsets_and_counts.len()..ViewClusterBindings::MAX_UNIFORM_ITEMS {
            self.cluster_offsets_and_counts.push(UVec4::ZERO);
        }
    }
}

pub fn prepare_clusters(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    global_light_meta: Res<GlobalLightMeta>,
    views: Query<
        (
            Entity,
            &ExtractedClusterConfig,
            &ExtractedClustersPointLights,
        ),
        With<RenderPhase<Transparent3d>>,
    >,
) {
    for (entity, cluster_config, extracted_clusters) in views.iter() {
        let mut view_clusters_bindings = ViewClusterBindings::default();
        view_clusters_bindings.reserve(&*render_device);

        let mut indices_full = false;

        let mut cluster_index = 0;
        for _y in 0..cluster_config.axis_slices.y {
            for _x in 0..cluster_config.axis_slices.x {
                for _z in 0..cluster_config.axis_slices.z {
                    let offset = view_clusters_bindings.n_indices();
                    let cluster_lights = &extracted_clusters[cluster_index];
                    let count = cluster_lights.len();
                    view_clusters_bindings.push_offset_and_count(offset, count);

                    if !indices_full {
                        for entity in cluster_lights.iter() {
                            if view_clusters_bindings.n_indices()
                                >= ViewClusterBindings::MAX_INDICES
                            {
                                warn!("Cluster light index lists is full! The PointLights in the view are affecting too many clusters.");
                                indices_full = true;
                                break;
                            }
                            let light_index =
                                *global_light_meta.entity_to_index.get(entity).unwrap();
                            view_clusters_bindings.push_index(light_index);
                        }
                    }

                    cluster_index += 1;
                }
            }
        }

        // NOTE: Because cluster_light_index_lists and cluster_offsets_and_counts are uniform buffers,
        //       they must be a fixed size so we pad them up to the binding sizes.
        view_clusters_bindings.pad_uniform_buffers();
        view_clusters_bindings
            .cluster_light_index_lists
            .write_buffer(&render_queue);
        view_clusters_bindings
            .cluster_offsets_and_counts
            .write_buffer(&render_queue);

        // dbg!(view_clusters_bindings.cluster_light_index_lists.values());
        // dbg!(view_clusters_bindings.cluster_offsets_and_counts.values());
        // panic!("SMURF");

        commands.get_or_spawn(entity).insert(view_clusters_bindings);
    }
}

pub fn queue_shadow_view_bind_group(
    render_device: Res<RenderDevice>,
    shadow_shaders: Res<ShadowShaders>,
    mut light_meta: ResMut<LightMeta>,
    view_uniforms: Res<ViewUniforms>,
) {
    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        light_meta.shadow_view_bind_group =
            Some(render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: view_binding,
                }],
                label: None,
                layout: &shadow_shaders.view_layout,
            }));
    }
}

pub fn queue_shadows(
    shadow_draw_functions: Res<DrawFunctions<Shadow>>,
    view_lights: Query<&ViewLightEntities>,
    mut view_light_shadow_phases: Query<(&LightEntity, &mut RenderPhase<Shadow>)>,
    point_light_entities: Query<&CubemapVisibleEntities, With<ExtractedPointLight>>,
    directional_light_entities: Query<&VisibleEntities, With<ExtractedDirectionalLight>>,
) {
    for view_lights in view_lights.iter() {
        let draw_shadow_mesh = shadow_draw_functions
            .read()
            .get_id::<DrawShadowMesh>()
            .unwrap();
        for view_light_entity in view_lights.lights.iter().copied() {
            let (light_entity, mut shadow_phase) =
                view_light_shadow_phases.get_mut(view_light_entity).unwrap();
            let visible_entities = match light_entity {
                LightEntity::Directional(light_entity) => directional_light_entities
                    .get(*light_entity)
                    .expect("Failed to get directional light visible entities"),
                LightEntity::Point((light_entity, face_index)) => point_light_entities
                    .get(*light_entity)
                    .expect("Failed to get point light visible entities")
                    .get(*face_index),
            };
            // NOTE: Lights with shadow mapping disabled will have no visible entities
            //       so no meshes will be queued
            for entity in visible_entities.iter().copied() {
                shadow_phase.add(Shadow {
                    draw_function: draw_shadow_mesh,
                    entity,
                    distance: 0.0, // TODO: sort back-to-front
                });
            }
        }
    }
}

pub struct Shadow {
    pub distance: f32,
    pub entity: Entity,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for Shadow {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

pub struct ShadowPassNode {
    main_view_query: QueryState<&'static ViewLightEntities>,
    view_light_query: QueryState<(&'static ShadowView, &'static RenderPhase<Shadow>)>,
}

impl ShadowPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
            view_light_query: QueryState::new(world),
        }
    }
}

impl Node for ShadowPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(ShadowPassNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.main_view_query.update_archetypes(world);
        self.view_light_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        if let Ok(view_lights) = self.main_view_query.get_manual(world, view_entity) {
            for view_light_entity in view_lights.lights.iter().copied() {
                let (view_light, shadow_phase) = self
                    .view_light_query
                    .get_manual(world, view_light_entity)
                    .unwrap();
                let pass_descriptor = RenderPassDescriptor {
                    label: Some(&view_light.pass_name),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &view_light.depth_texture_view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                };

                let draw_functions = world.get_resource::<DrawFunctions<Shadow>>().unwrap();

                let render_pass = render_context
                    .command_encoder
                    .begin_render_pass(&pass_descriptor);
                let mut draw_functions = draw_functions.write();
                let mut tracked_pass = TrackedRenderPass::new(render_pass);
                for item in shadow_phase.items.iter() {
                    let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                    draw_function.draw(world, &mut tracked_pass, view_light_entity, item);
                }
            }
        }

        Ok(())
    }
}

pub struct DrawShadowMesh {
    params: SystemState<(
        SRes<ShadowShaders>,
        SRes<LightMeta>,
        SRes<TransformBindGroup>,
        SRes<RenderAssets<Mesh>>,
        SQuery<(Read<DynamicUniformIndex<MeshUniform>>, Read<Handle<Mesh>>)>,
        SQuery<Read<ViewUniformOffset>>,
    )>,
}

impl DrawShadowMesh {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

impl Draw<Shadow> for DrawShadowMesh {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Shadow,
    ) {
        let (shadow_shaders, light_meta, transform_bind_group, meshes, items, views) =
            self.params.get(world);
        let (transform_index, mesh_handle) = items.get(item.entity).unwrap();
        let view_uniform_offset = views.get(view).unwrap();
        pass.set_render_pipeline(&shadow_shaders.into_inner().pipeline);
        pass.set_bind_group(
            0,
            light_meta
                .into_inner()
                .shadow_view_bind_group
                .as_ref()
                .unwrap(),
            &[view_uniform_offset.offset],
        );

        pass.set_bind_group(
            1,
            &transform_bind_group.into_inner().value,
            &[transform_index.index()],
        );

        let gpu_mesh = meshes.into_inner().get(mesh_handle).unwrap();
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        if let Some(index_info) = &gpu_mesh.index_info {
            pass.set_index_buffer(index_info.buffer.slice(..), 0, IndexFormat::Uint32);
            pass.draw_indexed(0..index_info.count, 0, 0..1);
        } else {
            panic!("non-indexed drawing not supported yet")
        }
    }
}
