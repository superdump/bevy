use std::collections::HashSet;

use bevy_asset::Assets;
use bevy_ecs::prelude::*;
use bevy_math::{Mat4, UVec2, UVec3, Vec2, Vec3, Vec3A, Vec3Swizzles, Vec4, Vec4Swizzles};
use bevy_reflect::Reflect;
use bevy_render::{
    camera::{Camera, CameraProjection, OrthographicProjection},
    color::Color,
    prelude::Image,
    primitives::{Aabb, CubemapFrusta, Frustum, Sphere},
    view::{ComputedVisibility, RenderLayers, Visibility, VisibleEntities},
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::tracing::warn;
use bevy_window::Windows;

use crate::{
    calculate_cluster_factors, CubeMapFace, CubemapVisibleEntities, ViewClusterBindings,
    CUBE_MAP_FACES, MAX_POINT_LIGHTS, POINT_LIGHT_NEAR_Z,
};

/// A light that emits light in all directions from a central point.
///
/// Real-world values for `intensity` (luminous power in lumens) based on the electrical power
/// consumption of the type of real-world light are:
///
/// | Luminous Power (lumen) (i.e. the intensity member) | Incandescent non-halogen (Watts) | Incandescent halogen (Watts) | Compact fluorescent (Watts) | LED (Watts |
/// |------|-----|----|--------|-------|
/// | 200  | 25  |    | 3-5    | 3     |
/// | 450  | 40  | 29 | 9-11   | 5-8   |
/// | 800  | 60  |    | 13-15  | 8-12  |
/// | 1100 | 75  | 53 | 18-20  | 10-16 |
/// | 1600 | 100 | 72 | 24-28  | 14-17 |
/// | 2400 | 150 |    | 30-52  | 24-30 |
/// | 3100 | 200 |    | 49-75  | 32    |
/// | 4000 | 300 |    | 75-100 | 40.5  |
///
/// Source: [Wikipedia](https://en.wikipedia.org/wiki/Lumen_(unit)#Lighting)
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct PointLight {
    pub color: Color,
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    pub shadows_enabled: bool,
    pub shadow_depth_bias: f32,
    /// A bias applied along the direction of the fragment's surface normal. It is scaled to the
    /// shadow map's texel size so that it can be small close to the camera and gets larger further
    /// away.
    pub shadow_normal_bias: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        PointLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            /// Luminous power in lumens
            intensity: 800.0, // Roughly a 60W non-halogen incandescent bulb
            range: 20.0,
            radius: 0.0,
            shadows_enabled: false,
            shadow_depth_bias: Self::DEFAULT_SHADOW_DEPTH_BIAS,
            shadow_normal_bias: Self::DEFAULT_SHADOW_NORMAL_BIAS,
        }
    }
}

impl PointLight {
    pub const DEFAULT_SHADOW_DEPTH_BIAS: f32 = 0.02;
    pub const DEFAULT_SHADOW_NORMAL_BIAS: f32 = 0.6;
}

#[derive(Clone, Debug)]
pub struct PointLightShadowMap {
    pub size: usize,
}

impl Default for PointLightShadowMap {
    fn default() -> Self {
        Self { size: 1024 }
    }
}

/// A Directional light.
///
/// Directional lights don't exist in reality but they are a good
/// approximation for light sources VERY far away, like the sun or
/// the moon.
///
/// Valid values for `illuminance` are:
///
/// | Illuminance (lux) | Surfaces illuminated by                        |
/// |-------------------|------------------------------------------------|
/// | 0.0001            | Moonless, overcast night sky (starlight)       |
/// | 0.002             | Moonless clear night sky with airglow          |
/// | 0.05–0.3          | Full moon on a clear night                     |
/// | 3.4               | Dark limit of civil twilight under a clear sky |
/// | 20–50             | Public areas with dark surroundings            |
/// | 50                | Family living room lights                      |
/// | 80                | Office building hallway/toilet lighting        |
/// | 100               | Very dark overcast day                         |
/// | 150               | Train station platforms                        |
/// | 320–500           | Office lighting                                |
/// | 400               | Sunrise or sunset on a clear day.              |
/// | 1000              | Overcast day; typical TV studio lighting       |
/// | 10,000–25,000     | Full daylight (not direct sun)                 |
/// | 32,000–100,000    | Direct sunlight                                |
///
/// Source: [Wikipedia](https://en.wikipedia.org/wiki/Lux)
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct DirectionalLight {
    pub color: Color,
    /// Illuminance in lux
    pub illuminance: f32,
    pub shadows_enabled: bool,
    pub shadow_projection: OrthographicProjection,
    pub shadow_depth_bias: f32,
    /// A bias applied along the direction of the fragment's surface normal. It is scaled to the
    /// shadow map's texel size so that it is automatically adjusted to the orthographic projection.
    pub shadow_normal_bias: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        let size = 100.0;
        DirectionalLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            illuminance: 100000.0,
            shadows_enabled: false,
            shadow_projection: OrthographicProjection {
                left: -size,
                right: size,
                bottom: -size,
                top: size,
                near: -size,
                far: size,
                ..Default::default()
            },
            shadow_depth_bias: Self::DEFAULT_SHADOW_DEPTH_BIAS,
            shadow_normal_bias: Self::DEFAULT_SHADOW_NORMAL_BIAS,
        }
    }
}

impl DirectionalLight {
    pub const DEFAULT_SHADOW_DEPTH_BIAS: f32 = 0.02;
    pub const DEFAULT_SHADOW_NORMAL_BIAS: f32 = 0.6;
}

#[derive(Clone, Debug)]
pub struct DirectionalLightShadowMap {
    pub size: usize,
}

impl Default for DirectionalLightShadowMap {
    fn default() -> Self {
        #[cfg(feature = "webgl")]
        return Self { size: 2048 };
        #[cfg(not(feature = "webgl"))]
        return Self { size: 4096 };
    }
}

/// An ambient light, which lights the entire scene equally.
#[derive(Debug)]
pub struct AmbientLight {
    pub color: Color,
    /// A direct scale factor multiplied with `color` before being passed to the shader.
    pub brightness: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: Color::rgb(1.0, 1.0, 1.0),
            brightness: 0.05,
        }
    }
}

/// Add this component to make a [`Mesh`](bevy_render::mesh::Mesh) not cast shadows.
#[derive(Component)]
pub struct NotShadowCaster;
/// Add this component to make a [`Mesh`](bevy_render::mesh::Mesh) not receive shadows.
#[derive(Component)]
pub struct NotShadowReceiver;

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum SimulationLightSystems {
    AddClusters,
    AssignLightsToClusters,
    UpdateDirectionalLightFrusta,
    UpdatePointLightFrusta,
    CheckLightVisibility,
}

// Clustered-forward rendering notes
// The main initial reference material used was this rather accessible article:
// http://www.aortiz.me/2018/12/21/CG.html
// Some inspiration was taken from “Practical Clustered Shading” which is part 2 of:
// https://efficientshading.com/2015/01/01/real-time-many-light-management-and-shadows-with-clustered-shading/
// (Also note that Part 3 of the above shows how we could support the shadow mapping for many lights.)
// The z-slicing method mentioned in the aortiz article is originally from Tiago Sousa’s Siggraph 2016 talk about Doom 2016:
// http://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf

/// Configure the far z-plane mode used for the furthest depth slice for clustered forward
/// rendering
#[derive(Debug, Copy, Clone)]
pub enum ClusterFarZMode {
    /// Use the camera far-plane to determine the z-depth of the furthest cluster layer
    CameraFarPlane,
    /// Calculate the required maximum z-depth based on currently visible lights.
    /// Makes better use of available clusters, speeding up GPU lighting operations
    /// at the expense of some CPU time and using more indices in the cluster light
    /// index lists.
    MaxLightRange,
    /// Constant max z-depth
    Constant(f32),
}

/// Configure the depth-slicing strategy for clustered forward rendering
#[derive(Debug, Copy, Clone)]
pub struct ClusterZConfig {
    /// Far z plane of the first depth slice
    pub first_slice_depth: f32,
    /// Strategy for how to evaluate the far z plane of the furthest depth slice
    pub far_z_mode: ClusterFarZMode,
}

impl Default for ClusterZConfig {
    fn default() -> Self {
        Self {
            first_slice_depth: 5.0,
            far_z_mode: ClusterFarZMode::MaxLightRange,
        }
    }
}

/// Configuration of the clustering strategy for clustered forward rendering
#[derive(Debug, Copy, Clone, Component)]
pub enum ClusterConfig {
    /// Disable light cluster calculations for this view
    None,
    /// One single cluster. Optimal for low-light complexity scenes or scenes where
    /// most lights affect the entire scene.
    Single,
    /// Explicit x, y and z counts (may yield non-square x/y clusters depending on the aspect ratio)
    XYZ {
        dimensions: UVec3,
        z_config: ClusterZConfig,
        /// Specify if clusters should automatically resize in x/y if there is a risk of exceeding
        /// the available cluster-light index limit
        dynamic_resizing: bool,
    },
    /// Fixed number of z slices, x and y calculated to give square clusters
    /// with at most total clusters. For top-down games where lights will generally always be within a
    /// short depth range, it may be useful to use this configuration with 1 or few z slices. This
    /// would reduce the number of lights per cluster by distributing more clusters in screen space
    /// x/y which matches how lights are distributed in the scene.
    FixedZ {
        total: u32,
        z_slices: u32,
        z_config: ClusterZConfig,
        /// Specify if clusters should automatically resize in x/y if there is a risk of exceeding
        /// the available cluster-light index limit
        dynamic_resizing: bool,
    },
}

impl Default for ClusterConfig {
    fn default() -> Self {
        // 24 depth slices, square clusters with at most 4096 total clusters
        // use max light distance as clusters max Z-depth, first slice extends to 5.0
        Self::FixedZ {
            total: 4096,
            z_slices: 24,
            z_config: ClusterZConfig::default(),
            dynamic_resizing: true,
        }
    }
}

impl ClusterConfig {
    fn dimensions_for_screen_size(&self, screen_size: UVec2) -> UVec3 {
        match &self {
            ClusterConfig::None => UVec3::ZERO,
            ClusterConfig::Single => UVec3::ONE,
            ClusterConfig::XYZ { dimensions, .. } => *dimensions,
            ClusterConfig::FixedZ {
                total, z_slices, ..
            } => {
                let aspect_ratio = screen_size.x as f32 / screen_size.y as f32;
                let mut z_slices = *z_slices;
                if *total < z_slices {
                    warn!("ClusterConfig has more z-slices than total clusters!");
                    z_slices = *total;
                }
                let per_layer = *total as f32 / z_slices as f32;

                let y = f32::sqrt(per_layer / aspect_ratio);

                let mut x = (y * aspect_ratio) as u32;
                let mut y = y as u32;

                // check extremes
                if x == 0 {
                    x = 1;
                    y = per_layer as u32;
                }
                if y == 0 {
                    x = per_layer as u32;
                    y = 1;
                }

                UVec3::new(x, y, z_slices)
            }
        }
    }

    fn first_slice_depth(&self) -> f32 {
        match self {
            ClusterConfig::None => 0.0,
            ClusterConfig::Single => 1.0e9, // FIXME note can't use f32::MAX as the aabb explodes
            ClusterConfig::XYZ { z_config, .. } | ClusterConfig::FixedZ { z_config, .. } => {
                z_config.first_slice_depth
            }
        }
    }

    fn far_z_mode(&self) -> ClusterFarZMode {
        match self {
            ClusterConfig::None => ClusterFarZMode::Constant(0.0),
            ClusterConfig::Single => ClusterFarZMode::Constant(1.0e9), // FIXME note can't use f32::MAX as the aabb explodes
            ClusterConfig::XYZ { z_config, .. } | ClusterConfig::FixedZ { z_config, .. } => {
                z_config.far_z_mode
            }
        }
    }

    fn dynamic_resizing(&self) -> bool {
        match self {
            ClusterConfig::None | ClusterConfig::Single => false,
            ClusterConfig::XYZ {
                dynamic_resizing, ..
            }
            | ClusterConfig::FixedZ {
                dynamic_resizing, ..
            } => *dynamic_resizing,
        }
    }
}

#[derive(Component, Debug)]
pub struct Clusters {
    /// Tile size
    pub(crate) tile_size: UVec2,
    /// Number of clusters in x / y / z in the view frustum
    pub(crate) axis_slices: UVec3,
    /// Distance to the far plane of the first depth slice. The first depth slice is special
    /// and explicitly-configured to avoid having unnecessarily many slices close to the camera.
    pub(crate) near: f32,
    pub(crate) far: f32,
    aabbs: Vec<Aabb>,
    pub(crate) lights: Vec<VisiblePointLights>,
}

impl Clusters {
    fn new(tile_size: UVec2, screen_size: UVec2, z_slices: u32, near: f32, far: f32) -> Self {
        let mut clusters = Self {
            tile_size,
            axis_slices: Default::default(),
            near,
            far,
            aabbs: Default::default(),
            lights: Default::default(),
        };
        clusters.update(tile_size, screen_size, z_slices);
        clusters
    }

    fn from_screen_size_and_dimensions(
        screen_size: UVec2,
        dimensions: UVec3,
        near: f32,
        far: f32,
    ) -> Self {
        debug_assert!(screen_size.x > 0 && screen_size.y > 0);
        debug_assert!(dimensions.x > 0 && dimensions.y > 0 && dimensions.z > 0);
        Clusters::new(
            (screen_size.as_vec2() / dimensions.xy().as_vec2())
                .ceil()
                .as_uvec2(),
            screen_size,
            dimensions.z,
            near,
            far,
        )
    }

    fn update(&mut self, tile_size: UVec2, screen_size: UVec2, z_slices: u32) {
        self.tile_size = tile_size;
        self.axis_slices = (screen_size.as_vec2() / tile_size.as_vec2())
            .ceil()
            .as_uvec2()
            .extend(z_slices);
        // NOTE: Maximum 4096 clusters due to uniform buffer size constraints
        debug_assert!(self.axis_slices.x * self.axis_slices.y * self.axis_slices.z <= 4096);
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

#[allow(clippy::too_many_arguments)]
fn compute_aabb_for_cluster(
    z_near: f32,
    z_far: f32,
    tile_size: Vec2,
    screen_size: Vec2,
    inverse_projection: Mat4,
    is_orthographic: bool,
    cluster_dimensions: UVec3,
    ijk: UVec3,
) -> Aabb {
    let ijk = ijk.as_vec3();

    // Calculate the minimum and maximum points in screen space
    let p_min = ijk.xy() * tile_size;
    let p_max = p_min + tile_size;

    let cluster_min;
    let cluster_max;
    if is_orthographic {
        // Use linear depth slicing for orthographic

        // Convert to view space at the cluster near and far planes
        // NOTE: 1.0 is the near plane due to using reverse z projections
        let p_min = screen_to_view(
            screen_size,
            inverse_projection,
            p_min,
            1.0 - (ijk.z / cluster_dimensions.z as f32),
        )
        .xyz();
        let p_max = screen_to_view(
            screen_size,
            inverse_projection,
            p_max,
            1.0 - ((ijk.z + 1.0) / cluster_dimensions.z as f32),
        )
        .xyz();

        cluster_min = p_min.min(p_max);
        cluster_max = p_min.max(p_max);
    } else {
        // Convert to view space at the near plane
        // NOTE: 1.0 is the near plane due to using reverse z projections
        let p_min = screen_to_view(screen_size, inverse_projection, p_min, 1.0);
        let p_max = screen_to_view(screen_size, inverse_projection, p_max, 1.0);

        let z_far_over_z_near = -z_far / -z_near;
        let cluster_near = if ijk.z == 0.0 {
            0.0
        } else {
            -z_near * z_far_over_z_near.powf((ijk.z - 1.0) / (cluster_dimensions.z - 1) as f32)
        };
        // NOTE: This could be simplified to:
        // cluster_far = cluster_near * z_far_over_z_near;
        let cluster_far = if cluster_dimensions.z == 1 {
            -z_far
        } else {
            -z_near * z_far_over_z_near.powf(ijk.z / (cluster_dimensions.z - 1) as f32)
        };

        // Calculate the four intersection points of the min and max points with the cluster near and far planes
        let p_min_near = line_intersection_to_z_plane(Vec3::ZERO, p_min.xyz(), cluster_near);
        let p_min_far = line_intersection_to_z_plane(Vec3::ZERO, p_min.xyz(), cluster_far);
        let p_max_near = line_intersection_to_z_plane(Vec3::ZERO, p_max.xyz(), cluster_near);
        let p_max_far = line_intersection_to_z_plane(Vec3::ZERO, p_max.xyz(), cluster_far);

        cluster_min = p_min_near.min(p_min_far).min(p_max_near.min(p_max_far));
        cluster_max = p_min_near.max(p_min_far).max(p_max_near.max(p_max_far));
    }

    Aabb::from_min_max(cluster_min, cluster_max)
}

pub fn add_clusters(
    mut commands: Commands,
    cameras: Query<(Entity, Option<&ClusterConfig>), (With<Camera>, Without<Clusters>)>,
) {
    for (entity, config) in cameras.iter() {
        let config = config.copied().unwrap_or_default();
        // actual settings here don't matter - they will be overwritten in assign_lights_to_clusters
        let clusters = Clusters::from_screen_size_and_dimensions(UVec2::ONE, UVec3::ONE, 1.0, 1.0);
        commands.entity(entity).insert_bundle((clusters, config));
    }
}

fn update_clusters(
    screen_size: UVec2,
    camera: &Camera,
    cluster_dimensions: UVec3,
    clusters: &mut Clusters,
    near: f32,
    far: f32,
) {
    let is_orthographic = camera.projection_matrix.w_axis.w == 1.0;
    let inverse_projection = camera.projection_matrix.inverse();
    // Don't update clusters if screen size is 0.
    if screen_size.x == 0 || screen_size.y == 0 {
        return;
    }
    *clusters =
        Clusters::from_screen_size_and_dimensions(screen_size, cluster_dimensions, near, far);
    let screen_size = screen_size.as_vec2();
    let tile_size_u32 = clusters.tile_size;
    let tile_size = tile_size_u32.as_vec2();

    // Calculate view space AABBs
    // NOTE: It is important that these are iterated in a specific order
    // so that we can calculate the cluster index in the fragment shader!
    // I (Rob Swain) choose to scan along rows of tiles in x,y, and for each tile then scan
    // along z
    let mut aabbs = Vec::with_capacity(
        (clusters.axis_slices.y * clusters.axis_slices.x * clusters.axis_slices.z) as usize,
    );
    for y in 0..clusters.axis_slices.y {
        for x in 0..clusters.axis_slices.x {
            for z in 0..clusters.axis_slices.z {
                aabbs.push(compute_aabb_for_cluster(
                    near,
                    far,
                    tile_size,
                    screen_size,
                    inverse_projection,
                    is_orthographic,
                    clusters.axis_slices,
                    UVec3::new(x, y, z),
                ));
            }
        }
    }
    clusters.aabbs = aabbs;
}

#[derive(Clone, Component, Debug, Default)]
pub struct VisiblePointLights {
    pub entities: Vec<Entity>,
}

impl VisiblePointLights {
    pub fn from_light_count(count: usize) -> Self {
        Self {
            entities: Vec::with_capacity(count),
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Entity> {
        self.entities.iter()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

fn view_z_to_z_slice(
    cluster_factors: Vec2,
    z_slices: f32,
    view_z: f32,
    is_orthographic: bool,
) -> u32 {
    if is_orthographic {
        // NOTE: view_z is correct in the orthographic case
        ((view_z - cluster_factors.x) * cluster_factors.y).floor() as u32
    } else {
        // NOTE: had to use -view_z to make it positive else log(negative) is nan
        ((-view_z).ln() * cluster_factors.x - cluster_factors.y + 1.0).clamp(0.0, z_slices - 1.0)
            as u32
    }
}

fn ndc_position_to_cluster(
    cluster_dimensions: UVec3,
    cluster_factors: Vec2,
    is_orthographic: bool,
    ndc_p: Vec3,
    view_z: f32,
) -> UVec3 {
    let cluster_dimensions_f32 = cluster_dimensions.as_vec3();
    let frag_coord =
        (ndc_p.xy() * Vec2::new(0.5, -0.5) + Vec2::splat(0.5)).clamp(Vec2::ZERO, Vec2::ONE);
    let xy = (frag_coord * cluster_dimensions_f32.xy()).floor();
    let z_slice = view_z_to_z_slice(
        cluster_factors,
        cluster_dimensions.z as f32,
        view_z,
        is_orthographic,
    );
    xy.as_uvec2()
        .extend(z_slice)
        .clamp(UVec3::ZERO, cluster_dimensions - UVec3::ONE)
}

// Calculate bounds for the light using a view space aabb.
// Returns a (Vec3, Vec3) containing min and max with
//     x and y in normalized device coordinates with range [-1, 1]
//     z in view space, with range [-inf, -f32::MIN_POSITIVE]
fn cluster_space_light_aabb(
    inverse_view_transform: Mat4,
    projection_matrix: Mat4,
    light_sphere: &Sphere,
) -> (Vec3, Vec3) {
    let light_aabb_view = Aabb {
        center: Vec3A::from(inverse_view_transform * light_sphere.center.extend(1.0)),
        half_extents: Vec3A::splat(light_sphere.radius),
    };
    let (mut light_aabb_view_min, mut light_aabb_view_max) =
        (light_aabb_view.min(), light_aabb_view.max());

    // Constrain view z to be negative - i.e. in front of the camera
    // When view z is >= 0.0 and we're using a perspective projection, bad things happen.
    // At view z == 0.0, ndc x,y are mathematically undefined. At view z > 0.0, i.e. behind the camera,
    // the perspective projection flips the directions of the axes. This breaks assumptions about
    // use of min/max operations as something that was to the left in view space is now returning a
    // coordinate that for view z in front of the camera would be on the right, but at view z behind the
    // camera is on the left. So, we just constrain view z to be < 0.0 and necessarily in front of the camera.
    light_aabb_view_min.z = light_aabb_view_min.z.min(-f32::MIN_POSITIVE);
    light_aabb_view_max.z = light_aabb_view_max.z.min(-f32::MIN_POSITIVE);

    // Is there a cheaper way to do this? The problem is that because of perspective
    // the point at max z but min xy may be less xy in screenspace, and similar. As
    // such, projecting the min and max xy at both the closer and further z and taking
    // the min and max of those projected points addresses this.
    let (
        light_aabb_view_xymin_near,
        light_aabb_view_xymin_far,
        light_aabb_view_xymax_near,
        light_aabb_view_xymax_far,
    ) = (
        light_aabb_view_min,
        light_aabb_view_min.xy().extend(light_aabb_view_max.z),
        light_aabb_view_max.xy().extend(light_aabb_view_min.z),
        light_aabb_view_max,
    );
    let (
        light_aabb_clip_xymin_near,
        light_aabb_clip_xymin_far,
        light_aabb_clip_xymax_near,
        light_aabb_clip_xymax_far,
    ) = (
        projection_matrix * light_aabb_view_xymin_near.extend(1.0),
        projection_matrix * light_aabb_view_xymin_far.extend(1.0),
        projection_matrix * light_aabb_view_xymax_near.extend(1.0),
        projection_matrix * light_aabb_view_xymax_far.extend(1.0),
    );
    let (
        light_aabb_ndc_xymin_near,
        light_aabb_ndc_xymin_far,
        light_aabb_ndc_xymax_near,
        light_aabb_ndc_xymax_far,
    ) = (
        light_aabb_clip_xymin_near.xyz() / light_aabb_clip_xymin_near.w,
        light_aabb_clip_xymin_far.xyz() / light_aabb_clip_xymin_far.w,
        light_aabb_clip_xymax_near.xyz() / light_aabb_clip_xymax_near.w,
        light_aabb_clip_xymax_far.xyz() / light_aabb_clip_xymax_far.w,
    );
    let (light_aabb_ndc_min, light_aabb_ndc_max) = (
        light_aabb_ndc_xymin_near
            .min(light_aabb_ndc_xymin_far)
            .min(light_aabb_ndc_xymax_near)
            .min(light_aabb_ndc_xymax_far),
        light_aabb_ndc_xymin_near
            .max(light_aabb_ndc_xymin_far)
            .max(light_aabb_ndc_xymax_near)
            .max(light_aabb_ndc_xymax_far),
    );

    // pack unadjusted z depth into the vecs
    let (aabb_min, aabb_max) = (
        light_aabb_ndc_min.xy().extend(light_aabb_view_min.z),
        light_aabb_ndc_max.xy().extend(light_aabb_view_max.z),
    );
    // clamp to ndc coords
    (
        aabb_min.clamp(
            Vec3::new(-1.0, -1.0, f32::MIN),
            Vec3::new(1.0, 1.0, f32::MAX),
        ),
        aabb_max.clamp(
            Vec3::new(-1.0, -1.0, f32::MIN),
            Vec3::new(1.0, 1.0, f32::MAX),
        ),
    )
}

// Sort point lights with shadows enabled first, then by a stable key so that the index
// can be used to render at most `MAX_POINT_LIGHT_SHADOW_MAPS` point light shadows and
// we keep a stable set of lights visible
pub(crate) fn point_light_order(
    (entity_1, shadows_enabled_1): (&Entity, &bool),
    (entity_2, shadows_enabled_2): (&Entity, &bool),
) -> std::cmp::Ordering {
    shadows_enabled_1
        .cmp(shadows_enabled_2)
        .reverse()
        .then_with(|| entity_1.cmp(entity_2))
}

#[derive(Clone, Copy)]
// data required for assigning lights to clusters
pub(crate) struct PointLightAssignmentData {
    entity: Entity,
    translation: Vec3,
    range: f32,
    shadows_enabled: bool,
}

// NOTE: Run this before update_point_light_frusta!
#[allow(clippy::too_many_arguments)]
pub(crate) fn assign_lights_to_clusters(
    mut commands: Commands,
    mut global_lights: ResMut<VisiblePointLights>,
    windows: Res<Windows>,
    images: Res<Assets<Image>>,
    mut views: Query<(
        Entity,
        &GlobalTransform,
        &Camera,
        &Frustum,
        &ClusterConfig,
        &mut Clusters,
    )>,
    lights_query: Query<(Entity, &GlobalTransform, &PointLight, &Visibility)>,
    mut lights: Local<Vec<PointLightAssignmentData>>,
    mut max_point_lights_warning_emitted: Local<bool>,
) {
    // collect just the relevant light query data into a persisted vec to avoid reallocating each frame
    lights.extend(
        lights_query
            .iter()
            .filter(|(.., visibility)| visibility.is_visible)
            .map(
                |(entity, transform, light, _visibility)| PointLightAssignmentData {
                    entity,
                    translation: transform.translation,
                    shadows_enabled: light.shadows_enabled,
                    range: light.range,
                },
            ),
    );

    if lights.len() > MAX_POINT_LIGHTS {
        lights.sort_by(|light_1, light_2| {
            point_light_order(
                (&light_1.entity, &light_1.shadows_enabled),
                (&light_2.entity, &light_2.shadows_enabled),
            )
        });

        // check each light against each view's frustum, keep only those that affect at least one of our views
        let frusta: Vec<_> = views
            .iter()
            .map(|(_, _, _, frustum, _, _)| *frustum)
            .collect();
        let mut lights_in_view_count = 0;
        lights.retain(|light| {
            // take one extra light to check if we should emit the warning
            if lights_in_view_count == MAX_POINT_LIGHTS + 1 {
                false
            } else {
                let light_sphere = Sphere {
                    center: Vec3A::from(light.translation),
                    radius: light.range,
                };

                let light_in_view = frusta
                    .iter()
                    .any(|frustum| frustum.intersects_sphere(&light_sphere, true));

                if light_in_view {
                    lights_in_view_count += 1;
                }

                light_in_view
            }
        });

        if lights.len() > MAX_POINT_LIGHTS && !*max_point_lights_warning_emitted {
            warn!("MAX_POINT_LIGHTS ({}) exceeded", MAX_POINT_LIGHTS);
            *max_point_lights_warning_emitted = true;
        }

        lights.truncate(MAX_POINT_LIGHTS);
    }

    let light_count = lights.len();
    let mut global_lights_set = HashSet::with_capacity(light_count);
    for (view_entity, view_transform, camera, frustum, config, mut clusters) in views.iter_mut() {
        if matches!(config, ClusterConfig::None) {
            commands.entity(view_entity).remove::<VisiblePointLights>();
            continue;
        }

        let view_transform = view_transform.compute_matrix();
        let inverse_view_transform = view_transform.inverse();
        let is_orthographic = camera.projection_matrix.w_axis.w == 1.0;

        let screen_size_u32 = camera.target.get_physical_size(&windows, &images);
        let screen_size_u32 = screen_size_u32.unwrap_or_default();
        if screen_size_u32.x == 0 || screen_size_u32.y == 0 {
            continue;
        }
        let mut cluster_dimensions = config.dimensions_for_screen_size(screen_size_u32);

        let far_z = match config.far_z_mode() {
            ClusterFarZMode::CameraFarPlane => camera.far,
            ClusterFarZMode::MaxLightRange => {
                let inverse_view_row_2 = inverse_view_transform.row(2);
                lights
                    .iter()
                    .map(|light| {
                        -inverse_view_row_2.dot(light.translation.extend(1.0)) + light.range
                    })
                    .reduce(f32::max)
                    .unwrap_or(0.0)
            }
            ClusterFarZMode::Constant(far) => far,
        };
        let first_slice_depth = match cluster_dimensions.z {
            1 => config.first_slice_depth().max(far_z),
            _ => config.first_slice_depth(),
        };
        // NOTE: Ensure the far_z is at least as far as the first_depth_slice to avoid clustering problems.
        let far_z = far_z.max(first_slice_depth);

        let cluster_factors = calculate_cluster_factors(
            first_slice_depth,
            far_z,
            cluster_dimensions.z as f32,
            is_orthographic,
        );

        let max_indices = ViewClusterBindings::MAX_INDICES;

        if config.dynamic_resizing() {
            let mut cluster_index_estimate = 0.0;
            for light in lights.iter() {
                let light_sphere = Sphere {
                    center: Vec3A::from(light.translation),
                    radius: light.range,
                };

                // Check if the light is within the view frustum
                if !frustum.intersects_sphere(&light_sphere, true) {
                    continue;
                }

                // calculate a conservative aabb estimate of number of clusters affected by this light
                // this overestimates index counts by at most 50% (and typically much less) when the whole light range is in view
                // it can overestimate more significantly when light ranges are only partially in view
                let (light_aabb_min, light_aabb_max) = cluster_space_light_aabb(
                    inverse_view_transform,
                    camera.projection_matrix,
                    &light_sphere,
                );

                // since we won't adjust z slices we can calculate exact number of slices required in z dimension
                let z_cluster_min = view_z_to_z_slice(
                    cluster_factors,
                    cluster_dimensions.z as f32,
                    light_aabb_min.z,
                    is_orthographic,
                );
                let z_cluster_max = view_z_to_z_slice(
                    cluster_factors,
                    cluster_dimensions.z as f32,
                    light_aabb_max.z,
                    is_orthographic,
                );
                let z_count =
                    z_cluster_min.max(z_cluster_max) - z_cluster_min.min(z_cluster_max) + 1;

                // calculate x/y count using floats to avoid overestimating counts due to large initial tile sizes
                let xy_min = light_aabb_min.xy();
                let xy_max = light_aabb_max.xy();
                // multiply by 0.5 to move from [-1,1] to [-0.5, 0.5], max extent of 1 in each dimension
                let xy_count = (xy_max - xy_min)
                    * 0.5
                    * Vec2::new(cluster_dimensions.x as f32, cluster_dimensions.y as f32);

                // add up to 2 to each axis to account for overlap
                let x_overlap = if xy_min.x <= -1.0 { 0.0 } else { 1.0 }
                    + if xy_max.x >= 1.0 { 0.0 } else { 1.0 };
                let y_overlap = if xy_min.y <= -1.0 { 0.0 } else { 1.0 }
                    + if xy_max.y >= 1.0 { 0.0 } else { 1.0 };
                cluster_index_estimate +=
                    (xy_count.x + x_overlap) * (xy_count.y + y_overlap) * z_count as f32;
            }

            if cluster_index_estimate > max_indices as f32 {
                // scale x and y cluster count to be able to fit all our indices

                // we take the ratio of the actual indices over the index estimate.
                // this not not guaranteed to be small enough due to overlapped tiles, but
                // the conservative estimate is more than sufficient to cover the
                // difference
                let index_ratio = max_indices as f32 / cluster_index_estimate as f32;
                let xy_ratio = index_ratio.sqrt();

                cluster_dimensions.x =
                    ((cluster_dimensions.x as f32 * xy_ratio).floor() as u32).max(1);
                cluster_dimensions.y =
                    ((cluster_dimensions.y as f32 * xy_ratio).floor() as u32).max(1);
            }
        }

        update_clusters(
            screen_size_u32,
            camera,
            cluster_dimensions,
            &mut clusters,
            first_slice_depth,
            far_z,
        );
        // NOTE: This is here to avoid bugs in future due to update_clusters() having updated clusters.axis_slices
        // but cluster_dimensions has a different configuration.
        #[allow(unused_assignments)]
        {
            cluster_dimensions = clusters.axis_slices;
        }
        let cluster_count = clusters.aabbs.len();

        let mut clusters_lights =
            vec![VisiblePointLights::from_light_count(light_count); cluster_count];
        let mut visible_lights = Vec::with_capacity(light_count);

        for light in lights.iter() {
            let light_sphere = Sphere {
                center: Vec3A::from(light.translation),
                radius: light.range,
            };

            // Check if the light is within the view frustum
            if !frustum.intersects_sphere(&light_sphere, true) {
                continue;
            }

            // NOTE: The light intersects the frustum so it must be visible and part of the global set
            global_lights_set.insert(light.entity);
            visible_lights.push(light.entity);

            // note: caching seems to be slower than calling twice for this aabb calculation
            let (light_aabb_xy_ndc_z_view_min, light_aabb_xy_ndc_z_view_max) =
                cluster_space_light_aabb(
                    inverse_view_transform,
                    camera.projection_matrix,
                    &light_sphere,
                );

            let min_cluster = ndc_position_to_cluster(
                clusters.axis_slices,
                cluster_factors,
                is_orthographic,
                light_aabb_xy_ndc_z_view_min,
                light_aabb_xy_ndc_z_view_min.z,
            );
            let max_cluster = ndc_position_to_cluster(
                clusters.axis_slices,
                cluster_factors,
                is_orthographic,
                light_aabb_xy_ndc_z_view_max,
                light_aabb_xy_ndc_z_view_max.z,
            );
            let (min_cluster, max_cluster) =
                (min_cluster.min(max_cluster), min_cluster.max(max_cluster));

            for y in min_cluster.y..=max_cluster.y {
                let row_offset = y * clusters.axis_slices.x;
                for x in min_cluster.x..=max_cluster.x {
                    let col_offset = (row_offset + x) * clusters.axis_slices.z;
                    for z in min_cluster.z..=max_cluster.z {
                        // NOTE: cluster_index = (y * dim.x + x) * dim.z + z
                        let cluster_index = (col_offset + z) as usize;
                        let cluster_aabb = &clusters.aabbs[cluster_index];
                        if light_sphere.intersects_obb(cluster_aabb, &view_transform) {
                            clusters_lights[cluster_index].entities.push(light.entity);
                        }
                    }
                }
            }
        }

        for cluster_lights in &mut clusters_lights {
            cluster_lights.entities.shrink_to_fit();
        }

        clusters.lights = clusters_lights;
        visible_lights.shrink_to_fit();
        commands.entity(view_entity).insert(VisiblePointLights {
            entities: visible_lights,
        });
    }
    global_lights.entities = global_lights_set.into_iter().collect();
    lights.clear();
}

pub fn update_directional_light_frusta(
    mut views: Query<
        (
            &GlobalTransform,
            &DirectionalLight,
            &mut Frustum,
            &Visibility,
        ),
        Or<(Changed<GlobalTransform>, Changed<DirectionalLight>)>,
    >,
) {
    for (transform, directional_light, mut frustum, visibility) in views.iter_mut() {
        // The frustum is used for culling meshes to the light for shadow mapping
        // so if shadow mapping is disabled for this light, then the frustum is
        // not needed.
        if !directional_light.shadows_enabled || !visibility.is_visible {
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
    mut views: Query<
        (Entity, &GlobalTransform, &PointLight, &mut CubemapFrusta),
        Or<(Changed<GlobalTransform>, Changed<PointLight>)>,
    >,
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
    visible_point_lights: Query<&VisiblePointLights>,
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
        &Visibility,
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
    for (directional_light, frustum, mut visible_entities, maybe_view_mask, visibility) in
        directional_lights.iter_mut()
    {
        visible_entities.entities.clear();

        // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
        if !directional_light.shadows_enabled || !visibility.is_visible {
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
                if !frustum.intersects_obb(aabb, &transform.compute_matrix(), true) {
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
                    center: Vec3A::from(transform.translation),
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
                            if frustum.intersects_obb(aabb, &model_to_world, true) {
                                computed_visibility.is_visible = true;
                                visible_entities.entities.push(entity);
                            }
                        }
                    } else {
                        computed_visibility.is_visible = true;
                        for visible_entities in cubemap_visible_entities.iter_mut() {
                            visible_entities.entities.push(entity);
                        }
                    }
                }

                // TODO: check for big changes in visible entities len() vs capacity() (ex: 2x) and resize
                // to prevent holding unneeded memory
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_cluster_tiling(config: ClusterConfig, screen_size: UVec2) -> Clusters {
        let dims = config.dimensions_for_screen_size(screen_size);

        // note: near & far do not affect tiling
        let clusters = Clusters::from_screen_size_and_dimensions(screen_size, dims, 5.0, 1000.0);

        // check we cover the screen
        assert!(clusters.tile_size.x * clusters.axis_slices.x >= screen_size.x);
        assert!(clusters.tile_size.y * clusters.axis_slices.y >= screen_size.y);
        // check a smaller number of clusters would not cover the screen
        assert!(clusters.tile_size.x * (clusters.axis_slices.x - 1) < screen_size.x);
        assert!(clusters.tile_size.y * (clusters.axis_slices.y - 1) < screen_size.y);
        // check a smaller tilesize would not cover the screen
        assert!((clusters.tile_size.x - 1) * clusters.axis_slices.x < screen_size.x);
        assert!((clusters.tile_size.y - 1) * clusters.axis_slices.y < screen_size.y);
        // check we don't have more clusters than pixels
        assert!(clusters.axis_slices.x <= screen_size.x);
        assert!(clusters.axis_slices.y <= screen_size.y);

        clusters
    }

    #[test]
    // check tiling for small screen sizes
    fn test_default_cluster_setup_small_screensizes() {
        for x in 1..100 {
            for y in 1..100 {
                let screen_size = UVec2::new(x, y);
                let clusters = test_cluster_tiling(ClusterConfig::default(), screen_size);
                assert!(
                    clusters.axis_slices.x * clusters.axis_slices.y * clusters.axis_slices.z
                        <= 4096
                );
            }
        }
    }

    #[test]
    // check tiling for long thin screen sizes
    fn test_default_cluster_setup_small_x() {
        for x in 1..10 {
            for y in 1..5000 {
                let screen_size = UVec2::new(x, y);
                let clusters = test_cluster_tiling(ClusterConfig::default(), screen_size);
                assert!(
                    clusters.axis_slices.x * clusters.axis_slices.y * clusters.axis_slices.z
                        <= 4096
                );

                let screen_size = UVec2::new(y, x);
                let clusters = test_cluster_tiling(ClusterConfig::default(), screen_size);
                assert!(
                    clusters.axis_slices.x * clusters.axis_slices.y * clusters.axis_slices.z
                        <= 4096
                );
            }
        }
    }
}
