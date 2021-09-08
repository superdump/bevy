// TODO: try merging this block with the binding?
// NOTE: Keep in sync with depth.wgsl
[[block]]
struct View {
    view_proj: mat4x4<f32>;
    projection: mat4x4<f32>;
    world_position: vec3<f32>;
    frame_number: u32;
};


[[block]]
struct Mesh {
    model: mat4x4<f32>;
    inverse_transpose_model: mat4x4<f32>;
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32;
};

let MESH_FLAGS_SHADOW_RECEIVER_BIT: u32 = 1u;

[[group(0), binding(0)]]
var view: View;
[[group(2), binding(0)]]
var mesh: Mesh;

struct Vertex {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_position = mesh.model * vec4<f32>(vertex.position, 1.0);

    var out: VertexOutput;
    out.uv = vertex.uv;
    out.world_position = world_position;
    out.clip_position = view.view_proj * world_position;
    out.world_normal = mat3x3<f32>(
        mesh.inverse_transpose_model.x.xyz,
        mesh.inverse_transpose_model.y.xyz,
        mesh.inverse_transpose_model.z.xyz
    ) * vertex.normal;
    return out;
}

// From the Filament design doc
// https://google.github.io/filament/Filament.html#table_symbols
// Symbol Definition
// v    View unit vector
// l    Incident light unit vector
// n    Surface normal unit vector
// h    Half unit vector between l and v
// f    BRDF
// f_d    Diffuse component of a BRDF
// f_r    Specular component of a BRDF
// α    Roughness, remapped from using input perceptualRoughness
// σ    Diffuse reflectance
// Ω    Spherical domain
// f0    Reflectance at normal incidence
// f90    Reflectance at grazing angle
// χ+(a)    Heaviside function (1 if a>0 and 0 otherwise)
// nior    Index of refraction (IOR) of an interface
// ⟨n⋅l⟩    Dot product clamped to [0..1]
// ⟨a⟩    Saturated value (clamped to [0..1])

// The Bidirectional Reflectance Distribution Function (BRDF) describes the surface response of a standard material
// and consists of two components, the diffuse component (f_d) and the specular component (f_r):
// f(v,l) = f_d(v,l) + f_r(v,l)
//
// The form of the microfacet model is the same for diffuse and specular
// f_r(v,l) = f_d(v,l) = 1 / { |n⋅v||n⋅l| } ∫_Ω D(m,α) G(v,l,m) f_m(v,l,m) (v⋅m) (l⋅m) dm
//
// In which:
// D, also called the Normal Distribution Function (NDF) models the distribution of the microfacets
// G models the visibility (or occlusion or shadow-masking) of the microfacets
// f_m is the microfacet BRDF and differs between specular and diffuse components
//
// The above integration needs to be approximated.

[[block]]
struct StandardMaterial {
    base_color: vec4<f32>;
    emissive: vec4<f32>;
    perceptual_roughness: f32;
    metallic: f32;
    reflectance: f32;
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32;
};

let STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT: u32         = 1u;
let STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BIT: u32           = 2u;
let STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 4u;
let STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BIT: u32          = 8u;
let STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT: u32               = 16u;
let STANDARD_MATERIAL_FLAGS_UNLIT_BIT: u32                      = 32u;

struct PointLight {
    projection_lh: mat4x4<f32>;
    inverse_projection_lh: mat4x4<f32>;
    color: vec4<f32>;
    position_lh: vec3<f32>;
    inverse_square_range: f32;
    radius: f32;
    near: f32;
    far: f32;
    shadow_depth_bias: f32;
    shadow_normal_bias: f32;
};

struct DirectionalLight {
    view_projection: mat4x4<f32>;
    inverse_view: mat4x4<f32>;
    projection: mat4x4<f32>;
    inverse_projection: mat4x4<f32>;
    color: vec4<f32>;
    direction_to_light: vec3<f32>;
    tan_half_light_angle: f32;
    shadow_depth_bias: f32;
    shadow_normal_bias: f32;
    left: f32;
    right: f32;
    bottom: f32;
    top: f32;
    near: f32;
    far: f32;
};

[[block]]
struct Lights {
    // NOTE: this array size must be kept in sync with the constants defined bevy_pbr2/src/render/light.rs
    // TODO: this can be removed if we move to storage buffers for light arrays
    point_lights: array<PointLight, 10>;
    directional_lights: array<DirectionalLight, 1>;
    ambient_color: vec4<f32>;
    n_point_lights: u32;
    n_directional_lights: u32;
};


[[group(0), binding(1)]]
var lights: Lights;
[[group(0), binding(2)]]
var point_shadow_textures: texture_depth_cube_array;
[[group(0), binding(3)]]
var point_shadow_comparison_sampler: sampler_comparison;
[[group(0), binding(4)]]
var point_shadow_sampler: sampler;
[[group(0), binding(5)]]
var directional_shadow_textures: texture_depth_2d_array;
[[group(0), binding(6)]]
var directional_shadow_comparison_sampler: sampler_comparison;
[[group(0), binding(7)]]
var directional_shadow_sampler: sampler;
[[group(0), binding(8)]]
var blue_noise_texture: texture_2d<f32>;
[[group(0), binding(9)]]
var blue_noise_sampler: sampler;

[[group(1), binding(0)]]
var material: StandardMaterial;
[[group(1), binding(1)]]
var base_color_texture: texture_2d<f32>;
[[group(1), binding(2)]]
var base_color_sampler: sampler;
[[group(1), binding(3)]]
var emissive_texture: texture_2d<f32>;
[[group(1), binding(4)]]
var emissive_sampler: sampler;
[[group(1), binding(5)]]
var metallic_roughness_texture: texture_2d<f32>;
[[group(1), binding(6)]]
var metallic_roughness_sampler: sampler;
[[group(1), binding(7)]]
var occlusion_texture: texture_2d<f32>;
[[group(1), binding(8)]]
var occlusion_sampler: sampler;

let PI: f32 = 3.141592653589793;

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

// Use with a projection on view z to get ndc depth, or with an inverse projection
// on ndc depth to get view z
fn project_depth(projection: mat4x4<f32>, z: f32) -> f32 {
    return (projection[2][2] * z + projection[3][2])
        / (projection[2][3] * z + projection[3][3]);
}

// distanceAttenuation is simply the square falloff of light intensity
// combined with a smooth attenuation at the edge of the light radius
//
// light radius is a non-physical construct for efficiency purposes,
// because otherwise every light affects every fragment in the scene
fn getDistanceAttenuation(distanceSquare: f32, inverseRangeSquared: f32) -> f32 {
    let factor = distanceSquare * inverseRangeSquared;
    let smoothFactor = saturate(1.0 - factor * factor);
    let attenuation = smoothFactor * smoothFactor;
    return attenuation * 1.0 / max(distanceSquare, 0.0001);
}

// Normal distribution function (specular D)
// Based on https://google.github.io/filament/Filament.html#citation-walter07

// D_GGX(h,α) = α^2 / { π ((n⋅h)^2 (α2−1) + 1)^2 }

// Simple implementation, has precision problems when using fp16 instead of fp32
// see https://google.github.io/filament/Filament.html#listing_speculardfp16
fn D_GGX(roughness: f32, NoH: f32, h: vec3<f32>) -> f32 {
    let oneMinusNoHSquared = 1.0 - NoH * NoH;
    let a = NoH * roughness;
    let k = roughness / (oneMinusNoHSquared + a * a);
    let d = k * k * (1.0 / PI);
    return d;
}

// Visibility function (Specular G)
// V(v,l,a) = G(v,l,α) / { 4 (n⋅v) (n⋅l) }
// such that f_r becomes
// f_r(v,l) = D(h,α) V(v,l,α) F(v,h,f0)
// where
// V(v,l,α) = 0.5 / { n⋅l sqrt((n⋅v)^2 (1−α2) + α2) + n⋅v sqrt((n⋅l)^2 (1−α2) + α2) }
// Note the two sqrt's, that may be slow on mobile, see https://google.github.io/filament/Filament.html#listing_approximatedspecularv
fn V_SmithGGXCorrelated(roughness: f32, NoV: f32, NoL: f32) -> f32 {
    let a2 = roughness * roughness;
    let lambdaV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
    let lambdaL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
    let v = 0.5 / (lambdaV + lambdaL);
    return v;
}

// Fresnel function
// see https://google.github.io/filament/Filament.html#citation-schlick94
// F_Schlick(v,h,f_0,f_90) = f_0 + (f_90 − f_0) (1 − v⋅h)^5
fn F_Schlick_vec(f0: vec3<f32>, f90: f32, VoH: f32) -> vec3<f32> {
    // not using mix to keep the vec3 and float versions identical
    return f0 + (f90 - f0) * pow(1.0 - VoH, 5.0);
}

fn F_Schlick(f0: f32, f90: f32, VoH: f32) -> f32 {
    // not using mix to keep the vec3 and float versions identical
    return f0 + (f90 - f0) * pow(1.0 - VoH, 5.0);
}

fn fresnel(f0: vec3<f32>, LoH: f32) -> vec3<f32> {
    // f_90 suitable for ambient occlusion
    // see https://google.github.io/filament/Filament.html#lighting/occlusion
    let f90 = saturate(dot(f0, vec3<f32>(50.0 * 0.33)));
    return F_Schlick_vec(f0, f90, LoH);
}

// Specular BRDF
// https://google.github.io/filament/Filament.html#materialsystem/specularbrdf

// Cook-Torrance approximation of the microfacet model integration using Fresnel law F to model f_m
// f_r(v,l) = { D(h,α) G(v,l,α) F(v,h,f0) } / { 4 (n⋅v) (n⋅l) }
fn specular(f0: vec3<f32>, roughness: f32, h: vec3<f32>, NoV: f32, NoL: f32,
              NoH: f32, LoH: f32, specularIntensity: f32) -> vec3<f32> {
    let D = D_GGX(roughness, NoH, h);
    let V = V_SmithGGXCorrelated(roughness, NoV, NoL);
    let F = fresnel(f0, LoH);

    return (specularIntensity * D * V) * F;
}

// Diffuse BRDF
// https://google.github.io/filament/Filament.html#materialsystem/diffusebrdf
// fd(v,l) = σ/π * 1 / { |n⋅v||n⋅l| } ∫Ω D(m,α) G(v,l,m) (v⋅m) (l⋅m) dm
//
// simplest approximation
// float Fd_Lambert() {
//     return 1.0 / PI;
// }
//
// vec3 Fd = diffuseColor * Fd_Lambert();
//
// Disney approximation
// See https://google.github.io/filament/Filament.html#citation-burley12
// minimal quality difference
fn Fd_Burley(roughness: f32, NoV: f32, NoL: f32, LoH: f32) -> f32 {
    let f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    let lightScatter = F_Schlick(1.0, f90, NoL);
    let viewScatter = F_Schlick(1.0, f90, NoV);
    return lightScatter * viewScatter * (1.0 / PI);
}

// From https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
fn EnvBRDFApprox(f0: vec3<f32>, perceptual_roughness: f32, NoV: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = perceptual_roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    let AB = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * AB.x + AB.y;
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    // clamp perceptual roughness to prevent precision problems
    // According to Filament design 0.089 is recommended for mobile
    // Filament uses 0.045 for non-mobile
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

// from https://64.github.io/tonemapping/
// reinhard on RGB oversaturates colors
fn reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (1.0 + color);
}

fn reinhard_extended(color: vec3<f32>, max_white: f32) -> vec3<f32> {
    let numerator = color * (1.0f + (color / vec3<f32>(max_white * max_white)));
    return numerator / (1.0 + color);
}

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    let l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

fn reinhard_luminance(color: vec3<f32>) -> vec3<f32> {
    let l_old = luminance(color);
    let l_new = l_old / (1.0f + l_old);
    return change_luminance(color, l_new);
}

fn reinhard_extended_luminance(color: vec3<f32>, max_white_l: f32) -> vec3<f32> {
    let l_old = luminance(color);
    let numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    let l_new = numerator / (1.0f + l_old);
    return change_luminance(color, l_new);
}

var flip_z: vec4<f32> = vec4<f32>(1.0, 1.0, -1.0, 1.0);

fn point_light(
    world_position: vec3<f32>, light: PointLight, roughness: f32, NdotV: f32, N: vec3<f32>, V: vec3<f32>,
    R: vec3<f32>, F0: vec3<f32>, diffuseColor: vec3<f32>
) -> vec3<f32> {
    let light_position_world_rh = light.position_lh.xyz * flip_z.xyz;
    let frag_to_light = light_position_world_rh - world_position.xyz;
    let distance_square = dot(frag_to_light, frag_to_light);
    let rangeAttenuation =
        getDistanceAttenuation(distance_square, light.inverse_square_range);

    // Specular.
    // Representative Point Area Lights.
    // see http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p14-16
    let a = roughness;
    let centerToRay = dot(frag_to_light, R) * R - frag_to_light;
    let closestPoint = frag_to_light + centerToRay * saturate(light.radius * inverseSqrt(dot(centerToRay, centerToRay)));
    let LspecLengthInverse = inverseSqrt(dot(closestPoint, closestPoint));
    let normalizationFactor = a / saturate(a + (light.radius * 0.5 * LspecLengthInverse));
    let specularIntensity = normalizationFactor * normalizationFactor;

    var L: vec3<f32> = closestPoint * LspecLengthInverse; // normalize() equivalent?
    var H: vec3<f32> = normalize(L + V);
    var NoL: f32 = saturate(dot(N, L));
    var NoH: f32 = saturate(dot(N, H));
    var LoH: f32 = saturate(dot(L, H));

    let specular_light = specular(F0, roughness, H, NdotV, NoL, NoH, LoH, specularIntensity);

    // Diffuse.
    // Comes after specular since its NoL is used in the lighting equation.
    L = normalize(frag_to_light);
    H = normalize(L + V);
    NoL = saturate(dot(N, L));
    NoH = saturate(dot(N, H));
    LoH = saturate(dot(L, H));

    let diffuse = diffuseColor * Fd_Burley(roughness, NdotV, NoL, LoH);

    // See https://google.github.io/filament/Filament.html#mjx-eqn-pointLightLuminanceEquation
    // Lout = f(v,l) Φ / { 4 π d^2 }⟨n⋅l⟩
    // where
    // f(v,l) = (f_d(v,l) + f_r(v,l)) * light_color
    // Φ is luminous power in lumens
    // our rangeAttentuation = 1 / d^2 multiplied with an attenuation factor for smoothing at the edge of the non-physical maximum light radius

    // For a point light, luminous intensity, I, in lumens per steradian is given by:
    // I = Φ / 4 π
    // The derivation of this can be seen here: https://google.github.io/filament/Filament.html#mjx-eqn-pointLightLuminousPower

    // NOTE: light.color.rgb is premultiplied with light.intensity / 4 π (which would be the luminous intensity) on the CPU

    // TODO compensate for energy loss https://google.github.io/filament/Filament.html#materialsystem/improvingthebrdfs/energylossinspecularreflectance

    return ((diffuse + specular_light) * light.color.rgb) * (rangeAttenuation * NoL);
}

fn directional_light(light: DirectionalLight, roughness: f32, NdotV: f32, normal: vec3<f32>, view: vec3<f32>, R: vec3<f32>, F0: vec3<f32>, diffuseColor: vec3<f32>) -> vec3<f32> {
    let incident_light = light.direction_to_light.xyz;

    let half_vector = normalize(incident_light + view);
    let NoL = saturate(dot(normal, incident_light));
    let NoH = saturate(dot(normal, half_vector));
    let LoH = saturate(dot(incident_light, half_vector));

    let diffuse = diffuseColor * Fd_Burley(roughness, NdotV, NoL, LoH);
    let specularIntensity = 1.0;
    let specular_light = specular(F0, roughness, half_vector, NdotV, NoL, NoH, LoH, specularIntensity);

    return (specular_light + diffuse) * light.color.rgb * NoL;
}

// indices are ordered +X,-X,+Y,-Y,+Z,-Z as in the cube map
fn world_direction_to_cubemap_face(
    world_direction: vec3<f32>,
) -> u32 {
    let world_direction_abs = abs(world_direction);
    let distance_to_light = max(world_direction_abs.x, max(world_direction_abs.y, world_direction_abs.z));
    if (distance_to_light == world_direction_abs.x) {
        if (distance_to_light == world_direction.x) {
            // +X
            return 0u;
        } else {
            // -X
            return 1u;
        }
    } elseif (distance_to_light == world_direction_abs.y) {
        if (distance_to_light == world_direction.y) {
            // +Y
            return 2u;
        } else {
            // -Y
            return 3u;
        }
    } else {
        if (distance_to_light == world_direction.z) {
            // +Z
            return 4u;
        } else {
            // -Z
            return 5u;
        }
    }
}

// NOTE: This function is a simplification of multiplying light.inverse_view * fragment_world
//       as this is much cheaper than a 4x4 matrix multiplication. If modifications are
//       made to the light's view matrices, this needs updating to match the inverse view.
fn cubemap_fragment_world_to_light_view(
    face_index: u32,
    fragment_world: vec3<f32>,
    offset_world: vec3<f32>,
) -> vec4<f32> {
    var fragment_light_view: vec4<f32> = vec4<f32>(0.0);
    if (face_index == 0u) {
        // +X
        fragment_light_view = fragment_light_view + vec4<f32>(
            -fragment_world.z + offset_world.z,
             fragment_world.y - offset_world.y,
             fragment_world.x - offset_world.x,
             1.0
        );
    } elseif (face_index == 1u) {
        // -X
        fragment_light_view = fragment_light_view + vec4<f32>(
             fragment_world.z - offset_world.z,
             fragment_world.y - offset_world.y,
            -fragment_world.x + offset_world.x,
             1.0
        );
    } elseif (face_index == 2u) {
        // +Y
        fragment_light_view = fragment_light_view + vec4<f32>(
             fragment_world.x - offset_world.x,
            -fragment_world.z + offset_world.z,
             fragment_world.y - offset_world.y,
             1.0
        );
    } elseif (face_index == 3u) {
        // -Y
        fragment_light_view = fragment_light_view + vec4<f32>(
             fragment_world.x - offset_world.x,
             fragment_world.z - offset_world.z,
            -fragment_world.y + offset_world.y,
             1.0
        );
    } elseif (face_index == 4u) {
        // +Z
        fragment_light_view = fragment_light_view + vec4<f32>(
             fragment_world.x - offset_world.x,
             fragment_world.y - offset_world.y,
             fragment_world.z - offset_world.z,
             1.0
        );
    } elseif (face_index == 5u) {
        // -Z
        fragment_light_view = fragment_light_view + vec4<f32>(
            -fragment_world.x + offset_world.x,
             fragment_world.y - offset_world.y,
            -fragment_world.z + offset_world.z,
             1.0
        );
    }
    return fragment_light_view;
}

struct TexelCenterOutput {
    fragment_light_view: vec4<f32>;
    fragment_light_view_normal: vec4<f32>;
    fragment_light_ndc: vec3<f32>;
    fragment_shadow_map_uv: vec2<f32>;
    shadow_map_texel_center_uv: vec2<f32>;
};

fn calculate_point_light_texel_center(
    light: PointLight,
    light_to_fragment_world: vec3<f32>,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
) -> TexelCenterOutput {
    let cubemap_face_index = world_direction_to_cubemap_face(
        light_to_fragment_world
    );
    let fragment_light_view = cubemap_fragment_world_to_light_view(
        cubemap_face_index,
        fragment_world,
        light.position_lh.xyz,
    );
    let fragment_light_view_normal = normalize(cubemap_fragment_world_to_light_view(
        cubemap_face_index,
        fragment_world_normal.xyz,
        vec3<f32>(0.0),
    ));


    // Calculate the shadow map texture coordinates and fragment light ndc depth
    let fragment_light_clip = light.projection_lh * fragment_light_view;
    let fragment_light_ndc = fragment_light_clip / fragment_light_clip.w;
    let fragment_shadow_map_uv = 0.5 * fragment_light_ndc.xy + vec2<f32>(0.5);

    // Calculate the shadow map texture coordinates at the center of the shadow map texel
    // that contains the fragment_shadow_map_uv
    let shadow_map_resolution = textureDimensions(point_shadow_textures).xy;
    let shadow_map_resolution_f32 = vec2<f32>(shadow_map_resolution);
    let shadow_map_texel_center_uv =
        (floor(fragment_shadow_map_uv * shadow_map_resolution_f32)
            + vec2<f32>(0.5))
        / shadow_map_resolution_f32;

    var out: TexelCenterOutput;

    out.fragment_light_view = fragment_light_view;
    out.fragment_light_view_normal = fragment_light_view_normal;
    out.fragment_light_ndc = fragment_light_ndc.xyz;
    out.fragment_shadow_map_uv = fragment_shadow_map_uv;
    out.shadow_map_texel_center_uv = shadow_map_texel_center_uv;

    return out;
}

fn calculate_point_light_texel_center_ray_intersection(
    light: PointLight,
    shadow_map_texel_center_uv: vec2<f32>,
    fragment_light_view: vec3<f32>,
    fragment_light_view_normal: vec3<f32>
) -> vec3<f32> {
    // Generate a ray from the near to far plane
    let texel_center_light_ndc_xy = 2.0 * shadow_map_texel_center_uv - vec2<f32>(1.0);
    // NOTE: left-handed implies z is 1.0 here
    let texel_center_at_distance_one = vec3<f32>(texel_center_light_ndc_xy, 1.0);
    let texel_center_near_light_view = light.near * texel_center_at_distance_one;
    let texel_center_far_light_view = light.far * texel_center_at_distance_one;
    let ray_origin_light_view = texel_center_near_light_view;
    let ray_direction_light_view = texel_center_far_light_view - texel_center_near_light_view;

    // Calculate the intersection of the texel center ray with the plane defined by
    // the fragment light view normal and fragment light view position
    let t_hit = dot(fragment_light_view.xyz - ray_origin_light_view, fragment_light_view_normal.xyz)
        / dot(ray_direction_light_view, fragment_light_view_normal.xyz);
    let p_light_view = ray_origin_light_view + t_hit * ray_direction_light_view;

    // Calculate the projected position of the intersection point p_light_view
    let p_light_clip = light.projection_lh * vec4<f32>(p_light_view, 1.0);

    return p_light_clip.xyz / p_light_clip.w;
}

fn calculate_point_light_adaptive_epsilon(
    light: PointLight,
    light_world_direction: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    fragment_light_ndc_depth: f32
) -> f32 {
    // Adaptive Depth Bias for Soft Shadows adaptive epsilon scale factor
    // https://dspace5.zcu.cz/bitstream/11025/29520/1/Ehm.pdf
    // This avoids projective aliasing when the light direction is almost parallel to the fragment plane
    let max_adaptive_epsilon_scale = 100.0;
    let light_dir_dot_frag_normal = dot(light_world_direction, fragment_world_normal);
    let adaptive_epsilon_scale_factor = min(
        1.0 / (light_dir_dot_frag_normal * light_dir_dot_frag_normal),
        max_adaptive_epsilon_scale
    );

    // Calculate the adaptive epsilon to avoid self-shadowing
    let k = 0.0001;
    let scene_scale = 30.0;
    // left-handed infinite reverse perspective projection
    let adaptive_epsilon = -(fragment_light_ndc_depth * fragment_light_ndc_depth) * scene_scale * k * adaptive_epsilon_scale_factor
        / light.near;
    
    return adaptive_epsilon;
}

fn calculate_directional_light_texel_center(
    light: DirectionalLight,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
) -> TexelCenterOutput {
    let fragment_light_view = light.inverse_view * vec4<f32>(fragment_world, 1.0);
    let fragment_light_view_normal = normalize(light.inverse_view * vec4<f32>(fragment_world_normal, 0.0));

    // Calculate the shadow map texture coordinates and fragment light ndc depth
    let fragment_light_clip = light.projection * fragment_light_view;
    let fragment_light_ndc = fragment_light_clip / fragment_light_clip.w;
    let fragment_shadow_map_uv = vec2<f32>(0.5, -0.5) * fragment_light_ndc.xy + vec2<f32>(0.5);

    // Calculate the shadow map texture coordinates at the center of the shadow map texel
    // that contains the fragment_shadow_map_uv
    let shadow_map_resolution = textureDimensions(directional_shadow_textures);
    let shadow_map_resolution_f32 = vec2<f32>(shadow_map_resolution);
    let shadow_map_texel_center_uv =
        (floor(fragment_shadow_map_uv * shadow_map_resolution_f32)
            + vec2<f32>(0.5))
        / shadow_map_resolution_f32;

    var out: TexelCenterOutput;

    out.fragment_light_view = fragment_light_view;
    out.fragment_light_view_normal = fragment_light_view_normal;
    out.fragment_light_ndc = fragment_light_ndc.xyz;
    out.fragment_shadow_map_uv = fragment_shadow_map_uv;
    out.shadow_map_texel_center_uv = shadow_map_texel_center_uv;

    return out;
}

fn calculate_directional_light_texel_center_ray_intersection(
    light: DirectionalLight,
    shadow_map_texel_center_uv: vec2<f32>,
    fragment_light_view: vec3<f32>,
    fragment_light_view_normal: vec3<f32>
) -> vec3<f32> {
    // Generate a ray from the near to far plane
    let left_top = vec2<f32>(light.left, light.top);
    let projection_size = vec2<f32>(
        light.right - light.left,
        light.bottom - light.top
    );
    let texel_center_light_view_xy = left_top + shadow_map_texel_center_uv * projection_size;
    let ray_origin_light_view = vec3<f32>(texel_center_light_view_xy, -light.near);
    // far - near, except that as we have -z forward, we have -far - (-near) = -far + near
    let ray_direction_light_view = vec3<f32>(0.0, 0.0, -light.far + light.near);

    // Calculate the intersection of the texel center ray with the plane defined by
    // the fragment light view normal and fragment light view position
    let t_hit = dot(fragment_light_view.xyz - ray_origin_light_view, fragment_light_view_normal.xyz)
        / dot(ray_direction_light_view, fragment_light_view_normal.xyz);
    let p_light_view = ray_origin_light_view + t_hit * ray_direction_light_view;

    // Calculate the projected depth of the intersection point p_light_view
    let p_light_clip = light.projection * vec4<f32>(p_light_view, 1.0);

    return p_light_clip.xyz / p_light_clip.w;
}

fn calculate_directional_light_adaptive_epsilon(
    light: DirectionalLight,
    light_direction_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    fragment_light_ndc_depth: f32
) -> f32 {
    // Adaptive Depth Bias for Soft Shadows adaptive epsilon scale factor
    // https://dspace5.zcu.cz/bitstream/11025/29520/1/Ehm.pdf
    // This avoids projective aliasing when the light direction is almost parallel to the fragment plane
    let max_adaptive_epsilon_scale = 100.0;
    let light_dir_dot_frag_normal = dot(light_direction_world, fragment_world_normal);
    let adaptive_epsilon_scale_factor = min(
        1.0 / (light_dir_dot_frag_normal * light_dir_dot_frag_normal),
        max_adaptive_epsilon_scale
    );

    // Calculate the adaptive epsilon to avoid self-shadowing
    let k = 0.0001;
    let scene_scale = 30.0;
    let adaptive_epsilon = scene_scale * k * adaptive_epsilon_scale_factor
        // NOTE: for forward z this would be light.near - light.far
        / (light.far - light.near);
    
    return adaptive_epsilon;
}

// 0: cube sampling kernel
// 1: regular disc sampling kernel
// 2: blue noise disc sampling kernel
var point_shadow_sample_mode: u32 = 2u;

// 0: Depth / shadow map texel-scaled Normal bias, no special bias for PCF
// 1: Adaptive Depth Bias for Shadow Maps and Adaptive Depth Bias for Soft Shadows - per-sample (slow)
// 2: Adaptive Depth Bias for Shadow Maps and Adaptive Depth Bias for Soft Shadows - delta bias
var point_shadow_bias_mode: u32 = 1u;

// 0: off
// 1: Percentage-Closer Soft Shadows with a regular disc sampling kernel
// 2: Percentage-Closer Soft Shadows with a blue noise disc sampling kernel
var point_shadow_pcss_mode: u32 = 1u;

// 0: regular disc sampling kernel
// 1: blue noise disc sampling kernel
var directional_shadow_sample_mode: u32 = 0u;

// 0: Depth / shadow map texel-scaled Normal bias, and Receiver Plane Depth Bias (Isidoro06) for PCF
// 1: Adaptive Depth Bias for Shadow Maps and Adaptive Depth Bias for Soft Shadows
var directional_shadow_bias_mode: u32 = 0u;

// 0: off
// 1: Percentage-Closer Soft Shadows with a regular disc sampling kernel
// 2: Percentage-Closer Soft Shadows with a blue noise disc sampling kernel
var directional_shadow_pcss_mode: u32 = 2u;

// Filter width is used for regular sampling.
//     1u - single sample
//     3u - 3-wide regular pattern disc taking 5 samples (see pcf3 tables below)
//     5u - 5-wide regular pattern disc taking 21 samples (see pcf5 tables below)
//     7u - 7-wide regular pattern disc taking 37 samples (see pcf7 tables below)
var pcf_filter_width: u32 = 5u;
var pcss_blocker_search_filter_width: u32 = 5u;
// Sample count is used for blue noise (i.e. irregular) sampling. Use as many
// samples as you like.
var pcf_sample_count: u32 = 16u;
var pcss_blocker_search_sample_count: u32 = 16u;

// Radius of the Percentage-Closer Filter sampling region when not using
// Percentage-Closer Soft Shadows
var pcf_sampling_radius: f32 = 4.0;
var pcss_blocker_search_min_radius: f32 = 4.0;
var pcss_blocker_search_max_radius: f32 = 100.0;

var sample_offset_directions: array<vec3<f32>, 20> = array<vec3<f32>, 20>(
    vec3<f32>( 1.0,  1.0,  1.0), vec3<f32>( 1.0, -1.0,  1.0), vec3<f32>(-1.0, -1.0,  1.0), vec3<f32>(-1.0,  1.0,  1.0),
    vec3<f32>( 1.0,  1.0, -1.0), vec3<f32>( 1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0,  1.0, -1.0),
    vec3<f32>( 1.0,  1.0,  0.0), vec3<f32>( 1.0, -1.0,  0.0), vec3<f32>(-1.0, -1.0,  0.0), vec3<f32>(-1.0,  1.0,  0.0),
    vec3<f32>( 1.0,  0.0,  1.0), vec3<f32>(-1.0,  0.0,  1.0), vec3<f32>( 1.0,  0.0, -1.0), vec3<f32>(-1.0,  0.0, -1.0),
    vec3<f32>( 0.0,  1.0,  1.0), vec3<f32>( 0.0, -1.0,  1.0), vec3<f32>( 0.0, -1.0, -1.0), vec3<f32>( 0.0,  1.0, -1.0)
);

var pcf3_disc_count: u32 = 5;
var pcf3_disc: array<vec2<f32>, 5> = array<vec2<f32>, 5>(
                           vec2<f32>( 0.0,  1.0),
    vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.0,  0.0),
                           vec2<f32>( 0.0, -1.0)
);

var pcf5_disc_count: u32 = 21;
var pcf5_disc: array<vec2<f32>, 21> = array<vec2<f32>, 21>(
                           vec2<f32>(-1.0,  2.0), vec2<f32>( 0.0,  2.0), vec2<f32>( 1.0,  2.0),
    vec2<f32>(-2.0,  1.0), vec2<f32>(-1.0,  1.0), vec2<f32>( 0.0,  1.0), vec2<f32>( 1.0,  1.0), vec2<f32>( 2.0,  1.0),
    vec2<f32>(-2.0,  0.0), vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.0,  0.0), vec2<f32>( 2.0,  0.0),
    vec2<f32>(-2.0, -1.0), vec2<f32>(-1.0, -1.0), vec2<f32>( 0.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 2.0, -1.0),
                           vec2<f32>(-1.0, -2.0), vec2<f32>( 0.0, -2.0), vec2<f32>( 1.0, -2.0)
);

var pcf7_disc_count: u32 = 37;
var pcf7_disc: array<vec2<f32>, 37> = array<vec2<f32>, 37>(
                                                  vec2<f32>(-1.0,  3.0), vec2<f32>( 0.0,  3.0), vec2<f32>( 1.0,  3.0),
                           vec2<f32>(-2.0,  2.0), vec2<f32>(-1.0,  2.0), vec2<f32>( 0.0,  2.0), vec2<f32>( 1.0,  2.0), vec2<f32>( 2.0,  2.0),
    vec2<f32>(-3.0,  1.0), vec2<f32>(-2.0,  1.0), vec2<f32>(-1.0,  1.0), vec2<f32>( 0.0,  1.0), vec2<f32>( 1.0,  1.0), vec2<f32>( 2.0,  1.0), vec2<f32>( 3.0,  1.0),
    vec2<f32>(-3.0,  0.0), vec2<f32>(-2.0,  0.0), vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  0.0), vec2<f32>( 1.0,  0.0), vec2<f32>( 2.0,  0.0), vec2<f32>( 3.0,  0.0),
    vec2<f32>(-3.0, -1.0), vec2<f32>(-2.0, -1.0), vec2<f32>(-1.0, -1.0), vec2<f32>( 0.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 2.0, -1.0), vec2<f32>( 3.0, -1.0),
                           vec2<f32>(-2.0, -2.0), vec2<f32>(-1.0, -2.0), vec2<f32>( 0.0, -2.0), vec2<f32>( 1.0, -2.0), vec2<f32>( 2.0, -2.0),
                                                  vec2<f32>(-1.0, -3.0), vec2<f32>( 0.0, -3.0), vec2<f32>( 1.0, -3.0)
);

var golden_ratio: f32 = 0.618033988749895;
var tau: f32 = 6.283185307179586;

// 2x2 matrix inverse as WGSL does not have one yet
fn inverse2x2(m: mat2x2<f32>) -> mat2x2<f32> {
    let inv_det = 1.0 / determinant(m);
    return mat2x2<f32>(
        vec2<f32>( m[1][1], -m[0][1]),
        vec2<f32>(-m[1][0],  m[0][0])
    ) * inv_det;
}

// R2 given indices 0, 1, 2, ... will return values in [0.0, 1.0] that are 'maximally distant'
// from each other. Scale this by a blue noise texture size to sample a blue noise texture due
// to the way blue noise textures work (very different / very similar close for close neighbors
// and this 'rippling' drops of to 0 at about 10 samples distance but maximally distant is best.)
fn R2(index: u32) -> vec2<f32> {
    // Generalized golden ratio to 2d.
    // Solution to x^3 = x + 1
    // AKA plastic constant.
    // from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    let g = 1.32471795724474602596;
    return fract(vec2<f32>(f32(index) / g, f32(index) / (g * g)));
}

// This method is used here: https://learnopengl.com/Advanced-Lighting/Shadows/Point-Shadows
fn sample_point_shadow_pcf_cube(
    light_to_fragment_world: vec3<f32>,
    texture_index: i32,
    depth: f32,
    radius: f32
) -> f32 {
    // FIXME: Assumes square shadow map
    let shadow_map_step_texels = vec3<f32>(radius / f32(textureDimensions(point_shadow_textures).x));

    let inverse_sample_count = 1.0 / f32(20);
    var shadow: f32 = 0.0;
    for (var i: i32 = 0; i < 20; i = i + 1) {
        // 2x2 hardware bilinear-filtered PCF
        shadow = shadow + textureSampleCompareLevel(
            point_shadow_textures,
            point_shadow_comparison_sampler,
            light_to_fragment_world + sample_offset_directions[i] * shadow_map_step_texels,
            texture_index,
            depth
        ) * inverse_sample_count;
    }
    return shadow;
}

// Regular distribution of samples in a disc about the normalized surface to light vector
fn sample_point_shadow_pcf_regular_disc(
    light_to_fragment_world: vec3<f32>,
    texture_index: i32,
    depth: f32,
    right: vec3<f32>,
    up: vec3<f32>,
    radius: f32
) -> f32 {
    var is_lit: f32 = 0.0;
    if (pcf_filter_width == 7u) {
        let inverse_sample_count = 1.0 / f32(pcf7_disc_count);
        // NOTE: 7.0 - 1.0 because one sample is in the center
        let radius_step = 2.0 * radius / (7.0 - 1.0);
        for (var i: u32 = 0u; i < pcf7_disc_count; i = i + 1u) {
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                point_shadow_textures,
                point_shadow_comparison_sampler,
                light_to_fragment_world
                    + radius_step * pcf7_disc[i].x * right
                    + radius_step * pcf7_disc[i].y * up,
                texture_index,
                depth
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 5u) {
        let inverse_sample_count = 1.0 / f32(pcf5_disc_count);
        // NOTE: 5.0 - 1.0 because one sample is in the center
        let radius_step = 2.0 * radius / (5.0 - 1.0);
        for (var i: u32 = 0u; i < pcf5_disc_count; i = i + 1u) {
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                point_shadow_textures,
                point_shadow_comparison_sampler,
                light_to_fragment_world
                    + radius_step * pcf5_disc[i].x * right
                    + radius_step * pcf5_disc[i].y * up,
                texture_index,
                depth
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 3u) {
        let inverse_sample_count = 1.0 / f32(pcf3_disc_count);
        // NOTE: 3.0 - 1.0 because one sample is in the center
        let radius_step = 2.0 * radius / (3.0 - 1.0);
        for (var i: u32 = 0u; i < pcf3_disc_count; i = i + 1u) {
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                point_shadow_textures,
                point_shadow_comparison_sampler,
                light_to_fragment_world
                    + radius_step * pcf3_disc[i].x * right
                    + radius_step * pcf3_disc[i].y * up,
                texture_index,
                depth
            ) * inverse_sample_count;
        }
    } else {
        is_lit = textureSampleCompareLevel(
            point_shadow_textures,
            point_shadow_comparison_sampler,
            light_to_fragment_world,
            texture_index,
            depth
        );
    }

    return is_lit;
}

fn calculate_offset_fragment_light_view(
    fragment_world: vec3<f32>,
    light_world: vec3<f32>,
    fragment_light_view: vec3<f32>,
    cubemap_face_index: u32,
    offset_light_to_fragment_world: vec3<f32>,
) -> vec3<f32> {
    var cubemap_fragment_light_view: vec3<f32>;

    let offset_cubemap_face_index = world_direction_to_cubemap_face(offset_light_to_fragment_world);
    if (offset_cubemap_face_index == cubemap_face_index) {
        cubemap_fragment_light_view = fragment_light_view;
    } else {
        let offset_cubemap_face_fragment_light_view = cubemap_fragment_world_to_light_view(
            offset_cubemap_face_index,
            fragment_world,
            light_world,
        );
        cubemap_fragment_light_view = offset_cubemap_face_fragment_light_view.xyz;
    }

    return cubemap_fragment_light_view;
}

fn calculate_point_light_optimal_offset_fragment_depth(
    light: PointLight,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    fragment_light_view: vec3<f32>,
    cubemap_face_index: u32,
    offset_light_to_fragment_world: vec3<f32>,
    offset_fragment_world: vec3<f32>,
) -> f32 {
    let cubemap_fragment_light_view = calculate_offset_fragment_light_view(
        fragment_world,
        light.position_lh.xyz,
        fragment_light_view,
        cubemap_face_index,
        offset_light_to_fragment_world,
    );
    let texel_center = calculate_point_light_texel_center(
        light,
        offset_light_to_fragment_world,
        offset_fragment_world,
        fragment_world_normal,
    );
    let f_o = calculate_point_light_texel_center_ray_intersection(
        light,
        texel_center.shadow_map_texel_center_uv,
        cubemap_fragment_light_view,
        texel_center.fragment_light_view_normal.xyz
    );
    let optimal_fragment_depth = min(
        texel_center.fragment_light_ndc.z,
        f_o.z
    );

    return optimal_fragment_depth;
}

// Regular distribution of samples in a disc about the normalized surface to light vector
fn sample_point_shadow_pcf_regular_disc_adaptive_bias(
    light_to_fragment_world: vec3<f32>,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    fragment_light_view: vec3<f32>,
    texture_index: i32,
    depth: f32,
    dbias: vec2<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    radius: f32
) -> f32 {
    let light = lights.point_lights[texture_index];
    let cubemap_face_index = world_direction_to_cubemap_face(light_to_fragment_world);
    var is_lit: f32 = 0.0;
    if (pcf_filter_width == 7u) {
        let inverse_sample_count = 1.0 / f32(pcf7_disc_count);
        // NOTE: 7.0 - 1.0 because one sample is in the center
        let radius_step = 2.0 * radius / (7.0 - 1.0);
        for (var i: u32 = 0u; i < pcf7_disc_count; i = i + 1u) {
            let offset_texels = radius_step * pcf7_disc[i];

            let offset_fragment_world = fragment_world.xyz
                + offset_texels.x * right
                + offset_texels.y * up;
            let offset_light_to_fragment_world = offset_fragment_world - light.position_lh.xyz;

            var optimal_fragment_depth: f32;
            if (point_shadow_bias_mode == 1u) {
                optimal_fragment_depth = calculate_point_light_optimal_offset_fragment_depth(
                    light,
                    fragment_world,
                    fragment_world_normal,
                    fragment_light_view,
                    cubemap_face_index,
                    offset_light_to_fragment_world,
                    offset_fragment_world,
                );
            } else { // point_shadow_bias_mode == 2u
                optimal_fragment_depth = depth + dot(dbias, offset_texels);
            }

            let offset_light_world_direction = normalize(offset_light_to_fragment_world);
            let adaptive_epsilon = calculate_point_light_adaptive_epsilon(
                light,
                offset_light_world_direction,
                fragment_world_normal,
                optimal_fragment_depth
            );

            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                point_shadow_textures,
                point_shadow_comparison_sampler,
                offset_light_to_fragment_world,
                texture_index,
                optimal_fragment_depth - adaptive_epsilon
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 5u) {
        let inverse_sample_count = 1.0 / f32(pcf5_disc_count);
        // NOTE: 5.0 - 1.0 because one sample is in the center
        let radius_step = 2.0 * radius / (5.0 - 1.0);
        for (var i: u32 = 0u; i < pcf5_disc_count; i = i + 1u) {
            let offset_texels = radius_step * pcf5_disc[i];

            let offset_fragment_world = fragment_world
                + offset_texels.x * right
                + offset_texels.y * up;
            let offset_light_to_fragment_world = offset_fragment_world - light.position_lh.xyz;

            var optimal_fragment_depth: f32;
            if (point_shadow_bias_mode == 1u) {
                optimal_fragment_depth = calculate_point_light_optimal_offset_fragment_depth(
                    light,
                    fragment_world,
                    fragment_world_normal,
                    fragment_light_view,
                    cubemap_face_index,
                    offset_light_to_fragment_world,
                    offset_fragment_world,
                );
            } else { // point_shadow_bias_mode == 2u
                optimal_fragment_depth = depth + dot(dbias, offset_texels);
            }

            let offset_light_direction_world = normalize(offset_light_to_fragment_world);
            let adaptive_epsilon = calculate_point_light_adaptive_epsilon(
                light,
                offset_light_direction_world,
                fragment_world_normal,
                optimal_fragment_depth
            );

            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                point_shadow_textures,
                point_shadow_comparison_sampler,
                offset_light_to_fragment_world,
                texture_index,
                optimal_fragment_depth - adaptive_epsilon
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 3u) {
        let inverse_sample_count = 1.0 / f32(pcf3_disc_count);
        // NOTE: 3.0 - 1.0 because one sample is in the center
        let radius_step = 2.0 * radius / (3.0 - 1.0);
        for (var i: u32 = 0u; i < pcf3_disc_count; i = i + 1u) {
            let offset_texels = radius_step * pcf3_disc[i];

            let offset_fragment_world = fragment_world.xyz
                + offset_texels.x * right
                + offset_texels.y * up;
            let offset_light_to_fragment_world = offset_fragment_world - light.position_lh.xyz;
 
            var optimal_fragment_depth: f32;
            if (point_shadow_bias_mode == 1u) {
                optimal_fragment_depth = calculate_point_light_optimal_offset_fragment_depth(
                    light,
                    fragment_world,
                    fragment_world_normal,
                    fragment_light_view,
                    cubemap_face_index,
                    offset_light_to_fragment_world,
                    offset_fragment_world,
                );
            } else { // point_shadow_bias_mode == 2u
                optimal_fragment_depth = depth + dot(dbias, offset_texels);
            }

            let offset_light_world_direction = normalize(offset_light_to_fragment_world);
            let adaptive_epsilon = calculate_point_light_adaptive_epsilon(
                light,
                offset_light_world_direction,
                fragment_world_normal,
                optimal_fragment_depth
            );

            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                point_shadow_textures,
                point_shadow_comparison_sampler,
                offset_light_to_fragment_world,
                texture_index,
                optimal_fragment_depth - adaptive_epsilon
            ) * inverse_sample_count;
        }
    } else {
        let offset_texels = radius * vec2<f32>(0.0, -1.0);
        let offset_fragment_world = fragment_world.xyz
            + offset_texels.x * right
            + offset_texels.y * up;
        let offset_light_to_fragment_world = offset_fragment_world - light.position_lh.xyz;
        let offset_light_world_direction = normalize(offset_light_to_fragment_world);

        // let texel_center = calculate_point_light_texel_center(
        //     light,
        //     offset_light_to_fragment_world,
        //     offset_fragment_world,
        //     fragment_world_normal,
        // );
        // let f_o = calculate_point_light_texel_center_ray_intersection(
        //     light,
        //     texel_center.shadow_map_texel_center_uv,
        //     fragment_light_view,
        //     texel_center.fragment_light_view_normal.xyz
        // );
        // let optimal_fragment_depth = min(
        //     texel_center.fragment_light_ndc.z,
        //     f_o.z
        // );

        let optimal_fragment_depth = depth + dot(dbias, offset_texels);

        let light_direction_world = normalize(light_to_fragment_world);
        let adaptive_epsilon = calculate_point_light_adaptive_epsilon(
            light,
            offset_light_world_direction,
            fragment_world_normal,
            optimal_fragment_depth
        );
        // 2x2 hardware bilinear-filtered PCF
        is_lit = textureSampleCompareLevel(
            point_shadow_textures,
            point_shadow_comparison_sampler,
            offset_light_to_fragment_world,
            texture_index,
            optimal_fragment_depth - adaptive_epsilon
        );
    }

    return is_lit;
}

// Blue noise distribution of samples in a disc about the normalized surface to light vector
fn sample_point_shadow_pcf_blue_noise_disc(
    light_world_direction: vec3<f32>,
    texture_index: i32,
    depth: f32,
    frag_coord: vec2<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    radius: f32
) -> f32 {
    // tile noise texture over screen, based on screen dimensions divided by noise size
    let noise_scale = vec2<f32>(1.0) / vec2<f32>(textureDimensions(blue_noise_texture));

    // Frame variation
    // Offset the blue noise _UVs_ by the R2 sequence based on the frame number
    let base_noise_uv = frag_coord * noise_scale + R2(view.frame_number % 64u);
    // Offset the blue noise _value_ by a frame number multiple of the golden ratio
    // let frame_golden_ratio_offset = vec2<f32>(f32(view.frame_number % 64u) * golden_ratio);

    let inverse_sample_count = 1.0 / f32(pcf_sample_count);
    var is_lit: f32 = 0.0;
    for (var i: u32 = 0u; i < pcf_sample_count; i = i + 1u) {
        // Use R2 sequence for maximally-distant ideal sampling of the blue noise texture
        let blue_noise_values = textureSampleLevel(
            blue_noise_texture,
            blue_noise_sampler,
            base_noise_uv + R2(i),
            0.0
        ).xy;
        // let frame_blue_noise_values = fract(blue_noise_values + frame_golden_ratio_offset);
        let offset_texels =
            radius * sqrt(blue_noise_values.y)
            * vec2<f32>(sin(blue_noise_values.x * tau), cos(blue_noise_values.x * tau));
        // 2x2 hardware bilinear-filtered PCF
        is_lit = is_lit + textureSampleCompareLevel(
            point_shadow_textures,
            point_shadow_comparison_sampler,
            light_world_direction
                + offset_texels.x * right
                + offset_texels.y * up,
            texture_index,
            depth
        ) * inverse_sample_count;
    }
    return is_lit;
}


// Blue noise distribution of samples in a disc about the normalized surface to light vector
fn sample_point_shadow_pcf_blue_noise_disc_adaptive_bias(
    light_to_fragment_world: vec3<f32>,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    fragment_light_view: vec3<f32>,
    texture_index: i32,
    depth: f32,
    dbias: vec2<f32>,
    frag_coord: vec2<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    radius: f32
) -> f32 {
    let light = lights.point_lights[texture_index];

    // tile noise texture over screen, based on screen dimensions divided by noise size
    let noise_scale = vec2<f32>(1.0) / vec2<f32>(textureDimensions(blue_noise_texture));

    // Frame variation
    // Offset the blue noise _UVs_ by the R2 sequence based on the frame number
    let base_noise_uv = frag_coord * noise_scale + R2(view.frame_number % 64u);
    // Offset the blue noise _value_ by a frame number multiple of the golden ratio
    // let frame_golden_ratio_offset = vec2<f32>(f32(view.frame_number % 64u) * golden_ratio);

    let cubemap_face_index = world_direction_to_cubemap_face(light_to_fragment_world);
    let light_to_fragment_world_length = length(light_to_fragment_world);

    let inverse_sample_count = 1.0 / f32(pcf_sample_count);
    var is_lit: f32 = 0.0;
    for (var i: u32 = 0u; i < pcf_sample_count; i = i + 1u) {
        // Use R2 sequence for maximally-distant ideal sampling of the blue noise texture
        let blue_noise_values = textureSampleLevel(
            blue_noise_texture,
            blue_noise_sampler,
            base_noise_uv + R2(i),
            0.0
        ).xy;
        let offset_texels = radius
            * light_to_fragment_world_length
            * sqrt(blue_noise_values.y)
            * vec2<f32>(sin(blue_noise_values.x * tau), cos(blue_noise_values.x * tau));

        let offset_fragment_world = fragment_world
            + offset_texels.x * right
            + offset_texels.y * up;
        let offset_light_to_fragment_world = offset_fragment_world - light.position_lh.xyz;

        var optimal_fragment_depth: f32;
        if (point_shadow_bias_mode == 1u) {
            optimal_fragment_depth = calculate_point_light_optimal_offset_fragment_depth(
                light,
                fragment_world,
                fragment_world_normal,
                fragment_light_view,
                cubemap_face_index,
                offset_light_to_fragment_world,
                offset_fragment_world,
            );
        } else { // point_shadow_bias_mode == 2u
            optimal_fragment_depth = depth + dot(dbias, offset_texels);
        }

        let offset_light_direction_world = normalize(offset_light_to_fragment_world);
        let adaptive_epsilon = calculate_point_light_adaptive_epsilon(
            light,
            offset_light_direction_world,
            fragment_world_normal,
            optimal_fragment_depth
        );

        // 2x2 hardware bilinear-filtered PCF
        is_lit = is_lit + textureSampleCompareLevel(
            point_shadow_textures,
            point_shadow_comparison_sampler,
            offset_light_to_fragment_world,
            texture_index,
            optimal_fragment_depth - adaptive_epsilon
        ) * inverse_sample_count;
    }

    return is_lit;
}

// Blue noise distribution of samples in a disc about the normalized surface to light vector
fn sample_point_shadow_pcf_disc(
    light_to_fragment_world: vec3<f32>,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    texture_index: i32,
    depth: f32,
    radius: f32,
    frag_coord: vec2<f32>
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(point_shadow_textures).xy);
    let shadow_map_step_texel = vec2<f32>(1.0) / shadow_map_size;

    let light_world_direction = normalize(light_to_fragment_world);
    var up: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    // NOTE: left-handed
    var right: vec3<f32> = normalize(cross(up, light_world_direction));
    up = normalize(cross(light_world_direction, right));
    right = right * shadow_map_step_texel.x;
    up = up * shadow_map_step_texel.y;

    if (point_shadow_bias_mode == 0u) {
        if (point_shadow_sample_mode == 1u) {
            return sample_point_shadow_pcf_regular_disc(
                light_to_fragment_world,
                texture_index,
                depth,
                right,
                up,
                radius
            );
        } else {
            return sample_point_shadow_pcf_blue_noise_disc(
                light_world_direction,
                texture_index,
                depth,
                frag_coord,
                right,
                up,
                radius
            );
        }
    } else {
        let light = lights.point_lights[texture_index];

        // Calculate optimal fragment depth and x/y delta biases
        let texel_center = calculate_point_light_texel_center(
            light,
            light_to_fragment_world,
            fragment_world.xyz,
            fragment_world_normal,
        );
        let f_o = calculate_point_light_texel_center_ray_intersection(
            light,
            texel_center.shadow_map_texel_center_uv,
            texel_center.fragment_light_view.xyz,
            texel_center.fragment_light_view_normal.xyz
        );
        let optimal_fragment_depth = min(
            texel_center.fragment_light_ndc.z,
            f_o.z
        );

        let cubemap_face_index = world_direction_to_cubemap_face(light_to_fragment_world);

        let f_o_x_offset_fragment_world = fragment_world.xyz + right;
        let f_o_x_offset_light_to_fragment_world = f_o_x_offset_fragment_world - light.position_lh.xyz;

        let f_o_x_cubemap_fragment_light_view = calculate_offset_fragment_light_view(
            fragment_world,
            light.position_lh.xyz,
            texel_center.fragment_light_view.xyz,
            cubemap_face_index,
            f_o_x_offset_light_to_fragment_world,
        );
        let f_o_x_texel_center = calculate_point_light_texel_center(
            light,
            f_o_x_offset_light_to_fragment_world,
            f_o_x_offset_fragment_world,
            fragment_world_normal,
        );
        let f_o_x = calculate_point_light_texel_center_ray_intersection(
            light,
            f_o_x_texel_center.shadow_map_texel_center_uv,
            f_o_x_cubemap_fragment_light_view,
            f_o_x_texel_center.fragment_light_view_normal.xyz
        );

        let f_o_y_offset_fragment_world = fragment_world.xyz + up;
        let f_o_y_offset_light_to_fragment_world = f_o_y_offset_fragment_world - light.position_lh.xyz;

        let f_o_y_cubemap_fragment_light_view = calculate_offset_fragment_light_view(
            fragment_world,
            light.position_lh.xyz,
            texel_center.fragment_light_view.xyz,
            cubemap_face_index,
            f_o_y_offset_light_to_fragment_world,
        );
        let f_o_y_texel_center = calculate_point_light_texel_center(
            light,
            f_o_y_offset_light_to_fragment_world,
            f_o_y_offset_fragment_world,
            fragment_world_normal,
        );
        let f_o_y = calculate_point_light_texel_center_ray_intersection(
            light,
            f_o_y_texel_center.shadow_map_texel_center_uv,
            f_o_y_cubemap_fragment_light_view,
            f_o_y_texel_center.fragment_light_view_normal.xyz
        );

        var dbias: vec2<f32> = vec2<f32>(
            f_o_x.z - f_o.z,
            f_o_y.z - f_o.z
        );

        if (point_shadow_sample_mode == 1u) {
            return sample_point_shadow_pcf_regular_disc_adaptive_bias(
                light_to_fragment_world,
                fragment_world,
                fragment_world_normal,
                texel_center.fragment_light_view.xyz,
                texture_index,
                optimal_fragment_depth,
                dbias,
                right,
                up,
                radius
            );
        } else {
            return sample_point_shadow_pcf_blue_noise_disc_adaptive_bias(
                light_to_fragment_world,
                fragment_world,
                fragment_world_normal,
                texel_center.fragment_light_view.xyz,
                texture_index,
                optimal_fragment_depth,
                dbias,
                frag_coord,
                right,
                up,
                radius
            );
        }
    }
}

fn calculate_point_shadow_blocker_search_radius(
    light: PointLight,
    z_receiver_light_view: f32,
    shadow_map_step_texel: f32
) -> f32 {
    // Given the fragment light view depth, light near plane, and the light radius, calculate the blocker search
    // radius in UV units.

    // Use ratios of sides of similar triangles and rearrange for search radius
    // light radius / receiver depth == search radius / (receiver depth - near)
    // search radius = light radius * (receiver depth - near) / receiver depth
    let near_plane_search_radius = clamp(
        light.radius * (z_receiver_light_view - light.near)
            / z_receiver_light_view,
        pcss_blocker_search_min_radius * shadow_map_step_texel,
        pcss_blocker_search_max_radius * shadow_map_step_texel
    );

    // But we want to offset about the fragment position, so using similar triangles again:
    // search radius at receiver depth / receiver depth = search radius at near plane / near
    // search radius at receiver depth = receiver depth * search radius at near plane / near
    return z_receiver_light_view * near_plane_search_radius / light.near;
}

struct Blocker {
    count: u32;
    max: u32;
    mean_depth: f32;
};

fn find_point_shadow_blocker(
    light: PointLight,
    light_to_fragment_world: vec3<f32>,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    fragment_light_view: vec3<f32>,
    texture_index: i32,
    depth: f32,
    dbias: vec2<f32>,
    shadow_map_step_texel: vec2<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    frag_coord: vec2<f32>
) -> Blocker {
    // NOTE: fragment_light_view.z can never be closer than the near plane so this is always positive
    let blocker_search_radius = calculate_point_shadow_blocker_search_radius(
        light,
        abs(fragment_light_view.z),
        shadow_map_step_texel.x
    );
    // NOTE: shadow_map_step_texel.x is 1.0 / shadow map width in texels
    let blocker_search_radius_texels = blocker_search_radius / shadow_map_step_texel.x;

    var blocker: Blocker;
    blocker.count = 0u;

    var blocker_depth_sum: f32 = 0.0;

    if (point_shadow_pcss_mode == 1u) {
    } else { // point_shadow_pcss_mode == 2u
        blocker.max = pcss_blocker_search_sample_count;

        // tile noise texture over screen, based on screen dimensions divided by noise size
        let noise_scale = vec2<f32>(1.0) / vec2<f32>(textureDimensions(blue_noise_texture));

        // Frame variation
        // Offset the blue noise _UVs_ by the R2 sequence based on the frame number
        let base_noise_uv = frag_coord * noise_scale + R2(view.frame_number % 64u);
        // Offset the blue noise _value_ by a frame number multiple of the golden ratio
        // let frame_golden_ratio_offset = vec2<f32>(f32(view.frame_number % 64u) * golden_ratio);

        let cubemap_face_index = world_direction_to_cubemap_face(light_to_fragment_world);

        for (var i: u32 = 0u; i < pcss_blocker_search_sample_count; i = i + 1u) {
            // Use R2 sequence for maximally-distant ideal sampling of the blue noise texture
            let blue_noise_values = textureSampleLevel(
                blue_noise_texture,
                blue_noise_sampler,
                base_noise_uv + R2(i),
                0.0
            ).xy;
            let offset_texels = blocker_search_radius_texels
                * sqrt(blue_noise_values.y)
                * vec2<f32>(sin(blue_noise_values.x * tau), cos(blue_noise_values.x * tau));

            let offset_fragment_world = fragment_world
                + offset_texels.x * right
                + offset_texels.y * up;
            let offset_light_to_fragment_world = offset_fragment_world - light.position_lh.xyz;

            var optimal_fragment_depth: f32;
            if (point_shadow_bias_mode == 1u) {
                optimal_fragment_depth = calculate_point_light_optimal_offset_fragment_depth(
                    light,
                    fragment_world,
                    fragment_world_normal,
                    fragment_light_view,
                    cubemap_face_index,
                    offset_light_to_fragment_world,
                    offset_fragment_world,
                );
            } else { // point_shadow_bias_mode == 2u
                optimal_fragment_depth = depth + dot(dbias, offset_texels);
            }

            let offset_light_direction_world = normalize(offset_light_to_fragment_world);
            let adaptive_epsilon = calculate_point_light_adaptive_epsilon(
                light,
                offset_light_direction_world,
                fragment_world_normal,
                optimal_fragment_depth
            );

            let shadow_map_depth = textureSampleLevel(
                point_shadow_textures,
                point_shadow_sampler,
                offset_light_to_fragment_world,
                texture_index,
                0.0
            );
            if (shadow_map_depth > optimal_fragment_depth - adaptive_epsilon) {
                blocker.count = blocker.count + 1u;
                blocker_depth_sum = blocker_depth_sum + project_depth(light.inverse_projection_lh, shadow_map_depth);
            }
        }
    }

    if (blocker.count > 0u) {
        blocker.mean_depth = blocker_depth_sum / f32(blocker.count);
    }

    return blocker;
}

// In view/world space units
fn calculate_point_shadow_penumbra_radius(
    light: PointLight,
    z_receiver_light_view: f32,
    mean_blocker_depth: f32,
) -> f32 {
    // Apply the inverse projection to the mean blocker depth (in [0,1]) to get view space z
    // FIXME: Take mean of projected depth and unproject? Or unproject, sum, and then take mean?
    let z_blocker_light_view_abs = abs(mean_blocker_depth);// abs(project_depth(light.inverse_projection, mean_blocker_depth));
    let penumbra_light_view_radius = light.radius * ((abs(z_receiver_light_view) - z_blocker_light_view_abs)
        / z_blocker_light_view_abs);
    // Ensure the width is always positive
    return max(penumbra_light_view_radius, 0.0);
}

fn sample_point_shadow_pcss(
    light_to_fragment_world: vec3<f32>,
    fragment_world: vec3<f32>,
    fragment_world_normal: vec3<f32>,
    texture_index: i32,
    depth: f32,
    radius: f32,
    frag_coord: vec2<f32>
) -> f32 {
    let light = lights.point_lights[texture_index];

    let shadow_map_size = vec2<f32>(textureDimensions(point_shadow_textures).xy);
    let shadow_map_step_texel = vec2<f32>(1.0) / shadow_map_size;

    let light_world_direction = normalize(light_to_fragment_world);
    var up: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    // NOTE: left-handed
    var right: vec3<f32> = normalize(cross(up, light_world_direction));
    up = normalize(cross(light_world_direction, right));
    right = right * shadow_map_step_texel.x;
    up = up * shadow_map_step_texel.y;

    if (point_shadow_bias_mode == 0u) {
        // FIXME: Disable this as it doesn't make sense to try PCSS with no gradient bias
        if (point_shadow_sample_mode == 1u) {
            return sample_point_shadow_pcf_regular_disc(
                light_to_fragment_world,
                texture_index,
                depth,
                right,
                up,
                radius
            );
        } else {
            return sample_point_shadow_pcf_blue_noise_disc(
                light_world_direction,
                texture_index,
                depth,
                frag_coord,
                right,
                up,
                radius
            );
        }
    } else {
        let light = lights.point_lights[texture_index];

        // Calculate optimal fragment depth and x/y delta biases
        let texel_center = calculate_point_light_texel_center(
            light,
            light_to_fragment_world,
            fragment_world.xyz,
            fragment_world_normal,
        );
        let f_o = calculate_point_light_texel_center_ray_intersection(
            light,
            texel_center.shadow_map_texel_center_uv,
            texel_center.fragment_light_view.xyz,
            texel_center.fragment_light_view_normal.xyz
        );
        let optimal_fragment_depth = min(
            texel_center.fragment_light_ndc.z,
            f_o.z
        );

        let cubemap_face_index = world_direction_to_cubemap_face(light_to_fragment_world);

        let f_o_x_offset_fragment_world = fragment_world.xyz + right;
        let f_o_x_offset_light_to_fragment_world = f_o_x_offset_fragment_world - light.position_lh.xyz;

        let f_o_x_cubemap_fragment_light_view = calculate_offset_fragment_light_view(
            fragment_world,
            light.position_lh.xyz,
            texel_center.fragment_light_view.xyz,
            cubemap_face_index,
            f_o_x_offset_light_to_fragment_world,
        );
        let f_o_x_texel_center = calculate_point_light_texel_center(
            light,
            f_o_x_offset_light_to_fragment_world,
            f_o_x_offset_fragment_world,
            fragment_world_normal,
        );
        let f_o_x = calculate_point_light_texel_center_ray_intersection(
            light,
            f_o_x_texel_center.shadow_map_texel_center_uv,
            f_o_x_cubemap_fragment_light_view,
            f_o_x_texel_center.fragment_light_view_normal.xyz
        );

        let f_o_y_offset_fragment_world = fragment_world.xyz + up;
        let f_o_y_offset_light_to_fragment_world = f_o_y_offset_fragment_world - light.position_lh.xyz;

        let f_o_y_cubemap_fragment_light_view = calculate_offset_fragment_light_view(
            fragment_world,
            light.position_lh.xyz,
            texel_center.fragment_light_view.xyz,
            cubemap_face_index,
            f_o_y_offset_light_to_fragment_world,
        );
        let f_o_y_texel_center = calculate_point_light_texel_center(
            light,
            f_o_y_offset_light_to_fragment_world,
            f_o_y_offset_fragment_world,
            fragment_world_normal,
        );
        let f_o_y = calculate_point_light_texel_center_ray_intersection(
            light,
            f_o_y_texel_center.shadow_map_texel_center_uv,
            f_o_y_cubemap_fragment_light_view,
            f_o_y_texel_center.fragment_light_view_normal.xyz
        );

        var dbias: vec2<f32> = vec2<f32>(
            f_o_x.z - f_o.z,
            f_o_y.z - f_o.z
        );

        // PCSS blocker search
        let cubemap_face_index = world_direction_to_cubemap_face(light_to_fragment_world);
        let fragment_light_view = cubemap_fragment_world_to_light_view(
            cubemap_face_index,
            fragment_world,
            light.position_lh.xyz,
        );
        let blocker = find_point_shadow_blocker(
            light,
            light_to_fragment_world,
            fragment_world,
            fragment_world_normal,
            fragment_light_view.xyz,
            texture_index,
            optimal_fragment_depth,
            dbias,
            shadow_map_step_texel,
            right,
            up,
            frag_coord
        );

        if (blocker.count == 0u) {
            // PCSS early-out if no blockers are found
            return 1.0;
        }
        if (blocker.count == blocker.max) {
            // PCSS early-out if fully-blocked and so in the umbra
            return 0.0;
        }

        let penumbra_light_view_radius = calculate_point_shadow_penumbra_radius(
            light,
            fragment_light_view.z,
            blocker.mean_depth
        );

        // // NOTE: shadow_map_step_texel.x is 1.0 / shadow map width in texels
        let penumbra_radius_texels = penumbra_light_view_radius / shadow_map_step_texel.x;

        if (point_shadow_sample_mode == 1u) {
            return sample_point_shadow_pcf_regular_disc_adaptive_bias(
                light_to_fragment_world,
                fragment_world,
                fragment_world_normal,
                texel_center.fragment_light_view.xyz,
                texture_index,
                optimal_fragment_depth,
                dbias,
                right,
                up,
                penumbra_radius_texels
            );
        } else {
            return sample_point_shadow_pcf_blue_noise_disc_adaptive_bias(
                light_to_fragment_world,
                fragment_world,
                fragment_world_normal,
                texel_center.fragment_light_view.xyz,
                texture_index,
                optimal_fragment_depth,
                dbias,
                frag_coord,
                right,
                up,
                penumbra_radius_texels
            );
        }
    }
}

// NOTE: All cubemap processing is done using left-handed coordinates and transformations!
fn fetch_point_shadow(
    light_id: i32,
    fragment_world_rh: vec4<f32>,
    fragment_world_normal_rh: vec3<f32>,
    frag_coord: vec2<f32>
) -> f32 {
    // NOTE: Convert right-handed y-up world coordinates to left-handed y-up
    let fragment_world_lh = fragment_world_rh * flip_z;
    let fragment_world_normal_lh = fragment_world_normal_rh * flip_z.xyz;

    let light = lights.point_lights[light_id];

    // because the shadow maps align with the axes and the frustum planes are at 45 degrees
    // we can get the worldspace depth by taking the largest absolute axis
    var light_to_fragment_world_lh: vec3<f32> = fragment_world_lh.xyz - light.position_lh.xyz;
    // Full shadow beyond the range of the light
    if (dot(light_to_fragment_world_lh, light_to_fragment_world_lh) > light.far * light.far) {
        return 0.0;
    }
    let light_to_fragment_world_abs_lh = abs(light_to_fragment_world_lh);
    var major_axis_magnitude: f32 = max(
        light_to_fragment_world_abs_lh.x,
        max(
            light_to_fragment_world_abs_lh.y,
            light_to_fragment_world_abs_lh.z
        )
    );

    if (point_shadow_bias_mode == 0u) {
        // The normal bias here is already scaled by the texel size at 1 world unit from the light.
        // The texel size increases proportionally with distance from the light so multiplying by
        // distance to light scales the normal bias to the texel size at the fragment distance.
        let normal_offset = light.shadow_normal_bias * major_axis_magnitude * fragment_world_normal_lh.xyz;
        let depth_offset = light.shadow_depth_bias * normalize(-light_to_fragment_world_lh.xyz);
        let offset_position = fragment_world_lh.xyz + normal_offset + depth_offset;

        // similar largest-absolute-axis trick as above, but now with the offset fragment position
        light_to_fragment_world_lh = offset_position.xyz - light.position_lh.xyz;
        let abs_position_ls_lh = abs(light_to_fragment_world_lh);
        major_axis_magnitude = max(abs_position_ls_lh.x, max(abs_position_ls_lh.y, abs_position_ls_lh.z));
    }

    let depth = project_depth(light.projection_lh, major_axis_magnitude);

    if (point_shadow_pcss_mode == 0u) {
        if (point_shadow_sample_mode == 0u) {
            return sample_point_shadow_pcf_cube(
                light_to_fragment_world_lh,
                i32(light_id),
                depth,
                pcf_sampling_radius
            );
        } else {
            return sample_point_shadow_pcf_disc(
                light_to_fragment_world_lh,
                fragment_world_lh.xyz,
                fragment_world_normal_lh,
                i32(light_id),
                depth,
                pcf_sampling_radius,
                frag_coord
            );
        }
    } else {
        if (point_shadow_sample_mode == 0u) {
            return sample_point_shadow_pcf_cube(
                light_to_fragment_world_lh,
                i32(light_id),
                depth,
                pcf_sampling_radius
            );
        } else {
            return sample_point_shadow_pcss(
                light_to_fragment_world_lh,
                fragment_world_lh.xyz,
                fragment_world_normal_lh,
                i32(light_id),
                depth,
                pcf_sampling_radius,
                frag_coord
            );
        }
    }
}

fn sample_directional_shadow_pcf_regular_disc(
    fragment_shadow_map_uv: vec2<f32>,
    texture_index: i32,
    depth: f32,
    ddepth_duv: vec2<f32>,
    shadow_map_step_texels: vec2<f32>
) -> f32 {
    var is_lit: f32 = 0.0;
    if (pcf_filter_width == 7u) {
        let inverse_sample_count = 1.0 / f32(pcf7_disc_count);
        // NOTE: 7.0 - 1.0 because one sample is in the center
        let radius_step = shadow_map_step_texels * (2.0 / (7.0 - 1.0));
        for (var i: u32 = 0u; i < pcf7_disc_count; i = i + 1u) {
            let uv_offset = radius_step * pcf7_disc[i];
            let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                directional_shadow_textures,
                directional_shadow_comparison_sampler,
                fragment_shadow_map_uv + uv_offset,
                texture_index,
                depth + receiver_plane_depth_bias
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 5u) {
        let inverse_sample_count = 1.0 / f32(pcf5_disc_count);
        // NOTE: 5.0 - 1.0 because one sample is in the center
        let radius_step = shadow_map_step_texels * (2.0 / (5.0 - 1.0));
        for (var i: u32 = 0u; i < pcf5_disc_count; i = i + 1u) {
            let uv_offset = radius_step * pcf5_disc[i];
            let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                directional_shadow_textures,
                directional_shadow_comparison_sampler,
                fragment_shadow_map_uv + uv_offset,
                texture_index,
                depth + receiver_plane_depth_bias
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 3u) {
        let inverse_sample_count = 1.0 / f32(pcf3_disc_count);
        // NOTE: 3.0 - 1.0 because one sample is in the center
        let radius_step = shadow_map_step_texels * (2.0 / (3.0 - 1.0));
        for (var i: u32 = 0u; i < pcf3_disc_count; i = i + 1u) {
            let uv_offset = radius_step * pcf3_disc[i];
            let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                directional_shadow_textures,
                directional_shadow_comparison_sampler,
                fragment_shadow_map_uv + uv_offset,
                texture_index,
                depth + receiver_plane_depth_bias
            ) * inverse_sample_count;
        }
    } else {
        // 2x2 hardware bilinear-filtered PCF
        is_lit = textureSampleCompareLevel(
            directional_shadow_textures,
            directional_shadow_comparison_sampler,
            fragment_shadow_map_uv,
            texture_index,
            depth
        );
    }

    return is_lit;
}

fn sample_directional_shadow_pcf_blue_noise_disc(
    fragment_shadow_map_uv: vec2<f32>,
    texture_index: i32,
    depth: f32,
    ddepth_duv: vec2<f32>,
    shadow_map_step_texels: vec2<f32>,
    frag_coord: vec2<f32>
) -> f32 {
    // tile noise texture over screen, based on screen dimensions divided by noise size
    let noise_scale = vec2<f32>(1.0) / vec2<f32>(textureDimensions(blue_noise_texture));

    // Frame variation
    // Offset the blue noise _UVs_ by the R2 sequence based on the frame number
    let base_noise_uv = frag_coord * noise_scale + R2(view.frame_number % 64u);
    // Offset the blue noise _value_ by a frame number multiple of the golden ratio
    // let frame_golden_ratio_offset = vec2<f32>(f32(view.frame_number % 64u) * golden_ratio);

    let inverse_sample_count = 1.0 / f32(pcf_sample_count);
    var is_lit: f32 = 0.0;
    for (var i: u32 = 0u; i < pcf_sample_count; i = i + 1u) {
        // Use R2 sequence for maximally-distant ideal sampling of the blue noise texture
        let blue_noise_values = textureSampleLevel(
            blue_noise_texture,
            blue_noise_sampler,
            base_noise_uv + R2(i),
            0.0
        ).xy;
        // let frame_blue_noise_values = fract(blue_noise_values + frame_golden_ratio_offset);
        let uv_offset =
            sqrt(blue_noise_values.y)
            * vec2<f32>(sin(blue_noise_values.x * tau), cos(blue_noise_values.x * tau))
            * shadow_map_step_texels;
        let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
        // 2x2 hardware bilinear-filtered PCF
        is_lit = is_lit + textureSampleCompareLevel(
            directional_shadow_textures,
            directional_shadow_comparison_sampler,
            fragment_shadow_map_uv + uv_offset,
            texture_index,
            depth + receiver_plane_depth_bias
        ) * inverse_sample_count;
    }
    return is_lit;
}

fn sample_directional_shadow_pcf_regular_disc_adaptive_bias(
    fragment_shadow_map_uv: vec2<f32>,
    fragment_world_normal: vec3<f32>,
    texture_index: i32,
    depth: f32,
    dbias: vec2<f32>,
    shadow_map_step_texels: vec2<f32>
) -> f32 {
    let light = lights.directional_lights[texture_index];
    let light_direction_world = normalize(-light.direction_to_light);

    var is_lit: f32 = 0.0;
    if (pcf_filter_width == 7u) {
        // NOTE: 7.0 - 1.0 because one sample is in the center
        let radius_step = shadow_map_step_texels * (2.0 / (7.0 - 1.0));
        let inverse_sample_count = 1.0 / f32(pcf7_disc_count);
        for (var i: u32 = 0u; i < pcf7_disc_count; i = i + 1u) {
            let optimal_fragment_depth = depth + dot(dbias, pcf7_disc[i]);
            let adaptive_epsilon = calculate_directional_light_adaptive_epsilon(
                light,
                light_direction_world,
                fragment_world_normal,
                optimal_fragment_depth
            );
            let uv_offset = radius_step * pcf7_disc[i];
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                directional_shadow_textures,
                directional_shadow_comparison_sampler,
                fragment_shadow_map_uv + uv_offset,
                texture_index,
                optimal_fragment_depth + adaptive_epsilon
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 5u) {
        // NOTE: 5.0 - 1.0 because one sample is in the center
        let radius_step = shadow_map_step_texels * (2.0 / (5.0 - 1.0));
        let inverse_sample_count = 1.0 / f32(pcf5_disc_count);
        for (var i: u32 = 0u; i < pcf5_disc_count; i = i + 1u) {
            let optimal_fragment_depth = depth + dot(dbias, pcf5_disc[i]);
            let adaptive_epsilon = calculate_directional_light_adaptive_epsilon(
                light,
                light_direction_world,
                fragment_world_normal,
                optimal_fragment_depth
            );
            let uv_offset = radius_step * pcf5_disc[i];
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                directional_shadow_textures,
                directional_shadow_comparison_sampler,
                fragment_shadow_map_uv + uv_offset,
                texture_index,
                optimal_fragment_depth + adaptive_epsilon
            ) * inverse_sample_count;
        }
    } elseif (pcf_filter_width == 3u) {
        // NOTE: 3.0 - 1.0 because one sample is in the center
        let radius_step = shadow_map_step_texels * (2.0 / (3.0 - 1.0));
        let inverse_sample_count = 1.0 / f32(pcf3_disc_count);
        for (var i: u32 = 0u; i < pcf3_disc_count; i = i + 1u) {
            let optimal_fragment_depth = depth + dot(dbias, pcf3_disc[i]);
            let adaptive_epsilon = calculate_directional_light_adaptive_epsilon(
                light,
                light_direction_world,
                fragment_world_normal,
                optimal_fragment_depth
            );
            let uv_offset = radius_step * pcf3_disc[i];
            // 2x2 hardware bilinear-filtered PCF
            is_lit = is_lit + textureSampleCompareLevel(
                directional_shadow_textures,
                directional_shadow_comparison_sampler,
                fragment_shadow_map_uv + uv_offset,
                texture_index,
                optimal_fragment_depth + adaptive_epsilon
            ) * inverse_sample_count;
        }
    } else {
        let optimal_fragment_depth = depth;
        let adaptive_epsilon = calculate_directional_light_adaptive_epsilon(
            light,
            light_direction_world,
            fragment_world_normal,
            optimal_fragment_depth
        );
        // 2x2 hardware bilinear-filtered PCF
        is_lit = textureSampleCompareLevel(
            directional_shadow_textures,
            directional_shadow_comparison_sampler,
            fragment_shadow_map_uv,
            texture_index,
            optimal_fragment_depth + adaptive_epsilon
        );
    }

    return is_lit;
}

fn sample_directional_shadow_pcf_blue_noise_disc_adaptive_bias(
    fragment_shadow_map_uv: vec2<f32>,
    fragment_world_normal: vec3<f32>,
    texture_index: i32,
    depth: f32,
    dbias: vec2<f32>,
    shadow_map_step_texels: vec2<f32>,
    frag_coord: vec2<f32>,
) -> f32 {
    let light = lights.directional_lights[texture_index];
    let light_direction_world = normalize(-light.direction_to_light);

    // tile noise texture over screen, based on screen dimensions divided by noise size
    let noise_scale = vec2<f32>(1.0) / vec2<f32>(textureDimensions(blue_noise_texture));

    // Frame variation
    // Offset the blue noise _UVs_ by the R2 sequence based on the frame number
    let base_noise_uv = frag_coord * noise_scale + R2(view.frame_number % 64u);
    // Offset the blue noise _value_ by a frame number multiple of the golden ratio
    // let frame_golden_ratio_offset = vec2<f32>(f32(view.frame_number % 64u) * golden_ratio);

    let inverse_sample_count = 1.0 / f32(pcf_sample_count);
    var is_lit: f32 = 0.0;
    for (var i: u32 = 0u; i < pcf_sample_count; i = i + 1u) {
        // Use R2 sequence for maximally-distant ideal sampling of the blue noise texture
        let blue_noise_values = textureSampleLevel(
            blue_noise_texture,
            blue_noise_sampler,
            base_noise_uv + R2(i),
            0.0
        ).xy;
        // let frame_blue_noise_values = fract(blue_noise_values + frame_golden_ratio_offset);
        let uv_offset =
            sqrt(blue_noise_values.y)
            * vec2<f32>(sin(blue_noise_values.x * tau), cos(blue_noise_values.x * tau))
            * shadow_map_step_texels;
        let optimal_fragment_depth = depth + dot(dbias, uv_offset);
        let adaptive_epsilon = calculate_directional_light_adaptive_epsilon(
            light,
            light_direction_world,
            fragment_world_normal,
            optimal_fragment_depth
        );
        // 2x2 hardware bilinear-filtered PCF
        is_lit = is_lit + textureSampleCompareLevel(
            directional_shadow_textures,
            directional_shadow_comparison_sampler,
            fragment_shadow_map_uv + uv_offset,
            texture_index,
            optimal_fragment_depth + adaptive_epsilon
        ) * inverse_sample_count;
    }
    return is_lit;
}

fn calculate_directional_shadow_blocker_search_radius(
    light: DirectionalLight,
    z_receiver_light_view: f32,
    shadow_map_step_texel: f32
) -> f32 {
    // // https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    // let distance_to_sun = 149600000000.0; // ~149.6 million km
    // // alpha = ~1919 arcseconds diameter = 1919/3600 degrees = (1919 / 3600) * (pi / 180) radians
    // // radius = d * tan(0.5 * alpha) ~ 696000000m
    // // let radius_of_sun = distance_to_sun * tan(0.5 * (1919.0 / 3600.0) * (tau / 360.0));
    // let radius_of_sun = 696000000.0;
    // // let blocker_search_radius_view = 2.0 * radius_of_sun * ((distance_to_sun - fragment_light_view.z) - (distance_to_sun + light.near))
    // let blocker_search_radius_view = 2.0 * radius_of_sun * (-fragment_light_view.z - light.near)
    //     / (distance_to_sun - fragment_light_view.z);
    return clamp(
        z_receiver_light_view * light.tan_half_light_angle * abs(light.far - light.near)
            / 200.0,
        pcss_blocker_search_min_radius * 4.0 * shadow_map_step_texel,
        pcss_blocker_search_max_radius * 4.0 * shadow_map_step_texel
    );
}

fn find_directional_shadow_blocker(
    light: DirectionalLight,
    fragment_light_view: vec3<f32>,
    fragment_shadow_map_uv: vec2<f32>,
    texture_index: i32,
    depth: f32,
    ddepth_duv: vec2<f32>,
    shadow_map_step_texel: vec2<f32>,
    frag_coord: vec2<f32>
) -> Blocker {
    let blocker_search_radius = calculate_directional_shadow_blocker_search_radius(
        light,
        abs(fragment_light_view.z),
        shadow_map_step_texel.x
    );
    // NOTE: shadow_map_step_texel.x is 1.0 / shadow map width in texels
    let blocker_search_radius_texels = blocker_search_radius / shadow_map_step_texel.x;
    let shadow_map_step_texels = blocker_search_radius_texels * shadow_map_step_texel;

    var blocker: Blocker;
    blocker.count = 0u;

    var blocker_depth_sum: f32 = 0.0;

    if (directional_shadow_pcss_mode == 1u) {
        if (pcss_blocker_search_filter_width == 7u) {
            blocker.max = pcf7_disc_count;
            // NOTE: 7.0 - 1.0 because one sample is in the center
            let radius_step = shadow_map_step_texels * (2.0 / (7.0 - 1.0));
            for (var i: u32 = 0u; i < pcf7_disc_count; i = i + 1u) {
                let uv_offset = radius_step * pcf7_disc[i];
                let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
                let shadow_map_depth = textureSampleLevel(
                    directional_shadow_textures,
                    directional_shadow_sampler,
                    fragment_shadow_map_uv + uv_offset,
                    texture_index,
                    0.0
                );
                if (shadow_map_depth > depth + receiver_plane_depth_bias) {
                    blocker.count = blocker.count + 1u;
                    blocker_depth_sum = blocker_depth_sum
                        + project_depth(light.inverse_projection, shadow_map_depth);
                }
            }
        } elseif (pcss_blocker_search_filter_width == 5u) {
            blocker.max = pcf5_disc_count;
            // NOTE: 5.0 - 1.0 because one sample is in the center
            let radius_step = shadow_map_step_texels * (2.0 / (5.0 - 1.0));
            for (var i: u32 = 0u; i < pcf5_disc_count; i = i + 1u) {
                let uv_offset = radius_step * pcf5_disc[i];
                let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
                let offset_shadow_map_uv = fragment_shadow_map_uv + uv_offset;
                if (any(clamp(offset_shadow_map_uv, vec2<f32>(0.0), vec2<f32>(1.0)) != offset_shadow_map_uv)) {
                    continue;
                }
                let shadow_map_depth = textureSampleLevel(
                    directional_shadow_textures,
                    directional_shadow_sampler,
                    offset_shadow_map_uv,
                    texture_index,
                    0.0
                );
                if (shadow_map_depth > depth + receiver_plane_depth_bias) {
                    blocker.count = blocker.count + 1u;
                    blocker_depth_sum = blocker_depth_sum
                        + project_depth(light.inverse_projection, shadow_map_depth);
                }
            }
        } elseif (pcss_blocker_search_filter_width == 3u) {
            blocker.max = pcf3_disc_count;
            // NOTE: 3.0 - 1.0 because one sample is in the center
            let radius_step = shadow_map_step_texels * (2.0 / (3.0 - 1.0));
            for (var i: u32 = 0u; i < pcf3_disc_count; i = i + 1u) {
                let uv_offset = radius_step * pcf3_disc[i];
                let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
                let shadow_map_depth = textureSampleLevel(
                    directional_shadow_textures,
                    directional_shadow_sampler,
                    fragment_shadow_map_uv + uv_offset,
                    texture_index,
                    0.0
                );
                if (shadow_map_depth > depth + receiver_plane_depth_bias) {
                    blocker.count = blocker.count + 1u;
                    blocker_depth_sum = blocker_depth_sum
                        + project_depth(light.inverse_projection, shadow_map_depth);
                }
            }
        } else {
            blocker.max = 1u;
            let shadow_map_depth = textureSampleLevel(
                directional_shadow_textures,
                directional_shadow_sampler,
                fragment_shadow_map_uv,
                texture_index,
                depth
            );
            if (shadow_map_depth > depth) {
                blocker.count = blocker.count + 1u;
                blocker_depth_sum = blocker_depth_sum
                    + project_depth(light.inverse_projection, shadow_map_depth);
            }
        }
    } else { // directional_shadow_pcss_mode == 2u
        blocker.max = pcss_blocker_search_sample_count;
        // tile noise texture over screen, based on screen dimensions divided by noise size
        let noise_scale = vec2<f32>(1.0) / vec2<f32>(textureDimensions(blue_noise_texture));

        // Frame variation
        // Offset the blue noise _UVs_ by the R2 sequence based on the frame number
        let base_noise_uv = frag_coord * noise_scale + R2(view.frame_number % 64u);
        // Offset the blue noise _value_ by a frame number multiple of the golden ratio
        // let frame_golden_ratio_offset = vec2<f32>(f32(view.frame_number % 64u) * golden_ratio);

        for (var i: u32 = 0u; i < pcss_blocker_search_sample_count; i = i + 1u) {
            // Use R2 sequence for maximally-distant ideal sampling of the blue noise texture
            var blue_noise_values: vec2<f32> = textureSampleLevel(
                blue_noise_texture,
                blue_noise_sampler,
                base_noise_uv + R2(i),
                0.0
            ).xy;
            // blue_noise_values = fract(blue_noise_values + frame_golden_ratio_offset);
            let uv_offset =
                sqrt(blue_noise_values.y)
                * vec2<f32>(sin(blue_noise_values.x * tau), cos(blue_noise_values.x * tau))
                * shadow_map_step_texels;
            let receiver_plane_depth_bias = dot(uv_offset, ddepth_duv);
            let offset_shadow_map_uv = fragment_shadow_map_uv + uv_offset;
            if (any(clamp(offset_shadow_map_uv, vec2<f32>(0.0), vec2<f32>(1.0)) != offset_shadow_map_uv)) {
                continue;
            }
            let shadow_map_depth = textureSampleLevel(
                directional_shadow_textures,
                directional_shadow_sampler,
                offset_shadow_map_uv,
                texture_index,
                0.0
            );
            if (shadow_map_depth > depth + receiver_plane_depth_bias) {
                blocker.count = blocker.count + 1u;
                blocker_depth_sum = blocker_depth_sum
                    + project_depth(light.inverse_projection, shadow_map_depth);
            }
        }
    }

    if (blocker.count > 0u) {
        blocker.mean_depth = blocker_depth_sum / f32(blocker.count);
    }

    return blocker;
}

// In light view (i.e. world) units
fn calculate_directional_shadow_penumbra_radius(
    light: DirectionalLight,
    z_receiver_light_view: f32,
    mean_blocker_depth: f32,
) -> f32 {
    return (z_receiver_light_view - mean_blocker_depth)
        * light.tan_half_light_angle * abs(light.far - light.near);
        // / 200.0;
}

fn sample_directional_shadow_pcss(
    fragment_shadow_map_uv: vec2<f32>,
    fragment_world: vec4<f32>,
    fragment_world_normal: vec3<f32>,
    texture_index: i32,
    depth: f32,
    radius: f32,
    frag_coord: vec2<f32>,
) -> f32 {
    let light = lights.directional_lights[texture_index];

    // Receiver Plane Depth Bias
    // from https://developer.amd.com/wordpress/media/2012/10/Isidoro-ShadowMapping.pdf
    let uv_jacobian = mat2x2<f32>(dpdx(fragment_shadow_map_uv), dpdy(fragment_shadow_map_uv));
    let ddepth_dscreenuv = vec2<f32>(dpdx(depth), dpdy(depth));
    let ddepth_duv = transpose(inverse2x2(uv_jacobian)) * ddepth_dscreenuv;

    let shadow_map_size = vec2<f32>(textureDimensions(directional_shadow_textures));
    let shadow_map_step_texel = vec2<f32>(1.0) / shadow_map_size;

    // PCSS blocker search
    let fragment_light_view = light.inverse_view * fragment_world;
    let blocker = find_directional_shadow_blocker(
        light,
        fragment_light_view.xyz,
        fragment_shadow_map_uv,
        texture_index,
        depth,
        ddepth_duv,
        shadow_map_step_texel,
        frag_coord
    );

    // if (blocker.count == 0u) {
    //     // PCSS early-out if no blockers are found
    //     return 1.0;
    // }
    // if (blocker.count == blocker.max) {
    //     // PCSS early-out if in full shadow (umbra)
    //     return 0.0;
    // }

    return f32(blocker.max - blocker.count) / f32(blocker.max);
    // return blocker.mean_depth / 100.0;

    // let penumbra_light_view_radius = calculate_directional_shadow_penumbra_radius(
    //     light,
    //     abs(fragment_light_view.z),
    //     blocker.mean_depth
    // );
    // return penumbra_light_view_radius;// / 40.0;
    // let penumbra_radius_texels = penumbra_light_view_radius / shadow_map_step_texel.x;
    // // let shadow_map_step_texels = vec2<f32>(penumbra_radius_texels * shadow_map_step_texel);
    // // let shadow_map_step_texels = vec2<f32>(penumbra_radius_texels);
    // let shadow_map_step_texels = vec2<f32>(penumbra_light_view_radius);

    // if (directional_shadow_bias_mode == 0u) {
    //     if (directional_shadow_sample_mode == 0u) {
    //         return sample_directional_shadow_pcf_regular_disc(
    //             fragment_shadow_map_uv,
    //             texture_index,
    //             depth,
    //             ddepth_duv,
    //             shadow_map_step_texels
    //         );
    //     } else {
    //         return sample_directional_shadow_pcf_blue_noise_disc(
    //             fragment_shadow_map_uv,
    //             texture_index,
    //             depth,
    //             ddepth_duv,
    //             shadow_map_step_texels,
    //             frag_coord
    //         );
    //     }
    // } else {
    //     let light = lights.directional_lights[texture_index];

    //     // Calculate optimal fragment depth and x/y delta biases
    //     let texel_center = calculate_directional_light_texel_center(
    //         light,
    //         fragment_world.xyz,
    //         fragment_world_normal,
    //     );

    //     let f_o = calculate_directional_light_texel_center_ray_intersection(
    //         light,
    //         texel_center.shadow_map_texel_center_uv,
    //         texel_center.fragment_light_view.xyz,
    //         texel_center.fragment_light_view_normal.xyz
    //     );
    //     let optimal_fragment_depth = min(
    //         texel_center.fragment_light_ndc.z,
    //         f_o.z
    //     );
    //     let f_o_x = calculate_directional_light_texel_center_ray_intersection(
    //         light,
    //         texel_center.shadow_map_texel_center_uv + vec2<f32>(shadow_map_step_texel.x, 0.0),
    //         texel_center.fragment_light_view.xyz,
    //         texel_center.fragment_light_view_normal.xyz
    //     );
    //     let f_o_y = calculate_directional_light_texel_center_ray_intersection(
    //         light,
    //         texel_center.shadow_map_texel_center_uv + vec2<f32>(0.0, shadow_map_step_texel.y),
    //         texel_center.fragment_light_view.xyz,
    //         texel_center.fragment_light_view_normal.xyz
    //     );

    //     let dbias = vec2<f32>(
    //         f_o_x.z - f_o.z,
    //         f_o_y.z - f_o.z
    //     );

    //     if (directional_shadow_sample_mode == 0u) {
    //         return sample_directional_shadow_pcf_regular_disc_adaptive_bias(
    //             fragment_shadow_map_uv,
    //             fragment_world_normal,
    //             texture_index,
    //             optimal_fragment_depth,
    //             dbias,
    //             shadow_map_step_texels
    //         );
    //     } else {
    //         return sample_directional_shadow_pcf_blue_noise_disc_adaptive_bias(
    //             fragment_shadow_map_uv,
    //             fragment_world_normal,
    //             texture_index,
    //             optimal_fragment_depth,
    //             dbias,
    //             shadow_map_step_texels,
    //             frag_coord
    //         );
    //     }
    // }
}

fn sample_directional_shadow_pcf_disc(
    fragment_shadow_map_uv: vec2<f32>,
    fragment_world: vec4<f32>,
    fragment_world_normal: vec3<f32>,
    texture_index: i32,
    depth: f32,
    radius: f32,
    frag_coord: vec2<f32>,
) -> f32 {
    // Receiver Plane Depth Bias
    // from https://developer.amd.com/wordpress/media/2012/10/Isidoro-ShadowMapping.pdf
    let uv_jacobian = mat2x2<f32>(dpdx(fragment_shadow_map_uv), dpdy(fragment_shadow_map_uv));
    let ddepth_dscreenuv = vec2<f32>(dpdx(depth), dpdy(depth));
    let ddepth_duv = transpose(inverse2x2(uv_jacobian)) * ddepth_dscreenuv;

    let shadow_map_size = vec2<f32>(textureDimensions(directional_shadow_textures));
    let shadow_map_step_texel = vec2<f32>(1.0) / shadow_map_size;
    let shadow_map_step_texels = vec2<f32>(radius * shadow_map_step_texel);

    if (directional_shadow_bias_mode == 0u) {
        if (directional_shadow_sample_mode == 0u) {
            return sample_directional_shadow_pcf_regular_disc(
                fragment_shadow_map_uv,
                texture_index,
                depth,
                ddepth_duv,
                shadow_map_step_texels
            );
        } else {
            return sample_directional_shadow_pcf_blue_noise_disc(
                fragment_shadow_map_uv,
                texture_index,
                depth,
                ddepth_duv,
                shadow_map_step_texels,
                frag_coord
            );
        }
    } else {
        let light = lights.directional_lights[texture_index];

        // Calculate optimal fragment depth and x/y delta biases
        let texel_center = calculate_directional_light_texel_center(
            light,
            fragment_world.xyz,
            fragment_world_normal,
        );

        let f_o = calculate_directional_light_texel_center_ray_intersection(
            light,
            texel_center.shadow_map_texel_center_uv,
            texel_center.fragment_light_view.xyz,
            texel_center.fragment_light_view_normal.xyz
        );
        let optimal_fragment_depth = min(
            texel_center.fragment_light_ndc.z,
            f_o.z
        );
        let f_o_x = calculate_directional_light_texel_center_ray_intersection(
            light,
            texel_center.shadow_map_texel_center_uv + vec2<f32>(shadow_map_step_texel.x, 0.0),
            texel_center.fragment_light_view.xyz,
            texel_center.fragment_light_view_normal.xyz
        );
        let f_o_y = calculate_directional_light_texel_center_ray_intersection(
            light,
            texel_center.shadow_map_texel_center_uv + vec2<f32>(0.0, shadow_map_step_texel.y),
            texel_center.fragment_light_view.xyz,
            texel_center.fragment_light_view_normal.xyz
        );

        let dbias = vec2<f32>(
            f_o_x.z - f_o.z,
            f_o_y.z - f_o.z
        );

        if (directional_shadow_sample_mode == 0u) {
            return sample_directional_shadow_pcf_regular_disc_adaptive_bias(
                fragment_shadow_map_uv,
                fragment_world_normal,
                texture_index,
                optimal_fragment_depth,
                dbias,
                shadow_map_step_texels
            );
        } else {
            return sample_directional_shadow_pcf_blue_noise_disc_adaptive_bias(
                fragment_shadow_map_uv,
                fragment_world_normal,
                texture_index,
                optimal_fragment_depth,
                dbias,
                shadow_map_step_texels,
                frag_coord
            );
        }
    }
}

fn fetch_directional_shadow(
    light_id: i32,
    frag_position: vec4<f32>,
    surface_normal: vec3<f32>,
    frag_coord: vec2<f32>
) -> f32 {
    let light = lights.directional_lights[light_id];

    var fragment_world: vec4<f32> = frag_position;

    if (directional_shadow_bias_mode == 0u) {
        // The normal bias is scaled to the texel size.
        let normal_offset = light.shadow_normal_bias * surface_normal.xyz;
        let depth_offset = light.shadow_depth_bias * light.direction_to_light.xyz;
        fragment_world = vec4<f32>(frag_position.xyz + normal_offset + depth_offset, frag_position.w);
    }

    let fragment_light_clip = light.view_projection * fragment_world;
    if (fragment_light_clip.w <= 0.0) {
        return 1.0;
    }
    let fragment_light_ndc = fragment_light_clip.xyz / fragment_light_clip.w;
    // No shadow outside the orthographic projection volume
    if (any(fragment_light_ndc.xy < vec2<f32>(-1.0)) || fragment_light_ndc.z < 0.0
            || any(fragment_light_ndc < vec3<f32>(0.0))) {
        return 1.0;
    }

    // compute texture coordinates for shadow lookup, compensating for the Y-flip difference
    // between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    let fragment_shadow_map_uv = fragment_light_ndc.xy * flip_correction + vec2<f32>(0.5, 0.5);

    if (directional_shadow_pcss_mode == 0u) {
        return sample_directional_shadow_pcf_disc(
            fragment_shadow_map_uv,
            fragment_world,
            surface_normal,
            light_id,
            fragment_light_ndc.z,
            pcf_sampling_radius,
            frag_coord
        );
    } else { // directional_shadow_pcss_mode == 1u or 2u
        return sample_directional_shadow_pcss(
            fragment_shadow_map_uv,
            fragment_world,
            surface_normal,
            light_id,
            fragment_light_ndc.z,
            pcf_sampling_radius,
            frag_coord
        );
    }
}

struct FragmentInput {
    [[builtin(front_facing)]] is_front: bool;
    [[builtin(position)]] frag_coord: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    var output_color: vec4<f32> = material.base_color;
    if ((material.flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    }

    // // NOTE: Unlit bit not set means == 0 is true, so the true case is if lit
    if ((material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {
        // TODO use .a for exposure compensation in HDR
        var emissive: vec4<f32> = material.emissive;
        if ((material.flags & STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BIT) != 0u) {
            emissive = vec4<f32>(emissive.rgb * textureSample(emissive_texture, emissive_sampler, in.uv).rgb, 1.0);
        }

        // calculate non-linear roughness from linear perceptualRoughness
        var metallic: f32 = material.metallic;
        var perceptual_roughness: f32 = material.perceptual_roughness;
        if ((material.flags & STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BIT) != 0u) {
            let metallic_roughness = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
            // Sampling from GLTF standard channels for now
            metallic = metallic * metallic_roughness.b;
            perceptual_roughness = perceptual_roughness * metallic_roughness.g;
        }
        let roughness = perceptualRoughnessToRoughness(perceptual_roughness);

        var occlusion: f32 = 1.0;
        if ((material.flags & STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BIT) != 0u) {
            occlusion = textureSample(occlusion_texture, occlusion_sampler, in.uv).r;
        }

        var N: vec3<f32> = normalize(in.world_normal);

        // FIXME: Normal maps need an additional vertex attribute and vertex stage output/fragment stage input
        //        Just use a separate shader for lit with normal maps?
        // #    ifdef STANDARDMATERIAL_NORMAL_MAP
        //     vec3 T = normalize(v_WorldTangent.xyz);
        //     vec3 B = cross(N, T) * v_WorldTangent.w;
        // #    endif

        if ((material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u) {
            if (!in.is_front) {
                N = -N;
            }
        // #        ifdef STANDARDMATERIAL_NORMAL_MAP
        //     T = gl_FrontFacing ? T : -T;
        //     B = gl_FrontFacing ? B : -B;
        // #        endif
        }

        // #    ifdef STANDARDMATERIAL_NORMAL_MAP
        //     mat3 TBN = mat3(T, B, N);
        //     N = TBN * normalize(texture(sampler2D(normal_map, normal_map_sampler), v_Uv).rgb * 2.0 - 1.0);
        // #    endif

        var V: vec3<f32>;
        if (view.projection.w.w != 1.0) { // If the projection is not orthographic
            // Only valid for a perpective projection
            V = normalize(view.world_position.xyz - in.world_position.xyz);
        } else {
            // Ortho view vec
            V = normalize(vec3<f32>(view.view_proj.x.z, view.view_proj.y.z, view.view_proj.z.z));
        }

        // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
        let NdotV = max(dot(N, V), 0.0001);

        // Remapping [0,1] reflectance to F0
        // See https://google.github.io/filament/Filament.html#materialsystem/parameterization/remapping
        let reflectance = material.reflectance;
        let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + output_color.rgb * metallic;

        // Diffuse strength inversely related to metallicity
        let diffuse_color = output_color.rgb * (1.0 - metallic);

        let R = reflect(-V, N);

        // accumulate color
        var light_accum: vec3<f32> = vec3<f32>(0.0);
        let n_point_lights = i32(lights.n_point_lights);
        let n_directional_lights = i32(lights.n_directional_lights);
        for (var i: i32 = 0; i < n_point_lights; i = i + 1) {
            let light = lights.point_lights[i];
            var shadow: f32;
            if ((mesh.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u) {
                shadow = fetch_point_shadow(i, in.world_position, in.world_normal, in.frag_coord.xy);
            } else {
                shadow = 1.0;
            }
            let light_contrib = point_light(in.world_position.xyz, light, roughness, NdotV, N, V, R, F0, diffuse_color);
            light_accum = light_accum + light_contrib * shadow;
        }
        for (var i: i32 = 0; i < n_directional_lights; i = i + 1) {
            let light = lights.directional_lights[i];
            var shadow: f32;
            if ((mesh.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u) {
                shadow = fetch_directional_shadow(i, in.world_position, in.world_normal, in.frag_coord.xy);
            } else {
                shadow = 1.0;
            }
            let light_contrib = directional_light(light, roughness, NdotV, N, V, R, F0, diffuse_color);
            light_accum = light_accum + light_contrib * shadow;
        }

        let diffuse_ambient = EnvBRDFApprox(diffuse_color, 1.0, NdotV);
        let specular_ambient = EnvBRDFApprox(F0, perceptual_roughness, NdotV);

        output_color = vec4<f32>(
            light_accum
                + (diffuse_ambient + specular_ambient) * lights.ambient_color.rgb * occlusion
                + emissive.rgb * output_color.a,
            output_color.a);

        // tone_mapping
        output_color = vec4<f32>(reinhard_luminance(output_color.rgb), output_color.a);
        // Gamma correction.
        // Not needed with sRGB buffer
        // output_color.rgb = pow(output_color.rgb, vec3(1.0 / 2.2));
    }

    return output_color;
}
