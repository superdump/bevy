// NOTE: Keep in sync with depth.wgsl and depth_prepass.wgsl
[[block]]
struct View {
    view_proj: mat4x4<f32>;
    inverse_view: mat4x4<f32>;
    projection: mat4x4<f32>;
    world_position: vec3<f32>;
    near: f32;
    far: f32;
    width: f32;
    height: f32;
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
var<uniform> view: View;
[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

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
    alpha_cutoff: f32;
};

let STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT: u32         = 1u;
let STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BIT: u32           = 2u;
let STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 4u;
let STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BIT: u32          = 8u;
let STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT: u32               = 16u;
let STANDARD_MATERIAL_FLAGS_UNLIT_BIT: u32                      = 32u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE: u32              = 64u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK: u32                = 128u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND: u32               = 256u;

struct PointLight {
    // NOTE: .z.z .z.w .w.z .w.w
    projection_lr: vec4<f32>;
    color_inverse_square_range: vec4<f32>;
    position_radius: vec4<f32>;
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32;
    shadow_depth_bias: f32;
    shadow_normal_bias: f32;
};

let POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT: u32 = 1u;

struct DirectionalCascade {
    view_projection: mat4x4<f32>;
    // NOTE: .xy are texel_size, .zw are scale
    texel_size_scale: vec4<f32>;
};

// NOTE: Keep in sync with light.rs!
let MAX_CASCADES_PER_LIGHT: u32 = 8u;

struct DirectionalLight {
    // NOTE: there array sizes must be kept in sync with the constants defined bevy_pbr2/src/render/light.rs
    cascades: array<DirectionalCascade, 8u>; // MAX_CASCADES_PER_LIGHT
    // NOTE: contains the far view z bounds of each cascade
    cascades_far_bounds: array<vec4<f32>, 2u>; // (MAX_CASCADES_PER_LIGHT + 3) / 4
    color: vec4<f32>;
    direction_to_light: vec3<f32>;
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32;
    shadow_depth_bias: f32;
    shadow_normal_bias: f32;
    n_cascades: u32;
    // The proportion by which adjacent cascades overlap
    cascade_overlap_proportion: f32;
};

let DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT: u32 = 1u;

[[block]]
struct Lights {
    // NOTE: this array size must be kept in sync with the constants defined bevy_pbr2/src/render/light.rs
    directional_lights: array<DirectionalLight, 1u>;
    ambient_color: vec4<f32>;
    cluster_dimensions: vec4<u32>; // x/y/z dimensions
    cluster_factors: vec4<f32>; // xy are vec2<f32<(cluster_dimensions.xy) / vec2<f32<(view.width, view.height)
                                // z is cluster_dimensions.z / log(far / near)
                                // w is cluster_dimensions.z * log(near) / log(far / near)
    n_directional_lights: u32;
};

[[block]]
struct PointLights {
    data: array<PointLight, 256u>;
};

[[block]]
struct ClusterLightIndexLists {
    data: array<vec4<u32>, 1024u>; // each u32 contains 4 u8 indices into the PointLights array
};

[[block]]
struct ClusterOffsetsAndCounts {
    data: array<vec4<u32>, 1024u>; // each u32 contains a 24-bit index into ClusterLightIndexLists in the high 24 bits
                             // and an 8-bit count of the number of lights in the low 8 bits
};


[[group(0), binding(1)]]
var<uniform> lights: Lights;
[[group(0), binding(2)]]
var point_shadow_textures: texture_depth_cube_array;
[[group(0), binding(3)]]
var point_shadow_textures_sampler: sampler_comparison;
[[group(0), binding(4)]]
var directional_shadow_textures: texture_depth_2d_array;
[[group(0), binding(5)]]
var directional_shadow_textures_sampler: sampler_comparison;
[[group(0), binding(6)]]
var<uniform> point_lights: PointLights;
[[group(0), binding(7)]]
var<uniform> cluster_light_index_lists: ClusterLightIndexLists;
[[group(0), binding(8)]]
var<uniform> cluster_offsets_and_counts: ClusterOffsetsAndCounts;
[[group(0), binding(9)]]
var blue_noise_texture: texture_2d<f32>;
[[group(0), binding(10)]]
var blue_noise_sampler: sampler;

[[group(1), binding(0)]]
var<uniform> material: StandardMaterial;
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
    let numerator = color * (1.0 + (color / vec3<f32>(max_white * max_white)));
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
    let l_new = l_old / (1.0 + l_old);
    return change_luminance(color, l_new);
}

fn reinhard_extended_luminance(color: vec3<f32>, max_white_l: f32) -> vec3<f32> {
    let l_old = luminance(color);
    let numerator = l_old * (1.0 + (l_old / (max_white_l * max_white_l)));
    let l_new = numerator / (1.0 + l_old);
    return change_luminance(color, l_new);
}

fn view_z_to_z_slice(view_z: f32) -> u32 {
    // NOTE: had to use -view_z to make it positive else log(negative) is nan
    return u32(floor(log(-view_z) * lights.cluster_factors.z - lights.cluster_factors.w));
}

fn fragment_cluster_index(frag_coord: vec2<f32>, view_z: f32) -> u32 {
    let xy = vec2<u32>(floor(frag_coord * lights.cluster_factors.xy));
    let z_slice = view_z_to_z_slice(view_z);
    return (xy.y * lights.cluster_dimensions.x + xy.x) * lights.cluster_dimensions.z + z_slice;
}

struct ClusterOffsetAndCount {
    offset: u32;
    count: u32;
};

fn unpack_offset_and_count(cluster_index: u32) -> ClusterOffsetAndCount {
    let offset_and_count = cluster_offsets_and_counts.data[cluster_index >> 2u][cluster_index & ((1u << 2u) - 1u)];
    var output: ClusterOffsetAndCount;
    // The offset is stored in the upper 24 bits
    output.offset = (offset_and_count >> 8u) & ((1u << 24u) - 1u);
    // The count is stored in the lower 8 bits
    output.count = offset_and_count & ((1u << 8u) - 1u);
    return output;
}

fn get_light_id(index: u32) -> u32 {
    // The index is correct but in cluster_light_index_lists we pack 4 u8s into a u32
    // This means the index into cluster_light_index_lists is index / 4
    let indices = cluster_light_index_lists.data[index >> 4u][(index >> 2u) & ((1u << 2u) - 1u)];
    // And index % 4 gives the sub-index of the u8 within the u32 so we shift by 8 * sub-index
    return (indices >> (8u * (index & ((1u << 2u) - 1u)))) & ((1u << 8u) - 1u);
}

fn point_light(
    world_position: vec3<f32>, light: PointLight, roughness: f32, NdotV: f32, N: vec3<f32>, V: vec3<f32>,
    R: vec3<f32>, F0: vec3<f32>, diffuseColor: vec3<f32>
) -> vec3<f32> {
    let light_to_frag = light.position_radius.xyz - world_position.xyz;
    let distance_square = dot(light_to_frag, light_to_frag);
    let rangeAttenuation =
        getDistanceAttenuation(distance_square, light.color_inverse_square_range.w);

    // Specular.
    // Representative Point Area Lights.
    // see http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p14-16
    let a = roughness;
    let centerToRay = dot(light_to_frag, R) * R - light_to_frag;
    let closestPoint = light_to_frag + centerToRay * saturate(light.position_radius.w * inverseSqrt(dot(centerToRay, centerToRay)));
    let LspecLengthInverse = inverseSqrt(dot(closestPoint, closestPoint));
    let normalizationFactor = a / saturate(a + (light.position_radius.w * 0.5 * LspecLengthInverse));
    let specularIntensity = normalizationFactor * normalizationFactor;

    var L: vec3<f32> = closestPoint * LspecLengthInverse; // normalize() equivalent?
    var H: vec3<f32> = normalize(L + V);
    var NoL: f32 = saturate(dot(N, L));
    var NoH: f32 = saturate(dot(N, H));
    var LoH: f32 = saturate(dot(L, H));

    let specular_light = specular(F0, roughness, H, NdotV, NoL, NoH, LoH, specularIntensity);

    // Diffuse.
    // Comes after specular since its NoL is used in the lighting equation.
    L = normalize(light_to_frag);
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

    return ((diffuse + specular_light) * light.color_inverse_square_range.rgb) * (rangeAttenuation * NoL);
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

fn fetch_point_shadow(light_id: u32, frag_position: vec4<f32>, surface_normal: vec3<f32>) -> f32 {
    let light = point_lights.data[light_id];

    // because the shadow maps align with the axes and the frustum planes are at 45 degrees
    // we can get the worldspace depth by taking the largest absolute axis
    let surface_to_light = light.position_radius.xyz - frag_position.xyz;
    let surface_to_light_abs = abs(surface_to_light);
    let distance_to_light = max(surface_to_light_abs.x, max(surface_to_light_abs.y, surface_to_light_abs.z));

    // The normal bias here is already scaled by the texel size at 1 world unit from the light.
    // The texel size increases proportionally with distance from the light so multiplying by
    // distance to light scales the normal bias to the texel size at the fragment distance.
    let normal_offset = light.shadow_normal_bias * distance_to_light * surface_normal.xyz;
    let depth_offset = light.shadow_depth_bias * normalize(surface_to_light.xyz);
    let offset_position = frag_position.xyz + normal_offset + depth_offset;

    // similar largest-absolute-axis trick as above, but now with the offset fragment position
    let frag_ls = light.position_radius.xyz - offset_position.xyz;
    let abs_position_ls = abs(frag_ls);
    let major_axis_magnitude = max(abs_position_ls.x, max(abs_position_ls.y, abs_position_ls.z));

    // NOTE: These simplifications come from multiplying:
    //       projection * vec4(0, 0, -major_axis_magnitude, 1.0)
    //       and keeping only the terms that have any impact on the depth.
    // Projection-agnostic approach:
    let zw = -major_axis_magnitude * light.projection_lr.xy + light.projection_lr.zw;
    let depth = zw.x / zw.y;

    // do the lookup, using HW PCF and comparison
    // NOTE: Due to the non-uniform control flow above, we must use the Level variant of
    //       textureSampleCompare to avoid undefined behaviour due to some of the fragments in
    //       a quad (2x2 fragments) being processed not being sampled, and this messing with
    //       mip-mapping functionality. The shadow maps have no mipmaps so Level just samples
    //       from LOD 0.
    return textureSampleCompareLevel(point_shadow_textures, point_shadow_textures_sampler, frag_ls, i32(light_id), depth);
}

let CASCADE_INDEX_OFFSETS_4: vec4<u32> = vec4<u32>(0u, 1u, 2u, 3u);

fn get_cascade_index(light_id: u32, view_z: f32) -> u32 {
    var light: DirectionalLight = lights.directional_lights[light_id];

    // NOTE: This parallel comparison technique is from the CascadedShadowMaps11 DirextX SDK example code
    //       which is MIT licensed
    var index_f32: f32 = 0.0;
    let view_z4 = vec4<f32>(-view_z);
    let max_elements = (light.n_cascades + 3u) / 4u;
    let n_cascades_4 = vec4<u32>(
        light.n_cascades,
        light.n_cascades,
        light.n_cascades,
        light.n_cascades
    );
    for (var i: u32 = 0u; i < max_elements; i = i + 1u) {
        // NOTE: Comparison contains true for each cascade far bound where view z is closer
        let comparison = view_z4 <= light.cascades_far_bounds[i];
        let four_i4 = vec4<u32>(i * 4u);
        index_f32 = index_f32 + dot(
            vec4<f32>(comparison),
            vec4<f32>((four_i4 + CASCADE_INDEX_OFFSETS_4) < n_cascades_4)
        );
    }

    // NOTE: If view z is closer than all cascade far bounds, then it is the 0th cascade.
    //       If it is not closer than any, then it is beyond the last cascade and should
    //       return light.n_cascades or greater.
    return light.n_cascades - u32(index_f32);
}

// From: https://github.com/TheRealMJP/Shadows
// For Poisson Disk PCF sampling
var<private> POISSON_SAMPLES: array<vec2<f32>, 64> = array<vec2<f32>, 64>(
    vec2<f32>(-0.5119625, -0.4827938),
    vec2<f32>(-0.2171264, -0.4768726),
    vec2<f32>(-0.7552931, -0.2426507),
    vec2<f32>(-0.7136765, -0.4496614),
    vec2<f32>(-0.5938849, -0.6895654),
    vec2<f32>(-0.3148003, -0.7047654),
    vec2<f32>(-0.42215, -0.2024607),
    vec2<f32>(-0.9466816, -0.2014508),
    vec2<f32>(-0.8409063, -0.03465778),
    vec2<f32>(-0.6517572, -0.07476326),
    vec2<f32>(-0.1041822, -0.02521214),
    vec2<f32>(-0.3042712, -0.02195431),
    vec2<f32>(-0.5082307, 0.1079806),
    vec2<f32>(-0.08429877, -0.2316298),
    vec2<f32>(-0.9879128, 0.1113683),
    vec2<f32>(-0.3859636, 0.3363545),
    vec2<f32>(-0.1925334, 0.1787288),
    vec2<f32>(0.003256182, 0.138135),
    vec2<f32>(-0.8706837, 0.3010679),
    vec2<f32>(-0.6982038, 0.1904326),
    vec2<f32>(0.1975043, 0.2221317),
    vec2<f32>(0.1507788, 0.4204168),
    vec2<f32>(0.3514056, 0.09865579),
    vec2<f32>(0.1558783, -0.08460935),
    vec2<f32>(-0.0684978, 0.4461993),
    vec2<f32>(0.3780522, 0.3478679),
    vec2<f32>(0.3956799, -0.1469177),
    vec2<f32>(0.5838975, 0.1054943),
    vec2<f32>(0.6155105, 0.3245716),
    vec2<f32>(0.3928624, -0.4417621),
    vec2<f32>(0.1749884, -0.4202175),
    vec2<f32>(0.6813727, -0.2424808),
    vec2<f32>(-0.6707711, 0.4912741),
    vec2<f32>(0.0005130528, -0.8058334),
    vec2<f32>(0.02703013, -0.6010728),
    vec2<f32>(-0.1658188, -0.9695674),
    vec2<f32>(0.4060591, -0.7100726),
    vec2<f32>(0.7713396, -0.4713659),
    vec2<f32>(0.573212, -0.51544),
    vec2<f32>(-0.3448896, -0.9046497),
    vec2<f32>(0.1268544, -0.9874692),
    vec2<f32>(0.7418533, -0.6667366),
    vec2<f32>(0.3492522, 0.5924662),
    vec2<f32>(0.5679897, 0.5343465),
    vec2<f32>(0.5663417, 0.7708698),
    vec2<f32>(0.7375497, 0.6691415),
    vec2<f32>(0.2271994, -0.6163502),
    vec2<f32>(0.2312844, 0.8725659),
    vec2<f32>(0.4216993, 0.9002838),
    vec2<f32>(0.4262091, -0.9013284),
    vec2<f32>(0.2001408, -0.808381),
    vec2<f32>(0.149394, 0.6650763),
    vec2<f32>(-0.09640376, 0.9843736),
    vec2<f32>(0.7682328, -0.07273844),
    vec2<f32>(0.04146584, 0.8313184),
    vec2<f32>(0.9705266, -0.1143304),
    vec2<f32>(0.9670017, 0.1293385),
    vec2<f32>(0.9015037, -0.3306949),
    vec2<f32>(-0.5085648, 0.7534177),
    vec2<f32>(0.9055501, 0.3758393),
    vec2<f32>(0.7599946, 0.1809109),
    vec2<f32>(-0.2483695, 0.7942952),
    vec2<f32>(-0.4241052, 0.5581087),
    vec2<f32>(-0.1020106, 0.6724468)
);

let MAX_KERNEL_SIZE: f32 = 9.0;
let FILTER_SIZE: vec2<f32> = vec2<f32>(8.0, 8.0);
let NUM_DISC_SAMPLES: u32 = 1u;
let TAU: f32 = 6.283185307179586;

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

// 2x2 matrix inverse as WGSL does not have one yet
fn inverse2x2(m: mat2x2<f32>) -> mat2x2<f32> {
    let inv_det = 1.0 / determinant(m);
    return mat2x2<f32>(
        vec2<f32>( m[1][1], -m[0][1]),
        vec2<f32>(-m[1][0],  m[0][0])
    ) * inv_det;
}

fn compute_receiver_plane_depth_bias(tex_coord_dx: vec3<f32>, tex_coord_dy: vec3<f32>) -> vec2<f32> {
    var bias_uv: vec2<f32> = vec2<f32>(
        tex_coord_dy.y * tex_coord_dx.z - tex_coord_dx.y * tex_coord_dy.z,
        tex_coord_dx.x * tex_coord_dy.z - tex_coord_dy.x * tex_coord_dx.z
    );
    bias_uv = bias_uv * 1.0 / ((tex_coord_dx.x * tex_coord_dy.y) - (tex_coord_dx.y * tex_coord_dy.x));
    return bias_uv;
}

// fn compute_receiver_plane_depth_bias(duvdepth_dscreenu: vec3<f32>, duvdepth_dscreenv: vec3<f32>) -> vec2<f32> {
//     // Receiver Plane Depth Bias
//     // from https://developer.amd.com/wordpress/media/2012/10/Isidoro-ShadowMapping.pdf
//     let uv_jacobian = mat2x2<f32>(duvdepth_dscreenu.xy, duvdepth_dscreenv.xy);
//     let ddepth_dscreenuv = vec2<f32>(duvdepth_dscreenu.z, duvdepth_dscreenv.z);
//     return transpose(inverse2x2(uv_jacobian)) * ddepth_dscreenuv;
// }

//--------------------------------------------------------------------------------------
// Samples the shadow map using a PCF kernel made up from random points on a disc
//--------------------------------------------------------------------------------------
fn sample_shadow_map_random_disc_pcf(
    shadow_pos: vec3<f32>,
    light_id: u32,
    cascade_index: u32,
    cascade_0_scale: f32,
    cascade_scale: f32,
    screen_pos: vec2<u32>
) -> f32 {
    var light: DirectionalLight = lights.directional_lights[light_id];
    let cascade = light.cascades[cascade_index];

    let shadow_pos_dx = dpdx(shadow_pos);
    let shadow_pos_dy = dpdy(shadow_pos);

    let max_filter_size = vec2<f32>(MAX_KERNEL_SIZE / abs(cascade_0_scale));
    let filter_size = clamp(min(FILTER_SIZE, max_filter_size) * abs(cascade_scale), vec2<f32>(1.0), vec2<f32>(MAX_KERNEL_SIZE));
    // let filter_size = vec2<f32>(2.0);

    var result: f32 = 1.0;

    // Get the size of the shadow map
    let shadow_map_size = vec2<f32>(textureDimensions(directional_shadow_textures).xy);

    // #if UsePlaneDepthBias_
        let texel_size = vec2<f32>(1.0) / shadow_map_size;

        // let receiver_plane_depth_bias = compute_receiver_plane_depth_bias(shadow_pos_dx, shadow_pos_dy);
        // let receiver_plane_depth_bias = ddepth_duv;
        let receiver_plane_depth_bias = vec2<f32>(0.0);

        // Static depth biasing to make up for incorrect fractional sampling on the shadow map grid
        let fractional_sampling_error = dot(texel_size, vec2<f32>(abs(receiver_plane_depth_bias)));
        // let shadow_depth = shadow_pos.z + min(fractional_sampling_error, 0.01);
    // #else
       let shadow_depth = shadow_pos.z;
    // #endif

    if (filter_size.x > 1.0 || filter_size.y > 1.0) {
        // #if RandomizeOffsets_

        // tile noise texture over screen, based on screen dimensions divided by noise size
        let noise_scale = vec2<f32>(1.0) / vec2<f32>(textureDimensions(blue_noise_texture));

        // Frame variation
        // Offset the blue noise _UVs_ by the R2 sequence based on the frame number
        let base_noise_uv = vec2<f32>(screen_pos) * noise_scale + R2(0u % 64u);
        // Offset the blue noise _value_ by a frame number multiple of the golden ratio
        // let frame_golden_ratio_offset = vec2<f32>(f32(view.frame_number % 64u) * golden_ratio);

        let rotation_angle = textureSampleLevel(
            blue_noise_texture,
            blue_noise_sampler,
            base_noise_uv,
            0.0
        ).x * TAU;
        let c = cos(rotation_angle);
        let s = sin(rotation_angle);
        let random_rotation_matrix = mat2x2<f32>(
            vec2<f32>(c, -s),
            vec2<f32>(s,  c),
        );
        //     // Get a value to randomly rotate the kernel by
        //     uint2 random_rotations_size;
        //     RandomRotations.GetDimensions(random_rotations_size.x, random_rotations_size.y);
        //     uint2 random_sample_pos = screen_pos % random_rotations_size;
        //     float theta = RandomRotations[random_sample_pos] * Pi2;
        //     mat2x2<f32> random_rotation_matrix = mat2x2<f32>(vec2<f32>(cos(theta), -sin(theta)),
        //                                              vec2<f32>(sin(theta), cos(theta)));
        // #endif

        let sample_scale = (0.5 * filter_size) / shadow_map_size;

        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < NUM_DISC_SAMPLES; i = i + 1u)
        {
            // #if RandomizeOffsets_
            //     vec2<f32> sample_offset = mul(PoissonSamples[i], random_rotation_matrix) * sample_scale;
                let sample_offset = (random_rotation_matrix * POISSON_SAMPLES[i]) * sample_scale;
            // #else
            //    let sample_offset = POISSON_SAMPLES[i] * sample_scale;
            // #endif

            let sample_pos = shadow_pos.xy + sample_offset;

            // #if UsePlaneDepthBias_
                // Compute offset and apply planar depth bias
                let sample_depth = shadow_depth + dot(sample_offset, receiver_plane_depth_bias);
            // #else
            //    let sample_depth = shadow_depth;
            // #endif

            sum = sum + textureSampleCompareLevel(
                directional_shadow_textures,
                directional_shadow_textures_sampler,
                sample_pos,
                i32(light_id * MAX_CASCADES_PER_LIGHT + cascade_index),
                sample_depth
            );
        }

        result = sum / f32(NUM_DISC_SAMPLES);
    } else {
        result = textureSampleCompareLevel(
            directional_shadow_textures,
            directional_shadow_textures_sampler,
            shadow_pos.xy,
            i32(light_id * MAX_CASCADES_PER_LIGHT + cascade_index),
            shadow_depth
        );
    }

    return result;
}

//-------------------------------------------------------------------------------------------------
// Helper function for SampleShadowMapOptimizedPCF
//-------------------------------------------------------------------------------------------------
fn SampleShadowMap(
    base_uv: vec2<f32>,
    u: f32,
    v: f32,
    shadow_map_size_inv: vec2<f32>,
    light_id: u32,
    cascade_index: u32,
    cascade_scale: f32,
    depth: f32,
    receiver_plane_depth_bias: vec2<f32>
) -> f32 {

    let uv = base_uv + vec2<f32>(u, v) * shadow_map_size_inv;// / cascade_scale;

    // #if UsePlaneDepthBias_
    //     let z = depth + dot(vec2<f32>(u, v) * shadow_map_size_inv, receiver_plane_depth_bias);
    // #else
        let z = depth;
    // #endif

    return textureSampleCompareLevel(
        directional_shadow_textures,
        directional_shadow_textures_sampler,
        uv,
        i32(light_id * MAX_CASCADES_PER_LIGHT + cascade_index),
        z
    );
}

//-------------------------------------------------------------------------------------------------
// The method used in The Witness
//-------------------------------------------------------------------------------------------------
fn SampleShadowMapOptimizedPCF(
    shadowPos: vec3<f32>,
    // shadowPosDX: vec3<f32>,
    // shadowPosDY: vec3<f32>,
    light_id: u32,
    cascade_index: u32,
    cascade_scale: f32,
    filter_size: f32
) -> f32 {
    let shadowMapSize = vec3<f32>(textureDimensions(directional_shadow_textures));

    let shadowPosDX = dpdx(shadowPos);
    let shadowPosDY = dpdy(shadowPos);

    let lightDepth = shadowPos.z;

    // let bias = 0.1;

    // #if UsePlaneDepthBias_
    //     vec2<f32> texelSize = 1.0f / shadowMapSize;

    //     vec2<f32> receiverPlaneDepthBias = ComputeReceiverPlaneDepthBias(shadowPosDX, shadowPosDY);
        let receiverPlaneDepthBias = compute_receiver_plane_depth_bias(shadowPosDX, shadowPosDY);

    //     // Static depth biasing to make up for incorrect fractional sampling on the shadow map grid
    //     let fractionalSamplingError = 2 * dot(vec2<f32>(1.0f, 1.0f) * texelSize, abs(receiverPlaneDepthBias));
    //     lightDepth -= min(fractionalSamplingError, 0.01f);
    // #else
    //     vec2<f32> receiverPlaneDepthBias;
    //     lightDepth -= bias;
    // #endif

    let uv = shadowPos.xy * shadowMapSize.xy; // 1 unit - 1 texel

    let shadowMapSizeInv = 1.0 / shadowMapSize.xy;

    var base_uv: vec2<f32> = vec2<f32>(
        floor(uv.x + 0.5),
        floor(uv.y + 0.5)
    );

    let s = (uv.x + 0.5 - base_uv.x);
    let t = (uv.y + 0.5 - base_uv.y);

    base_uv = base_uv - vec2<f32>(0.5, 0.5);
    base_uv = base_uv * shadowMapSizeInv;

    var sum: f32 = 0.0;

    if (filter_size <= 2.0) {
        return textureSampleCompareLevel(
            directional_shadow_textures,
            directional_shadow_textures_sampler,
            shadowPos.xy,
            i32(light_id * MAX_CASCADES_PER_LIGHT + cascade_index),
            lightDepth
        );
    } elseif (filter_size <= 3.0) {
        let uw0 = (3.0 - 2.0 * s);
        let uw1 = (1.0 + 2.0 * s);

        let u0 = (2.0 - s) / uw0 - 1.0;
        let u1 = s / uw1 + 1.0;

        let vw0 = (3.0 - 2.0 * t);
        let vw1 = (1.0 + 2.0 * t);

        let v0 = (2.0 - t) / vw0 - 1.0;
        let v1 = t / vw1 + 1.0;

        sum = sum + uw0 * vw0 * SampleShadowMap(base_uv, u0, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw0 * SampleShadowMap(base_uv, u1, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw0 * vw1 * SampleShadowMap(base_uv, u0, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw1 * SampleShadowMap(base_uv, u1, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        return sum * 1.0 / 16.0;
    } elseif (filter_size <= 5.0) {
        let uw0 = (4.0 - 3.0 * s);
        let uw1 = 7.0;
        let uw2 = (1.0 + 3.0 * s);

        let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
        let u1 = (3.0 + s) / uw1;
        let u2 = s / uw2 + 2.0;

        let vw0 = (4.0 - 3.0 * t);
        let vw1 = 7.0;
        let vw2 = (1.0 + 3.0 * t);

        let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
        let v1 = (3.0 + t) / vw1;
        let v2 = t / vw2 + 2.0;

        sum = sum + uw0 * vw0 * SampleShadowMap(base_uv, u0, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw0 * SampleShadowMap(base_uv, u1, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw2 * vw0 * SampleShadowMap(base_uv, u2, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        sum = sum + uw0 * vw1 * SampleShadowMap(base_uv, u0, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw1 * SampleShadowMap(base_uv, u1, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw2 * vw1 * SampleShadowMap(base_uv, u2, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        sum = sum + uw0 * vw2 * SampleShadowMap(base_uv, u0, v2, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw2 * SampleShadowMap(base_uv, u1, v2, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw2 * vw2 * SampleShadowMap(base_uv, u2, v2, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        return sum * 1.0 / 144.0;
    } else {
        // filter_size == 7.0

        let uw0 = (5.0 * s - 6.0);
        let uw1 = (11.0 * s - 28.0);
        let uw2 = -(11.0 * s + 17.0);
        let uw3 = -(5.0 * s + 1.0);

        let u0 = (4.0 * s - 5.0) / uw0 - 3.0;
        let u1 = (4.0 * s - 16.0) / uw1 - 1.0;
        let u2 = -(7.0 * s + 5.0) / uw2 + 1.0;
        let u3 = -s / uw3 + 3.0;

        let vw0 = (5.0 * t - 6.0);
        let vw1 = (11.0 * t - 28.0);
        let vw2 = -(11.0 * t + 17.0);
        let vw3 = -(5.0 * t + 1.0);

        let v0 = (4.0 * t - 5.0) / vw0 - 3.0;
        let v1 = (4.0 * t - 16.0) / vw1 - 1.0;
        let v2 = -(7.0 * t + 5.0) / vw2 + 1.0;
        let v3 = -t / vw3 + 3.0;

        sum = sum + uw0 * vw0 * SampleShadowMap(base_uv, u0, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw0 * SampleShadowMap(base_uv, u1, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw2 * vw0 * SampleShadowMap(base_uv, u2, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw3 * vw0 * SampleShadowMap(base_uv, u3, v0, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        sum = sum + uw0 * vw1 * SampleShadowMap(base_uv, u0, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw1 * SampleShadowMap(base_uv, u1, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw2 * vw1 * SampleShadowMap(base_uv, u2, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw3 * vw1 * SampleShadowMap(base_uv, u3, v1, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        sum = sum + uw0 * vw2 * SampleShadowMap(base_uv, u0, v2, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw2 * SampleShadowMap(base_uv, u1, v2, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw2 * vw2 * SampleShadowMap(base_uv, u2, v2, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw3 * vw2 * SampleShadowMap(base_uv, u3, v2, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        sum = sum + uw0 * vw3 * SampleShadowMap(base_uv, u0, v3, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw1 * vw3 * SampleShadowMap(base_uv, u1, v3, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw2 * vw3 * SampleShadowMap(base_uv, u2, v3, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);
        sum = sum + uw3 * vw3 * SampleShadowMap(base_uv, u3, v3, shadowMapSizeInv, light_id, cascade_index, cascade_scale, lightDepth, receiverPlaneDepthBias);

        return sum * 1.0 / 2704.0;
    }
}

fn sample_cascade(
    light_id: u32,
    cascade_index: u32,
    frag_position: vec4<f32>,
    surface_normal: vec3<f32>,
    frag_coord: vec4<f32>,
    cascade_0_scale: f32,
    cascade_scale: f32
) -> f32 {
    var light: DirectionalLight = lights.directional_lights[light_id];
    let cascade = light.cascades[cascade_index];


    // let frag_position_dpdx = dpdx(frag_position.xyz);
    // let frag_position_dpdy = dpdy(frag_position.xyz);

    // The normal bias is scaled to the texel size.
    let normal_offset = light.shadow_normal_bias * cascade.texel_size_scale.x * surface_normal.xyz;
    let depth_offset = light.shadow_depth_bias * light.direction_to_light.xyz;
    let offset_position = vec4<f32>(frag_position.xyz + normal_offset + depth_offset, frag_position.w);

    let offset_position_clip = cascade.view_projection * offset_position;
    if (offset_position_clip.w <= 0.0) {
        return 1.0;
    }
    let offset_position_ndc = offset_position_clip.xyz / offset_position_clip.w;
    // No shadow outside the orthographic projection volume
    if (any(offset_position_ndc.xy < vec2<f32>(-1.0)) || offset_position_ndc.z < 0.0
            || any(offset_position_ndc > vec3<f32>(1.0))) {
        return 1.0;
    }

    // compute texture coordinates for shadow lookup, compensating for the Y-flip difference
    // between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    let shadow_map_uv = offset_position_ndc.xy * flip_correction + vec2<f32>(0.5, 0.5);

    let depth = offset_position_ndc.z;

    let shadow_map_uv_depth = vec3<f32>(shadow_map_uv, depth);
    // let receiver_plane_depth_bias = compute_receiver_plane_depth_bias(shadow_map_uv_depth);

    // return SampleShadowMapOptimizedPCF(
    //     shadow_map_uv_depth,
    //     // shadowPosDX: vec3<f32>,
    //     // shadowPosDY: vec3<f32>,
    //     light_id,
    //     cascade_index,
    //     cascade_scale,
    //     5.0
    // );

    return sample_shadow_map_random_disc_pcf(
        shadow_map_uv_depth,
        // receiver_plane_depth_bias,
        light_id,
        cascade_index,
        cascade_0_scale,
        cascade_scale,
        vec2<u32>(frag_coord.xy)
    );

    // // do the lookup, using HW PCF and comparison
    // // NOTE: Due to non-uniform control flow above, we must use the level variant of the texture
    // //       sampler to avoid use of implicit derivatives causing possible undefined behavior.
    // return textureSampleCompareLevel(
    //     directional_shadow_textures,
    //     directional_shadow_textures_sampler,
    //     shadow_map_uv,
    //     i32(light_id * MAX_CASCADES_PER_LIGHT + cascade_index),
    //     depth
    // );
}

fn fetch_directional_shadow(light_id: u32, frag_position: vec4<f32>, surface_normal: vec3<f32>, view_z: f32, frag_coord: vec4<f32>) -> f32 {
    var light = lights.directional_lights[light_id];

    let cascade_index = get_cascade_index(light_id, view_z);
    if (cascade_index >= light.n_cascades) {
        return 1.0;
    }

    var cascade_near_bound: f32 = 0.0;
    if (cascade_index > 0u) {
        let prev_cascade_index = cascade_index - 1u;
        cascade_near_bound = light.cascades_far_bounds[prev_cascade_index >> 2u][prev_cascade_index & 3u];
    }
    let cascade_far_bound = light.cascades_far_bounds[cascade_index >> 2u][cascade_index & 3u];
    let final_cascade_index = light.n_cascades - 1u;
    let shadow_far_bound = light.cascades_far_bounds[final_cascade_index >> 2u][final_cascade_index & 3u];

    let cascade_0_scale = shadow_far_bound / light.cascades_far_bounds[0u][0u];
    let cascade_scale = shadow_far_bound / (cascade_far_bound - cascade_near_bound);


    var shadow: f32 = sample_cascade(light_id, cascade_index, frag_position, surface_normal, frag_coord, cascade_0_scale, cascade_scale);

    let next_cascade_index = cascade_index + 1u;
    if (next_cascade_index < light.n_cascades) {
        let next_cascade_near_bound = (1.0 - light.cascade_overlap_proportion) * cascade_far_bound;
        if (-view_z >= next_cascade_near_bound) {
            let next_cascade_shadow = sample_cascade(light_id, next_cascade_index, frag_position, surface_normal, frag_coord, cascade_0_scale, cascade_scale);
            shadow = mix(
                shadow,
                next_cascade_shadow,
                (-view_z - next_cascade_near_bound)
                    / (cascade_far_bound - next_cascade_near_bound)
            );
        }
    }

    return shadow;
}

fn random1D(s: f32) -> f32 {
    return fract(sin(s * 12.9898) * 43758.5453123);
}

fn hsv2rgb(hue: f32, saturation: f32, value: f32) -> vec3<f32> {
    let rgb = clamp(
        abs(
            ((hue * 6.0 + vec3<f32>(0.0, 4.0, 2.0)) % 6.0) - 3.0
        ) - 1.0,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

	return value * mix( vec3<f32>(1.0), rgb, vec3<f32>(saturation));
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

        if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE) != 0u) {
            // NOTE: If rendering as opaque, alpha should be ignored so set to 1.0
            output_color.a = 1.0;
        } elseif ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK) != 0u) {
            if (output_color.a >= material.alpha_cutoff) {
                // NOTE: If rendering as masked alpha and >= the cutoff, render as fully opaque
                output_color.a = 1.0;
            }
            // NOTE: output_color.a < material.alpha_cutoff should not reach here as it will not
            //       be discarded in the depth prepass and we use the 'equals' depth buffer comparison
            //       function
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

        let view_z = dot(vec4<f32>(
            view.inverse_view.x.z,
            view.inverse_view.y.z,
            view.inverse_view.z.z,
            view.inverse_view.w.z
        ), in.world_position);
        let cluster_index = fragment_cluster_index(in.frag_coord.xy, view_z);
        let offset_and_count = unpack_offset_and_count(cluster_index);
        for (var i: u32 = offset_and_count.offset; i < offset_and_count.offset + offset_and_count.count; i = i + 1u) {
            let light_id = get_light_id(i);
            let light = point_lights.data[light_id];
            var shadow: f32 = 1.0;
            if ((mesh.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
                    || (light.flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
                shadow = fetch_point_shadow(light_id, in.world_position, in.world_normal);
            }
            let light_contrib = point_light(in.world_position.xyz, light, roughness, NdotV, N, V, R, F0, diffuse_color);
            light_accum = light_accum + light_contrib * shadow;
        }

        let n_directional_lights = lights.n_directional_lights;
        for (var i: u32 = 0u; i < n_directional_lights; i = i + 1u) {
            let light = lights.directional_lights[i];
            var shadow: f32 = 1.0;
            if ((mesh.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
                    || (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
                shadow = fetch_directional_shadow(i, in.world_position, in.world_normal, view_z, in.frag_coord);
            }
            let light_contrib = directional_light(light, roughness, NdotV, N, V, R, F0, diffuse_color);
            light_accum = light_accum + light_contrib * shadow;
        }

        let diffuse_ambient = EnvBRDFApprox(diffuse_color, 1.0, NdotV);
        let specular_ambient = EnvBRDFApprox(F0, perceptual_roughness, NdotV);

        output_color = vec4<f32>(
            light_accum +
                (diffuse_ambient + specular_ambient) * lights.ambient_color.rgb * occlusion +
                emissive.rgb * output_color.a,
            output_color.a);

        // Debug overlay alpha
        let debug_overlay_alpha = 0.4;

        // Cascade debug
        let cascade_debug_mode = 0;
        if (cascade_debug_mode == 1) {
            // Visualize the cascades

            var light: DirectionalLight = lights.directional_lights[0u];

            var cascade_index: u32 = get_cascade_index(0u, view_z);
            var cascade_color: vec3<f32> = hsv2rgb(f32(cascade_index) / f32(light.n_cascades + 1u), 1.0, 0.5);

            var next_cascade_index: u32 = cascade_index + 1u;
            if (next_cascade_index < light.n_cascades) {
                let cascade_far_bound = light.cascades_far_bounds[cascade_index >> 2u][cascade_index & 3u];
                let next_cascade_near_bound = (1.0 - light.cascade_overlap_proportion) * cascade_far_bound;
                if (-view_z >= next_cascade_near_bound) {
                    let next_cascade_color = hsv2rgb(f32(next_cascade_index) / f32(light.n_cascades + 1u), 1.0, 0.5);
                    cascade_color = mix(
                        cascade_color,
                        next_cascade_color,
                        (-view_z - next_cascade_near_bound)
                            / (cascade_far_bound - next_cascade_near_bound)
                    );
                }
            }

            output_color = vec4<f32>(
                (1.0 - debug_overlay_alpha) * output_color.rgb + debug_overlay_alpha * cascade_color,
                output_color.a
            );
        }

        // Cluster allocation debug (using 'over' alpha blending)
        let cluster_debug_mode = 0;
        if (cluster_debug_mode == 1) {
            // Visualize the z slices
            var z_slice: u32 = view_z_to_z_slice(view_z);
            // A hack to make the colors alternate a bit more
            if ((z_slice & 1u) == 1u) {
                z_slice = z_slice + lights.cluster_dimensions.z / 2u;
            }
            let slice_color = hsv2rgb(f32(z_slice) / f32(lights.cluster_dimensions.z + 1u), 1.0, 0.5);
            output_color = vec4<f32>(
                (1.0 - debug_overlay_alpha) * output_color.rgb + debug_overlay_alpha * slice_color,
                output_color.a
            );
        } elseif (cluster_debug_mode == 2) {
            // Visualize the number of lights per cluster
            output_color.r = (1.0 - debug_overlay_alpha) * output_color.r
                + debug_overlay_alpha * smoothStep(0.0, 32.0, f32(offset_and_count.count));
            output_color.g = (1.0 - debug_overlay_alpha) * output_color.g
                + debug_overlay_alpha * (1.0 - smoothStep(0.0, 32.0, f32(offset_and_count.count)));
        } elseif (cluster_debug_mode == 3) {
            // Visualize the cluster
            let cluster_color = hsv2rgb(random1D(f32(cluster_index)), 1.0, 0.5);
            output_color = vec4<f32>(
                (1.0 - debug_overlay_alpha) * output_color.rgb + debug_overlay_alpha * cluster_color,
                output_color.a
            );
        }

        // tone_mapping (done later in the pipeline)
        // output_color = vec4<f32>(reinhard_luminance(output_color.rgb), output_color.a);
        // Gamma correction.
        // Not needed with sRGB buffer
        // output_color.rgb = pow(output_color.rgb, vec3(1.0 / 2.2));
    }

    return output_color;
}
