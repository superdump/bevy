[[block]]
struct View {
    view: mat4x4<f32>;
    view_inv: mat4x4<f32>;
    proj: mat4x4<f32>;
    proj_inv: mat4x4<f32>;
    view_proj: mat4x4<f32>;
    world_position: vec3<f32>;
};

[[block]]
struct SsaoConfig {
    kernel: [[stride(16)]] array<vec3<f32>, 32>;
    kernel_size: u32;
    radius: f32;
    bias: f32;
};

[[group(0), binding(0)]]
var view: View;
[[group(0), binding(1)]]
var depth_texture: texture_depth_2d;
[[group(0), binding(2)]]
var depth_sampler: sampler;
[[group(0), binding(3)]]
var normal_texture: texture_2d<f32>;
[[group(0), binding(4)]]
var normal_sampler: sampler;

[[group(1), binding(0)]]
var config: SsaoConfig;
[[group(1), binding(1)]]
var blue_noise_texture: texture_2d<f32>;
[[group(1), binding(2)]]
var blue_noise_sampler: sampler;

struct FragmentInput {
    [[location(0)]] uv: vec2<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] f32 {
    // tile noise texture over screen, based on screen dimensions divided by noise size
    let frame_size = textureDimensions(normal_texture);
    let noise_size = textureDimensions(blue_noise_texture);
    let noiseScale = vec2<f32>(frame_size) / vec2<f32>(noise_size);

    // Calculate the fragment position from the depth texture
    let frag_depth_ndc = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0.0);
    if (frag_depth_ndc == 0.0) {
        return 1.0;
    }
    // in.uv.x is [0,1] from left to right. * 2 - 1 remaps to [-1, 1] left to right which is NDC
    // in.uv.y is [0,1] top to bottom. (1-v)*2-1 = 2-2v-1 = 1-2v remaps to [-1, 1] bottom to top which is NDC
    let frag_view_homogeneous = view.proj_inv * vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - 2.0 * in.uv.y, frag_depth_ndc, 1.0);
    let frag_view = frag_view_homogeneous.xyz / frag_view_homogeneous.w;
    let normal_view = (textureSampleLevel(normal_texture, normal_sampler, in.uv, 0.0).xyz - 0.5) * 2.0;
    let blue_noise = textureSampleLevel(blue_noise_texture, blue_noise_sampler, in.uv * noiseScale, 0.0).xy;
    let randomVec = normalize(vec3<f32>(
        blue_noise * 2.0 - 1.0,
        0.0
    ));

    let tangent = normalize(randomVec - normal_view * dot(randomVec, normal_view));
    let bitangent = cross(normal_view, tangent);
    let TBN = mat3x3<f32>(tangent, bitangent, normal_view);

    var occlusion: f32 = 0.0;

    for (var i: i32 = 0; i < i32(config.kernel_size); i = i + 1) {
        let sample_offset_view = TBN * config.kernel[i].xyz; // from tangent to view space
        let sample_view = vec4<f32>(frag_view.xyz + sample_offset_view * config.radius, 1.0);
        let sample_clip = view.proj * sample_view; // from view to clip space
        let sample_ndc = sample_clip.xyz / sample_clip.w; // perspective divide
        // sample_ndc.x is [-1,1] left to right, so * 0.5 + 0.5 remaps to [0,1] left to right
        // sample_ndc.y is [-1,1] bottom to top, so * -0.5 + 0.5 remaps to [0,1] top to bottom
        let depth_uv = vec2<f32>(sample_ndc.x * 0.5 + 0.5, sample_ndc.y * -0.5 + 0.5);

        let sample_depth_ndc = textureSampleLevel(depth_texture, depth_sampler, depth_uv, 0.0);
        let sample_depth_view_homogeneous = view.proj_inv * vec4<f32>(0.0, 0.0, sample_depth_ndc, 1.0);
        let sample_depth_view = sample_depth_view_homogeneous.xyz / sample_depth_view_homogeneous.w;

        let range_check = smoothStep(0.0, 1.0, config.radius / abs(frag_view.z - sample_depth_view.z));
        if (sample_depth_view.z >= sample_view.z + config.bias) {
            occlusion = occlusion + range_check;
        }
    }

    return 1.0 - (occlusion / f32(config.kernel_size));
}
