#version 450

layout(location = 0) in vec2 v_Uv;

layout(location = 0) out float o_Target;

layout(set = 0, binding = 0) uniform CameraProj {
    mat4 Proj;
};
layout(set = 0, binding = 1) uniform CameraProjInv {
    mat4 ProjInv;
};

layout(set = 1, binding = 0) uniform texture2D depth_texture;
layout(set = 1, binding = 1) uniform sampler depth_texture_sampler;
layout(set = 1, binding = 2) uniform texture2D normal_texture;
layout(set = 1, binding = 3) uniform sampler normal_texture_sampler;
// layout(set = 1, binding = 4) uniform texture2D noise_texture;
// layout(set = 1, binding = 5) uniform sampler noise_texture_sampler;

struct SsaoConfig_t {
    vec4 kernel[32];
    uint kernel_size;
    float radius;
    float bias;
};
layout(set = 2, binding = 0) uniform SsaoConfig {
    SsaoConfig_t config;
};

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // TODO: For the 4x4 noise texture sampling
    // tile noise texture over screen, based on screen dimensions divided by noise size
    // const ivec2 size = textureSize(sampler2D(normal_texture, normal_texture_sampler), 0);
    // const vec2 noiseScale = vec2(size) / 4.0;

    // Calculate the fragment position from the depth texture
    float frag_depth_ndc = texture(sampler2D(depth_texture, depth_texture_sampler), v_Uv).x;
    if (frag_depth_ndc == 1.0) {
        o_Target = 1.0;
        return;
    }
    // v_Uv.x is [0,1] from left to right. * 2 - 1 remaps to [-1, 1] left to right which is NDC
    // v_Uv.y is [0,1] top to bottom. (1-v)*2-1 = 2-2v-1 = 1-2v remaps to [-1, 1] bottom to top which is NDC
    vec4 frag_view = ProjInv * vec4(v_Uv.x * 2.0 - 1.0, 1.0 - 2.0 * v_Uv.y, frag_depth_ndc, 1.0);
    frag_view.xyz /= frag_view.w;
    vec3 normal_view = (texture(sampler2D(normal_texture, normal_texture_sampler), v_Uv).xyz - 0.5) * 2.0;
    // TODO: Bind a 4x4 noise texture
    // vec3 randomVec = texture(sampler2D(noise_texture, noise_texture_sampler), v_Uv * noiseScale).xyz;
    vec3 randomVec = normalize(vec3(rand(v_Uv) * 2.0 - 1.0, rand(v_Uv + 1.0) * 2.0 - 1.0, 0.0));

    vec3 tangent = normalize(randomVec - normal_view * dot(randomVec, normal_view));
    vec3 bitangent = cross(normal_view, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal_view);

    float occlusion = 0.0;
    for (int i = 0; i < config.kernel_size; ++i) {
        vec3 sample_offset_view = TBN * config.kernel[i].xyz; // from tangent to view space
        vec4 sample_view = vec4(frag_view.xyz + sample_offset_view * config.radius, 1.0);
        vec4 sample_clip = Proj * sample_view; // from view to clip space
        vec3 sample_ndc = sample_clip.xyz / sample_clip.w; // perspective divide
        // sample_ndc.x is [-1,1] left to right, so * 0.5 + 0.5 remaps to [0,1] left to right
        // sample_ndc.y is [-1,1] bottom to top, so * -0.5 + 0.5 remaps to [0,1] top to bottom
        vec2 depth_uv = vec2(sample_ndc.x * 0.5 + 0.5, sample_ndc.y * -0.5 + 0.5);

        float sample_depth_ndc = texture(sampler2D(depth_texture, depth_texture_sampler), depth_uv).x;
        vec4 sample_depth_view = ProjInv * vec4(0.0, 0.0, sample_depth_ndc, 1.0);
        sample_depth_view.xyz /= sample_depth_view.w;

        float rangeCheck = smoothstep(0.0, 1.0, config.radius / abs(frag_view.z - sample_depth_view.z));
        occlusion += (sample_depth_view.z >= sample_view.z + config.bias ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = 1.0 - (occlusion / config.kernel_size);

    o_Target = occlusion;
}
