#version 450

// FIXME: Make these into uniforms
const float RADIUS = 0.1;
const float BIAS = 0.025;
const int KERNEL_SIZE = 32;
const float kernel[32][3] = // precalculated hemisphere kernel (low discrepancy noiser)
    {
        { -0.668154f, -0.084296f, 0.219458f },
        { -0.092521f, 0.141327f, 0.505343f },
        { -0.041960f, 0.700333f, 0.365754f },
        { 0.722389f, -0.015338f, 0.084357f },
        { -0.815016f, 0.253065f, 0.465702f },
        { 0.018993f, -0.397084f, 0.136878f },
        { 0.617953f, -0.234334f, 0.513754f },
        { -0.281008f, -0.697906f, 0.240010f },
        { 0.303332f, -0.443484f, 0.588136f },
        { -0.477513f, 0.559972f, 0.310942f },
        { 0.307240f, 0.076276f, 0.324207f },
        { -0.404343f, -0.615461f, 0.098425f },
        { 0.152483f, -0.326314f, 0.399277f },
        { 0.435708f, 0.630501f, 0.169620f },
        { 0.878907f, 0.179609f, 0.266964f },
        { -0.049752f, -0.232228f, 0.264012f },
        { 0.537254f, -0.047783f, 0.693834f },
        { 0.001000f, 0.177300f, 0.096643f },
        { 0.626400f, 0.524401f, 0.492467f },
        { -0.708714f, -0.223893f, 0.182458f },
        { -0.106760f, 0.020965f, 0.451976f },
        { -0.285181f, -0.388014f, 0.241756f },
        { 0.241154f, -0.174978f, 0.574671f },
        { -0.405747f, 0.080275f, 0.055816f },
        { 0.079375f, 0.289697f, 0.348373f },
        { 0.298047f, -0.309351f, 0.114787f },
        { -0.616434f, -0.117369f, 0.475924f },
        { -0.035249f, 0.134591f, 0.840251f },
        { 0.175849f, 0.971033f, 0.211778f },
        { 0.024805f, 0.348056f, 0.240006f },
        { -0.267123f, 0.204885f, 0.688595f },
        { -0.077639f, -0.753205f, 0.070938f }
    };

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

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // mat4 Proj;
    // Proj[0] = vec4(1.357995, 0.0, 0.0, 0.0);
    // Proj[1] = vec4(0.0, 2.4142134, 0.0, 0.0);
    // Proj[2] = vec4(0.0, 0.0, -1.001001, -1.0);
    // Proj[3] = vec4(0.0, 0.0, -1.001001, 0.0);

    // mat4 ProjInv;
    // ProjInv[0] = vec4(0.7363797, 0.0, -0.0, 0.0);
    // ProjInv[1] = vec4(0.0, 0.41421357, 0.0, -0.0);
    // ProjInv[2] = vec4(-0.0, 0.0, -0.0, -0.99899995);
    // ProjInv[3] = vec4(0.0, -0.0, -1.0, 1.0);

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
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        vec3 sample_offset_view = TBN * vec3(kernel[i][0], kernel[i][1], kernel[i][2]); // from tangent to view space
        vec4 sample_view = vec4(frag_view.xyz + sample_offset_view * RADIUS, 1.0);
        vec4 sample_clip = Proj * sample_view; // from view to clip space
        vec3 sample_ndc = sample_clip.xyz / sample_clip.w; // perspective divide
        // sample_ndc.x is [-1,1] left to right, so * 0.5 + 0.5 remaps to [0,1] left to right
        // sample_ndc.y is [-1,1] bottom to top, so * -0.5 + 0.5 remaps to [0,1] top to bottom
        vec2 depth_uv = vec2(sample_ndc.x * 0.5 + 0.5, sample_ndc.y * -0.5 + 0.5);

        float sample_depth_ndc = texture(sampler2D(depth_texture, depth_texture_sampler), depth_uv).x;
        vec4 sample_depth_view = ProjInv * vec4(0.0, 0.0, sample_depth_ndc, 1.0);
        sample_depth_view.xyz /= sample_depth_view.w;

        float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(frag_view.z - sample_depth_view.z));
        occlusion += (sample_depth_view.z >= sample_view.z + BIAS ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = 1.0 - (occlusion / KERNEL_SIZE);

    o_Target = occlusion;
}
