#version 450

// FIXME: Move to a uniform!
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

// layout(set = 0, binding = 0) uniform CameraProj {
//     mat4 Proj;
// };
// layout(set = 1, binding = 1) uniform CameraInvProj {
//     mat4 InvProj;
// };

layout(set = 0, binding = 0) uniform texture2D depth_texture;
layout(set = 0, binding = 1) uniform sampler depth_texture_sampler;
layout(set = 0, binding = 2) uniform texture2D normal_texture;
layout(set = 0, binding = 3) uniform sampler normal_texture_sampler;
// layout(set = 2, binding = 4) uniform texture2D noise_texture;
// layout(set = 2, binding = 5) uniform sampler noise_texture_sampler;

// FIXME: Make these into uniforms
const float RADIUS = 0.5;
const float BIAS = 0.025;
const int KERNEL_SIZE = 32;

// From Matt Pettineo's article: https://mynameismjp.wordpress.com/2010/09/05/position-from-depth-3/
// but I derived my own method
// const float NearClipDistance = 1.0;
// const float FarClipDistance = 1000.0;
// const float ProjectionA = FarClipDistance / (FarClipDistance - NearClipDistance);
// const float ProjectionB = (-FarClipDistance * NearClipDistance) / (FarClipDistance - NearClipDistance);

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // Hard=coded perspective_rh with fov_y_radians 45, aspect ratio 16/9, near 0.1, far 1000.0
    // Can't yet bind camera projection in a fullscreen pass node and that may not make sense
    mat4 Proj;
    Proj[0] = vec4(1.357995, 0.0, 0.0, 0.0);
    Proj[1] = vec4(0.0, 2.4142134, 0.0, 0.0);
    Proj[2] = vec4(0.0, 0.0, -1.001001, -1.0);
    Proj[3] = vec4(0.0, 0.0, -1.001001, 0.0);

    mat4 InvProj;
    InvProj[0] = vec4(0.7363797, 0.0, -0.0, 0.0);
    InvProj[1] = vec4(0.0, 0.41421357, 0.0, -0.0);
    InvProj[2] = vec4(-0.0, 0.0, -0.0, -0.99899995);
    InvProj[3] = vec4(0.0, -0.0, -1.0, 1.0);

    // TODO: For the 4x4 noise texture sampling
    // tile noise texture over screen, based on screen dimensions divided by noise size
    // const ivec2 size = textureSize(sampler2D(normal_texture, normal_texture_sampler), 0);
    // const vec2 noiseScale = vec2(size) / 4.0;

    // Calculate the fragment position from the depth texture
    float depth = texture(sampler2D(depth_texture, depth_texture_sampler), v_Uv).x;
    if (depth == 1.0) {
        o_Target = 1.0;
        return;
    }
    vec3 frag_ndc = vec3(v_Uv * 2.0 - 1.0, depth);
    vec4 frag_view = InvProj * vec4(frag_ndc, 1.0);
    frag_view.xyz = frag_view.xyz / frag_view.w;
    vec3 normal = (texture(sampler2D(normal_texture, normal_texture_sampler), v_Uv).xyz - 0.5) * 2.0;
    vec3 randomVec = normalize(vec3(rand(v_Uv), 0.0, rand(v_Uv + 1234.0)));
    // TODO: Bind a 4x4 noise texture
    // vec3 randomVec = texture(sampler2D(noise_texture, noise_texture_sampler), v_Uv * noiseScale).xyz;

    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        vec3 sample_view = TBN * vec3(kernel[i][0], kernel[i][1], kernel[i][2]);
        sample_view = frag_view.xyz + sample_view * RADIUS;

        vec4 offset_view = vec4(sample_view, 1.0);
        vec4 offset_clip = Proj * offset_view; // from view to clip space
        vec3 offset_ndc = offset_clip.xyz / offset_clip.w; // perspective divide
        vec2 uv = offset_ndc.xy * 0.5 + 0.5;
        // offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 to 1.0

        float sampleDepth = texture(sampler2D(depth_texture, depth_texture_sampler), uv).x;
        // sampleDepth = dot(vec3(sampleDepth), InvProj[2].xyz);

        // float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(frag_view.z - sampleDepth));
        // occlusion += (sampleDepth >= sample_view.z + BIAS ? 1.0 : 0.0) * rangeCheck;
        float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(frag_ndc.z - sampleDepth));
        occlusion += (sampleDepth >= offset_ndc.z + BIAS ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = 1.0 - (occlusion / KERNEL_SIZE);

    o_Target = occlusion;
}
