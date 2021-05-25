#version 450

layout(location = 0) in vec3 v_ViewNormal;
layout(location = 1) in vec2 v_Uv;
#ifdef STANDARDMATERIAL_NORMAL_MAP
layout(location = 2) in vec4 v_ViewTangent;
#endif

layout(location = 0) out vec4 o_Target;

#ifdef STANDARDMATERIAL_NORMAL_MAP
layout(set = 3, binding = 0) uniform texture2D StandardMaterial_normal_map;
layout(set = 3, binding = 1) uniform sampler StandardMaterial_normal_map_sampler;
#endif

void main() {
    vec3 N = normalize(v_ViewNormal);

#    ifdef STANDARDMATERIAL_NORMAL_MAP
    vec3 T = normalize(v_ViewTangent.xyz);
    vec3 B = cross(N, T) * v_ViewTangent.w;
    mat3 TBN = mat3(T, B, N);
    N = TBN * normalize(texture(sampler2D(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler), v_Uv).rgb * 2.0 - 1.0);
#    endif
    o_Target = vec4(N, 1.0);
}
