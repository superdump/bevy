#version 450

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_normal;
layout(location = 2) in vec2 vertex_uv;

layout(location = 0) out vec2 v_uv;

struct ColorGrading {
    float exposure,
    float gamma,
    float pre_saturation,
    float post_saturation,
}

layout(set = 0, binding = 0) uniform View {
    mat4 world_to_ndc,
    mat4 unjittered_world_to_ndc,
    mat4 ndc_to_world,
    mat4 view_to_world,
    mat4 world_to_view,
    mat4 view_to_ndc,
    mat4 ndc_to_view,
    vec3 world_position,
    // viewport(x_origin, y_origin, width, height)
    vec4 viewport,
    ColorGrading color_grading,
    float mip_bias,
};

struct Mesh {
    mat3x4 local_to_world;
    mat4 inverse_transpose_local_to_world;
    uint flags;
};

#ifdef PER_OBJECT_BUFFER_BATCH_SIZE
layout(set = 2, binding = 0) uniform Mesh meshes[#{PER_OBJECT_BUFFER_BATCH_SIZE}];
#else
layout(set = 2, binding = 0) readonly buffer _meshes {
    Mesh meshes[];
};
#endif // PER_OBJECT_BUFFER_BATCH_SIZE

mat4 affine_to_square(mat3x4 affine) {
    return transpose(mat4(
        affine[0],
        affine[1],
        affine[2],
        vec4(0.0, 0.0, 0.0, 1.0)
    ));
}

void main() {
    v_uv = vertex_uv;
    gl_Position = world_to_ndc
        * affine_to_square(meshes[gl_InstanceIndex].local_to_world)
        * vec4(vertex_position, 1.0);
}
