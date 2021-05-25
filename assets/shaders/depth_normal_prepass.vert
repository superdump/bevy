#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;
layout(location = 2) in vec2 Vertex_Uv;

#ifdef STANDARDMATERIAL_NORMAL_MAP
layout(location = 3) in vec4 Vertex_Tangent;
#endif

layout(location = 0) out vec3 v_ViewNormal;
layout(location = 1) out vec2 v_Uv;

#ifdef STANDARDMATERIAL_NORMAL_MAP
layout(location = 2) out vec4 v_ViewTangent;
#endif

layout(set = 0, binding = 0) uniform CameraViewProj {
    mat4 ViewProj;
};
layout(set = 1, binding = 0) uniform CameraView {
    mat4 View;
};
layout(set = 1, binding = 1) uniform CameraViewInv3 {
    mat4 ViewInv3;
};

layout(set = 2, binding = 0) uniform Transform {
    mat4 Model;
};
layout(set = 2, binding = 1) uniform ModelInvTrans3_model_inv_trans_3 {
    mat4 ModelInvTrans3;
};

void main() {
    // For non-uniform scaling, the Model matrix must be inverse-transposed which has the effect of
    // applying inverse scaling while retaining the correct rotation
    // The normals must be re-normalised after applying the inverse-transpose because this can affect
    // the length of the normal
    // The normals need to rotate inverse to the view rotation
    // Using mat3 is important else the translation in the Model matrix can have other unintended effects
    v_ViewNormal = mat3(ViewInv3) * mat3(ModelInvTrans3) * Vertex_Normal * 0.5 + 0.5;
#ifdef STANDARDMATERIAL_NORMAL_MAP
    v_ViewTangent = vec4(mat3(View) * mat3(Model) * Vertex_Tangent.xyz, Vertex_Tangent.w);
#endif
    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
}
