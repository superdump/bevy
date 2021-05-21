#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;

layout(location = 0) out vec3 v_ViewNormal;

layout(set = 0, binding = 0) uniform CameraViewProj {
    mat4 ViewProj;
};
layout(set = 1, binding = 0) uniform CameraViewInv3 {
    mat4 ViewInv3;
};

layout(set = 2, binding = 0) uniform Transform {
    mat4 Model;
};

mat3 inverse_temp(mat3 m) {
  float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
  float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
  float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

  float b01 = a22 * a11 - a12 * a21;
  float b11 = -a22 * a10 + a12 * a20;
  float b21 = a21 * a10 - a11 * a20;

  float det = a00 * b01 + a01 * b11 + a02 * b21;

  return mat3(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11),
              b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10),
              b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) / det;
}

void main() {
    // For non-uniform scaling, the Model matrix must be inverse-transposed which has the effect of
    // applying inverse scaling while retaining the correct rotation
    // The normals must be re-normalised after applying the inverse-transpose because this can affect
    // the length of the normal
    // The normals need to rotate inverse to the view rotation
    // Using mat3 is important else the translation in the Model matrix can have other unintended effects
    v_ViewNormal = normalize(mat3(ViewInv3) * transpose(inverse_temp(mat3(Model))) * Vertex_Normal) * 0.5 + 0.5;

    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
}
