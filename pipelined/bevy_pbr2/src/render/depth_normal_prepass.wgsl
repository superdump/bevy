[[block]]
struct View {
    view: mat4x4<f32>;
    view_inv: mat4x4<f32>;
    proj: mat4x4<f32>;
    proj_inv: mat4x4<f32>;
    view_proj: mat4x4<f32>;
    world_position: vec3<f32>;
};
[[group(0), binding(0)]]
var view: View;

[[block]]
struct Mesh {
    transform: mat4x4<f32>;
    transform_inverse_transpose: mat4x4<f32>;
};
[[group(1), binding(0)]]
var mesh: Mesh;

struct Vertex {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] view_normal: vec3<f32>;
};

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = view.view_proj * mesh.transform * vec4<f32>(vertex.position, 1.0);

    // For non-uniform scaling, the model matrix must be inverse-transposed which has the effect of
    // applying inverse scaling while retaining the correct rotation
    // The normals must be re-normalised after applying the inverse-transpose because this can affect
    // the length of the normal
    // The normals need to rotate inverse to the view rotation
    // Using mat3 is important else the translation in the model matrix can have other unintended effects
    out.view_normal =
        mat3x3<f32>(
            view.view_inv.x.xyz,
            view.view_inv.y.xyz,
            view.view_inv.z.xyz
        )
        * mat3x3<f32>(
            mesh.transform_inverse_transpose.x.xyz,
            mesh.transform_inverse_transpose.y.xyz,
            mesh.transform_inverse_transpose.z.xyz
        )
        * vertex.normal;

    return out;
}

struct FragmentInput {
    [[location(0)]] view_normal: vec3<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(normalize(in.view_normal) * 0.5 + 0.5, 1.0);
}
