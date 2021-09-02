// NOTE: Keep in sync with pbr.wgsl
[[block]]
struct View {
    view_proj: mat4x4<f32>;
    projection: mat4x4<f32>;
    world_position: vec3<f32>;
    frame_number: u32;
};
[[group(0), binding(0)]]
var view: View;


[[block]]
struct Mesh {
    model: mat4x4<f32>;
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
};

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // NOTE: mesh.model is right-handed. Apply the right-handed transform to the right-handed vertex position
    //       then flip the sign of the z component to make the result be left-handed y-up
    let world_position_rh = mesh.model * vec4<f32>(vertex.position.xyz, 1.0);
    // NOTE: The point light view_proj is left-handed
    out.clip_position = view.view_proj * vec4<f32>(
        world_position_rh.x,
        world_position_rh.y,
        -world_position_rh.z,
        1.0
    );
    return out;
}
