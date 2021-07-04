struct Vertex {
    [[builtin(vertex_index)]] index: u32;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] uv: vec2<f32>;
};

[[stage(vertex)]]
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    // Set up a single triangle
    let x = f32((vertex.index & 1u) << 2u);
    let y = f32((vertex.index & 2u) << 1u);
    out.uv = vec2<f32>(x * 0.5, 1.0 - (y * 0.5));
    out.clip_position = vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0);

    return out;
}
