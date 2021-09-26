struct VertexOutput {
	[[builtin(position)]] position: vec4<f32>;
	[[location(0)]] uv: vec2<f32>;
};

var<private> vertices: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
	vec2<f32>(-1.0, -1.0),
	vec2<f32>(3.0, -1.0),
	vec2<f32>(-1.0, 3.0),
);

[[stage(vertex)]]
fn main([[builtin(vertex_index)]] idx: u32) -> VertexOutput {
	var out: VertexOutput;

	out.position = vec4<f32>(vertices[idx], 0.0, 1.0);
	out.uv = vertices[idx] * vec2<f32>(0.5, -0.5);
	out.uv = out.uv + 0.5;

	return out;
}

[[group(0), binding(0)]]
var hdr_target: texture_2d<f32>;

[[group(0), binding(1)]]
var hdr_target_sampler: sampler;

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    let l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

fn reinhard_luminance(color: vec3<f32>) -> vec3<f32> {
    let l_old = luminance(color);
    let l_new = l_old / (1.0 + l_old);
    return change_luminance(color, l_new);
}

[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
	let color = textureSample(hdr_target, hdr_target_sampler, in.uv);

	return vec4<f32>(reinhard_luminance(color.rgb), color.a);
	//return color;
}
