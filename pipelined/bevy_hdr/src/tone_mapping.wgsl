struct VertexOutput {
	[[builtin(position)]] position: vec4<f32>;
	[[location(0)]] uv: vec2<f32>;
};

var vertices: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
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

fn tonemap_aces(x: f32) -> f32 {
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    let a = 2.51;
    let b = 0.03; 
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
} 

let RGBTOXYZ: mat3x3<f32> = mat3x3<f32>(
	vec3<f32>(0.4124564, 0.2126729, 0.0193339),
	vec3<f32>(0.3575761, 0.7151522, 0.1191920),
	vec3<f32>(0.1804375, 0.0721750, 0.9503041),
);

let XYZTORGB: mat3x3<f32> = mat3x3<f32>(
	vec3<f32>(3.2404542, -0.9692660, 0.0556434),
	vec3<f32>(-1.5371385, 1.8760108, -0.2040259),
	vec3<f32>(-0.4985314, 0.0415560, 1.0572252),
);

fn rgb_to_yxy(rgb: vec3<f32>) -> vec3<f32> {
	let xyz = RGBTOXYZ * rgb;

	let x = xyz.r / (xyz.r + xyz.g + xyz.b);
	let y = xyz.g / (xyz.r + xyz.g + xyz.b);

	return vec3<f32>(xyz.g, x, y);
}

fn yxy_to_rgb(yxy: vec3<f32>) -> vec3<f32> {
	let xyz = vec3<f32>(
		yxy.r * yxy.g / yxy.b,
		yxy.r,
		(1.0 - yxy.g -  yxy.b) * (yxy.r / yxy.b),
	);

	return XYZTORGB * xyz;
}

fn gamma_correct(rgb: vec3<f32>) -> vec3<f32> {
	return pow(rgb, vec3<f32>(1.0/2.2));
}

[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
	let color = textureSample(hdr_target, hdr_target_sampler, in.uv);

	var yxy: vec3<f32> = rgb_to_yxy(color.rgb);

	yxy = vec3<f32>(tonemap_aces(yxy.r), yxy.gb);

	let rgb = yxy_to_rgb(yxy);

	return vec4<f32>(rgb, color.a);
}