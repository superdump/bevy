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
fn vertex([[builtin(vertex_index)]] idx: u32) -> VertexOutput {
	var out: VertexOutput;

	out.position = vec4<f32>(vertices[idx], 0.0, 1.0);
	out.uv = vertices[idx] * vec2<f32>(0.5, -0.5);
	out.uv = out.uv + 0.5;

	return out;
}

[[block]]
struct Uniforms {
	threshold: f32;
	knee: f32;
	scale: f32;
};

[[group(0), binding(0)]]
var org: texture_2d<f32>;

[[group(0), binding(1)]]
var sampler: sampler;

[[group(0), binding(2)]]
var<uniform> uniforms: Uniforms;

[[group(0), binding(3)]]
var up: texture_2d<f32>;

fn quadratic_threshold(color: vec4<f32>, threshold: f32, curve: vec3<f32>) -> vec4<f32> {
	let br = max(max(color.r, color.g), color.b);

	var rq: f32 = clamp(br - curve.x, 0.0, curve.y);
	rq = curve.z * rq * rq;

	return color * max(rq, br - threshold) / max(br, 0.0001); 
}

[[stage(fragment)]]
fn down_sample_pre_filter(in: VertexOutput) -> [[location(0)]] vec4<f32> {
	let texel_size = 1.0 / vec2<f32>(textureDimensions(org));

	let scale = texel_size;

	let a = textureSample(org, sampler, in.uv + vec2<f32>(-1.0, -1.0) * scale);
	let b = textureSample(org, sampler, in.uv + vec2<f32>( 0.0, -1.0) * scale);
	let c = textureSample(org, sampler, in.uv + vec2<f32>( 1.0, -1.0) * scale);
	let d = textureSample(org, sampler, in.uv + vec2<f32>(-0.5, -0.5) * scale);
	let e = textureSample(org, sampler, in.uv + vec2<f32>( 0.5, -0.5) * scale);
	let f = textureSample(org, sampler, in.uv + vec2<f32>(-1.0,  0.0) * scale);
	let g = textureSample(org, sampler, in.uv + vec2<f32>( 0.0,  0.0) * scale);
	let h = textureSample(org, sampler, in.uv + vec2<f32>( 1.0,  0.0) * scale);
	let i = textureSample(org, sampler, in.uv + vec2<f32>(-0.5,  0.5) * scale);
	let j = textureSample(org, sampler, in.uv + vec2<f32>( 0.5,  0.5) * scale);
	let k = textureSample(org, sampler, in.uv + vec2<f32>(-1.0,  1.0) * scale);
	let l = textureSample(org, sampler, in.uv + vec2<f32>( 0.0,  1.0) * scale);
	let m = textureSample(org, sampler, in.uv + vec2<f32>( 1.0,  1.0) * scale); 

	let div = (1.0 / 4.0) * vec2<f32>(0.5, 0.125);

	var o: vec4<f32> = (d + e + i + j) * div.x;
	o = o + (a + b + g + f) * div.y;
	o = o + (b + c + h + g) * div.y;
	o = o + (f + g + l + k) * div.y;
	o = o + (g + h + m + l) * div.y;

	let curve = vec3<f32>(
		uniforms.threshold - uniforms.knee,
		uniforms.knee * 2.0,
		0.25 / uniforms.knee,
	);

	o = quadratic_threshold(o, uniforms.threshold, curve);
	o = max(o, vec4<f32>(0.00001));

	return o;
}

[[stage(fragment)]]
fn down_sample(in: VertexOutput) -> [[location(0)]] vec4<f32> {
	let texel_size = 1.0 / vec2<f32>(textureDimensions(org));

	let scale = texel_size;

	let a = textureSample(org, sampler, in.uv + vec2<f32>(-1.0, -1.0) * scale);
	let b = textureSample(org, sampler, in.uv + vec2<f32>( 0.0, -1.0) * scale);
	let c = textureSample(org, sampler, in.uv + vec2<f32>( 1.0, -1.0) * scale);
	let d = textureSample(org, sampler, in.uv + vec2<f32>(-0.5, -0.5) * scale);
	let e = textureSample(org, sampler, in.uv + vec2<f32>( 0.5, -0.5) * scale);
	let f = textureSample(org, sampler, in.uv + vec2<f32>(-1.0,  0.0) * scale);
	let g = textureSample(org, sampler, in.uv + vec2<f32>( 0.0,  0.0) * scale);
	let h = textureSample(org, sampler, in.uv + vec2<f32>( 1.0,  0.0) * scale);
	let i = textureSample(org, sampler, in.uv + vec2<f32>(-0.5,  0.5) * scale);
	let j = textureSample(org, sampler, in.uv + vec2<f32>( 0.5,  0.5) * scale);
	let k = textureSample(org, sampler, in.uv + vec2<f32>(-1.0,  1.0) * scale);
	let l = textureSample(org, sampler, in.uv + vec2<f32>( 0.0,  1.0) * scale);
	let m = textureSample(org, sampler, in.uv + vec2<f32>( 1.0,  1.0) * scale); 

	let div = (1.0 / 4.0) * vec2<f32>(0.5, 0.125);

	var o: vec4<f32> = (d + e + i + j) * div.x;
	o = o + (a + b + g + f) * div.y;
	o = o + (b + c + h + g) * div.y;
	o = o + (f + g + l + k) * div.y;
	o = o + (g + h + m + l) * div.y;

	return o;
}


[[stage(fragment)]]
fn up_sample(in: VertexOutput) -> [[location(0)]] vec4<f32> {
	let texel_size = 1.0 / vec2<f32>(textureDimensions(org)) * uniforms.scale;
	let d = vec4<f32>(1.0, 1.0, -1.0, 0.0);

	var s: vec4<f32> = textureSample(org, sampler, in.uv - d.xy * texel_size);
	s = s + textureSample(org, sampler, in.uv - d.wy * texel_size) * 2.0;
	s = s + textureSample(org, sampler, in.uv - d.zy * texel_size);

	s = s + textureSample(org, sampler, in.uv + d.zw * texel_size) * 2.0;
	s = s + textureSample(org, sampler, in.uv       			 ) * 4.0;
	s = s + textureSample(org, sampler, in.uv + d.xw * texel_size) * 2.0;

	s = s + textureSample(org, sampler, in.uv + d.zy * texel_size);
	s = s + textureSample(org, sampler, in.uv + d.wy * texel_size) * 2.0;
	s = s + textureSample(org, sampler, in.uv + d.xy * texel_size);
	
	var color: vec4<f32> = textureSample(up, sampler, in.uv);
	color = vec4<f32>(color.rgb + s.rgb / 16.0, color.a);

	return color;
}

[[stage(fragment)]]
fn up_sample_final(in: VertexOutput) -> [[location(0)]] vec4<f32> {
	let texel_size = 1.0 / vec2<f32>(textureDimensions(org)) * uniforms.scale;
	let d = vec4<f32>(1.0, 1.0, -1.0, 0.0);

	var s: vec4<f32> = textureSample(org, sampler, in.uv - d.xy * texel_size);
	s = s + textureSample(org, sampler, in.uv - d.wy * texel_size) * 2.0;
	s = s + textureSample(org, sampler, in.uv - d.zy * texel_size);

	s = s + textureSample(org, sampler, in.uv + d.zw * texel_size) * 2.0;
	s = s + textureSample(org, sampler, in.uv       		     ) * 4.0;
	s = s + textureSample(org, sampler, in.uv + d.xw * texel_size) * 2.0;

	s = s + textureSample(org, sampler, in.uv + d.zy * texel_size);
	s = s + textureSample(org, sampler, in.uv + d.wy * texel_size) * 2.0;
	s = s + textureSample(org, sampler, in.uv + d.xy * texel_size);

	return vec4<f32>(s.rgb / 16.0, 1.0);
}