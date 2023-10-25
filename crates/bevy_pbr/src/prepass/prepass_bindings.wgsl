#define_import_path bevy_pbr::prepass_bindings

#ifdef MOTION_VECTOR_PREPASS
@group(0) @binding(2) var<uniform> previous_world_to_ndc: mat4x4<f32>;
#endif // MOTION_VECTOR_PREPASS

// Material bindings will be in @group(1)
