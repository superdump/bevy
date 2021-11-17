mod depth_prepass;
mod light;

pub use depth_prepass::*;
pub use light::*;

use crate::{AlphaMode, NotShadowCaster, NotShadowReceiver, StandardMaterial, PBR_SHADER_HANDLE};
use bevy_asset::Handle;
use bevy_core_pipeline::{AlphaMask3d, Opaque3d, SetItemPipeline, Transparent3d};
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::Mat4;
use bevy_render2::{
    mesh::Mesh,
    render_asset::RenderAssets,
    render_component::{ComponentUniforms, DynamicUniformIndex},
    render_phase::{
        DrawFunctions, EntityPhaseItem, PhaseItem, RenderCommand, RenderPhase, TrackedRenderPass,
    },
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    texture::{BevyDefault, GpuImage, Image, TextureFormatPixelInfo},
    view::{
        ComputedVisibility, ExtractedView, Msaa, ViewUniformOffset, ViewUniforms, VisibleEntities,
    },
};
use bevy_transform::components::GlobalTransform;
use crevice::std140::AsStd140;
use wgpu::{
    Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, TextureDimension, TextureFormat,
    TextureViewDescriptor,
};

#[derive(AsStd140, Clone)]
pub struct MeshUniform {
    pub transform: Mat4,
    pub inverse_transpose_model: Mat4,
    pub flags: u32,
}

// NOTE: These must match the bit flags in bevy_pbr2/src/render/pbr.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    struct MeshFlags: u32 {
        const SHADOW_RECEIVER            = (1 << 0);
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

// NOTE: These must match the bit flags in bevy_pbr2/src/render/pbr.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    pub struct StandardMaterialFlags: u32 {
        const BASE_COLOR_TEXTURE         = (1 << 0);
        const EMISSIVE_TEXTURE           = (1 << 1);
        const METALLIC_ROUGHNESS_TEXTURE = (1 << 2);
        const OCCLUSION_TEXTURE          = (1 << 3);
        const DOUBLE_SIDED               = (1 << 4);
        const UNLIT                      = (1 << 5);
        const ALPHA_MODE_OPAQUE          = (1 << 6);
        const ALPHA_MODE_MASK            = (1 << 7);
        const ALPHA_MODE_BLEND           = (1 << 8);
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

pub fn extract_meshes(
    mut commands: Commands,
    mut previous_caster_len: Local<usize>,
    mut previous_not_caster_len: Local<usize>,
    caster_query: Query<
        (
            Entity,
            &ComputedVisibility,
            &GlobalTransform,
            &Handle<Mesh>,
            Option<&NotShadowReceiver>,
        ),
        Without<NotShadowCaster>,
    >,
    not_caster_query: Query<
        (
            Entity,
            &ComputedVisibility,
            &GlobalTransform,
            &Handle<Mesh>,
            Option<&NotShadowReceiver>,
        ),
        With<NotShadowCaster>,
    >,
) {
    let mut caster_values = Vec::with_capacity(*previous_caster_len);
    for (entity, computed_visibility, transform, handle, not_receiver) in caster_query.iter() {
        if !computed_visibility.is_visible {
            continue;
        }
        let transform = transform.compute_matrix();
        caster_values.push((
            entity,
            (
                handle.clone_weak(),
                MeshUniform {
                    flags: if not_receiver.is_some() {
                        MeshFlags::empty().bits
                    } else {
                        MeshFlags::SHADOW_RECEIVER.bits
                    },
                    transform,
                    inverse_transpose_model: transform.inverse().transpose(),
                },
            ),
        ));
    }
    *previous_caster_len = caster_values.len();
    commands.insert_or_spawn_batch(caster_values);

    let mut not_caster_values = Vec::with_capacity(*previous_not_caster_len);
    for (entity, computed_visibility, transform, handle, not_receiver) in not_caster_query.iter() {
        if !computed_visibility.is_visible {
            continue;
        }
        let transform = transform.compute_matrix();
        not_caster_values.push((
            entity,
            (
                handle.clone_weak(),
                MeshUniform {
                    flags: if not_receiver.is_some() {
                        MeshFlags::empty().bits
                    } else {
                        MeshFlags::SHADOW_RECEIVER.bits
                    },
                    transform,
                    inverse_transpose_model: transform.inverse().transpose(),
                },
                NotShadowCaster,
            ),
        ));
    }
    *previous_not_caster_len = not_caster_values.len();
    commands.insert_or_spawn_batch(not_caster_values);
}

#[derive(Clone)]
pub struct PbrPipeline {
    pub view_layout: BindGroupLayout,
    pub material_layout: BindGroupLayout,
    pub mesh_layout: BindGroupLayout,
    // This dummy white texture is to be used in place of optional StandardMaterial textures
    pub dummy_white_gpu_image: GpuImage,
}

impl FromWorld for PbrPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        // TODO: change this to ViewUniform::std140_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(224),
                    },
                    count: None,
                },
                // Lights
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        // TODO: change this to GpuLights::std140_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(256),
                    },
                    count: None,
                },
                // Point Shadow Texture Cube Array
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::CubeArray,
                    },
                    count: None,
                },
                // Point Shadow Texture Array Sampler
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: true,
                        filtering: true,
                    },
                    count: None,
                },
                // Directional Shadow Texture Array
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                // Directional Shadow Texture Array Sampler
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: true,
                        filtering: true,
                    },
                    count: None,
                },
                // PointLights
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // NOTE: Static size for uniform buffers. GpuPointLight has a padded
                        //       size of 128 bytes, so 16384 / 128 = 128 point lights max
                        min_binding_size: BufferSize::new(16384),
                    },
                    count: None,
                },
                // ClusteredLightIndexLists
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // NOTE: With 128 point lights max, indices need 7 bits. Use u8 for
                        //       convenience.
                        min_binding_size: BufferSize::new(16384),
                    },
                    count: None,
                },
                // ClusterOffsetsAndCounts
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // NOTE: The offset needs to address 16384 indices, which needs 21 bits.
                        //       The count can be at most all 128 lights so 7 bits.
                        //       Pack the offset into the upper 24 bits and the count into the
                        //       lower 8 bits for convenience.
                        min_binding_size: BufferSize::new(16384),
                    },
                    count: None,
                },
            ],
            label: Some("pbr_view_layout"),
        });

        let material_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // TODO: change this to StandardMaterialUniformData::std140_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(64),
                    },
                    count: None,
                },
                // Base Color Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Base Color Texture Sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                // Emissive Texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Emissive Texture Sampler
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                // Metallic Roughness Texture
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Metallic Roughness Texture Sampler
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                // Occlusion Texture
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Occlusion Texture Sampler
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                // Normal Map Texture
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Normal Map Texture Sampler
                BindGroupLayoutEntry {
                    binding: 10,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
            ],
            label: Some("pbr_material_layout"),
        });

        let mesh_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    // TODO: change this to MeshUniform::std140_size_static once crevice fixes this!
                    // Context: https://github.com/LPGhatguy/crevice/issues/29
                    min_binding_size: BufferSize::new(144),
                },
                count: None,
            }],
            label: Some("pbr_mesh_layout"),
        });
        // A 1x1x1 'all 1.0' texture to use as a dummy texture to use in place of optional StandardMaterial textures
        let dummy_white_gpu_image = {
            let image = Image::new_fill(
                Extent3d::default(),
                TextureDimension::D2,
                &[255u8; 4],
                TextureFormat::bevy_default(),
            );
            let texture = render_device.create_texture(&image.texture_descriptor);
            let sampler = render_device.create_sampler(&image.sampler_descriptor);

            let format_size = image.texture_descriptor.format.pixel_size();
            let render_queue = world.get_resource_mut::<RenderQueue>().unwrap();
            render_queue.write_texture(
                ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &image.data,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZeroU32::new(
                            image.texture_descriptor.size.width * format_size as u32,
                        )
                        .unwrap(),
                    ),
                    rows_per_image: None,
                },
                image.texture_descriptor.size,
            );

            let texture_view = texture.create_view(&TextureViewDescriptor::default());
            GpuImage {
                texture,
                texture_view,
                sampler,
            }
        };
        PbrPipeline {
            view_layout,
            material_layout,
            mesh_layout,
            dummy_white_gpu_image,
        }
    }
}

bitflags::bitflags! {
    #[repr(transparent)]
    // NOTE: Apparently quadro drivers support up to 64x MSAA.
    /// MSAA uses the highest 6 bits for the MSAA sample count - 1 to support up to 64x MSAA.
    pub struct PbrPipelineKey: u32 {
        const NONE                        = 0;
        const VERTEX_TANGENTS             = (1 << 0);
        const STANDARDMATERIAL_NORMAL_MAP = (1 << 1);
        const OPAQUE_MAIN_PASS            = (1 << 2);
        const ALPHA_MASK_MAIN_PASS        = (1 << 3);
        const TRANSPARENT_MAIN_PASS       = (1 << 4);
        const MSAA_RESERVED_BITS          = PbrPipelineKey::MSAA_MASK_BITS << PbrPipelineKey::MSAA_SHIFT_BITS;
    }
}

impl PbrPipelineKey {
    const MSAA_MASK_BITS: u32 = 0b111111;
    const MSAA_SHIFT_BITS: u32 = 32 - 6;

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits = ((msaa_samples - 1) & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        PbrPipelineKey::from_bits(msaa_bits).unwrap()
    }

    pub fn msaa_samples(&self) -> u32 {
        ((self.bits >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS) + 1
    }
}

impl SpecializedPipeline for PbrPipeline {
    type Key = PbrPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let (vertex_array_stride, vertex_attributes) =
            if key.contains(PbrPipelineKey::VERTEX_TANGENTS) {
                (
                    48,
                    vec![
                        // Position (GOTCHA! Vertex_Position isn't first in the buffer due to how Mesh sorts attributes (alphabetically))
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 0,
                        },
                        // Normal
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 1,
                        },
                        // Uv (GOTCHA! uv is no longer third in the buffer due to how Mesh sorts attributes (alphabetically))
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 40,
                            shader_location: 2,
                        },
                        // Tangent
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 24,
                            shader_location: 3,
                        },
                    ],
                )
            } else {
                (
                    32,
                    vec![
                        // Position (GOTCHA! Vertex_Position isn't first in the buffer due to how Mesh sorts attributes (alphabetically))
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 0,
                        },
                        // Normal
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 1,
                        },
                        // Uv
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                )
            };
        let mut shader_defs = Vec::new();
        if key.contains(PbrPipelineKey::VERTEX_TANGENTS) {
            shader_defs.push(String::from("VERTEX_TANGENTS"));
        }
        if key.contains(PbrPipelineKey::STANDARDMATERIAL_NORMAL_MAP) {
            shader_defs.push(String::from("STANDARDMATERIAL_NORMAL_MAP"));
        }
        let (label, blend, depth_compare) = if key.contains(PbrPipelineKey::TRANSPARENT_MAIN_PASS) {
            (
                Some("transparent_pbr_pipeline".into()),
                Some(BlendState::ALPHA_BLENDING),
                // For the transparent pass, fragments that are closer will be alpha blended
                CompareFunction::Greater,
            )
        } else {
            (
                Some("opaque_pbr_pipeline".into()),
                Some(BlendState::REPLACE),
                // For the opaque and alpha mask passes, only the fragments at
                // the depth buffer depth will be shaded
                CompareFunction::Equal,
            )
        };
        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: PBR_SHADER_HANDLE.typed::<Shader>(),
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![VertexBufferLayout {
                    array_stride: vertex_array_stride,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vertex_attributes,
                }],
            },
            fragment: Some(FragmentState {
                shader: PBR_SHADER_HANDLE.typed::<Shader>(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend,
                    write_mask: ColorWrites::ALL,
                }],
            }),
            layout: Some(vec![
                self.view_layout.clone(),
                self.material_layout.clone(),
                self.mesh_layout.clone(),
            ]),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                clamp_depth: false,
                conservative: false,
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label,
        }
    }
}

pub struct TransformBindGroup {
    pub value: BindGroup,
}

pub fn queue_transform_bind_group(
    mut commands: Commands,
    pbr_pipeline: Res<PbrPipeline>,
    render_device: Res<RenderDevice>,
    transform_uniforms: Res<ComponentUniforms<MeshUniform>>,
) {
    if let Some(binding) = transform_uniforms.uniforms().binding() {
        commands.insert_resource(TransformBindGroup {
            value: render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: binding,
                }],
                label: Some("transform_bind_group"),
                layout: &pbr_pipeline.mesh_layout,
            }),
        });
    }
}

pub struct PbrViewBindGroup {
    pub value: BindGroup,
}

#[allow(clippy::too_many_arguments)]
pub fn queue_meshes(
    mut commands: Commands,
    draw_functions: (
        Res<DrawFunctions<Opaque3d>>,
        Res<DrawFunctions<AlphaMask3d>>,
        Res<DrawFunctions<Transparent3d>>,
    ),
    render_device: Res<RenderDevice>,
    pbr_pipeline: Res<PbrPipeline>,
    shadow_pipeline: Res<ShadowPipeline>,
    mut pipelines: ResMut<SpecializedPipelines<PbrPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    light_meta: Res<LightMeta>,
    msaa: Res<Msaa>,
    global_light_meta: Res<GlobalLightMeta>,
    view_uniforms: Res<ViewUniforms>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<StandardMaterial>>,
    standard_material_meshes: Query<(&Handle<StandardMaterial>, &Handle<Mesh>, &MeshUniform)>,
    mut views: Query<(
        Entity,
        &ExtractedView,
        &ViewShadowBindings,
        &ViewClusterBindings,
        &VisibleEntities,
        &mut RenderPhase<Opaque3d>,
        &mut RenderPhase<AlphaMask3d>,
        &mut RenderPhase<Transparent3d>,
    )>,
) {
    // dbg!(view_uniforms.uniforms.binding());
    // dbg!(light_meta.view_gpu_lights.binding());
    // dbg!(global_light_meta.gpu_point_lights.binding());
    if let (Some(view_binding), Some(light_binding), Some(point_light_binding)) = (
        view_uniforms.uniforms.binding(),
        light_meta.view_gpu_lights.binding(),
        global_light_meta.gpu_point_lights.binding(),
    ) {
        for (
            entity,
            view,
            view_shadow_bindings,
            view_cluster_bindings,
            visible_entities,
            mut opaque_phase,
            mut alpha_mask_phase,
            mut transparent_phase,
        ) in views.iter_mut()
        {
            // dbg!(&point_light_binding);
            // dbg!(&view_cluster_bindings
            //     .cluster_light_index_lists
            //     .binding()
            //     .unwrap());
            // dbg!(&view_cluster_bindings
            //     .cluster_offsets_and_counts
            //     .binding()
            //     .unwrap());
            let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: view_binding.clone(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: light_binding.clone(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(
                            &view_shadow_bindings.point_light_depth_texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&shadow_pipeline.point_light_sampler),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(
                            &view_shadow_bindings.directional_light_depth_texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: BindingResource::Sampler(
                            &shadow_pipeline.directional_light_sampler,
                        ),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: point_light_binding.clone(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: view_cluster_bindings
                            .cluster_light_index_lists
                            .binding()
                            .unwrap(),
                    },
                    BindGroupEntry {
                        binding: 8,
                        resource: view_cluster_bindings
                            .cluster_offsets_and_counts
                            .binding()
                            .unwrap(),
                    },
                ],
                label: Some("pbr_view_bind_group"),
                layout: &pbr_pipeline.view_layout,
            });

            commands.entity(entity).insert(PbrViewBindGroup {
                value: view_bind_group,
            });

            let (draw_opaque_pbr, draw_alpha_mask_pbr, draw_transparent_pbr) = (
                draw_functions.0.read().get_id::<DrawPbr>().unwrap(),
                draw_functions.1.read().get_id::<DrawPbr>().unwrap(),
                draw_functions.2.read().get_id::<DrawPbr>().unwrap(),
            );

            let inverse_view_matrix = view.transform.compute_matrix().inverse();
            let inverse_view_row_2 = inverse_view_matrix.row(2);

            for visible_entity in visible_entities.entities.iter().copied() {
                if let Ok((material_handle, mesh_handle, mesh_uniform)) =
                    standard_material_meshes.get(visible_entity)
                {
                    let mut key = PbrPipelineKey::from_msaa_samples(msaa.samples);
                    let alpha_mode = if let Some(material) = render_materials.get(material_handle) {
                        if material.has_normal_map {
                            key |= PbrPipelineKey::STANDARDMATERIAL_NORMAL_MAP;
                        }
                        material.alpha_mode.clone()
                    } else {
                        continue;
                    };
                    if let Some(mesh) = render_meshes.get(mesh_handle) {
                        if mesh.has_tangents {
                            key |= PbrPipelineKey::VERTEX_TANGENTS;
                        }
                    }
                    key |= match alpha_mode {
                        AlphaMode::Opaque => PbrPipelineKey::OPAQUE_MAIN_PASS,
                        AlphaMode::Mask(_) => PbrPipelineKey::ALPHA_MASK_MAIN_PASS,
                        AlphaMode::Blend => PbrPipelineKey::TRANSPARENT_MAIN_PASS,
                    };
                    let pipeline_id = pipelines.specialize(&mut pipeline_cache, &pbr_pipeline, key);

                    // NOTE: row 2 of the inverse view matrix dotted with column 3 of the model matrix
                    // gives the z component of translation of the mesh in view space
                    let mesh_z = inverse_view_row_2.dot(mesh_uniform.transform.col(3));
                    match alpha_mode {
                        AlphaMode::Opaque => {
                            opaque_phase.add(Opaque3d {
                                entity: visible_entity,
                                draw_function: draw_opaque_pbr,
                                pipeline: pipeline_id,
                                // NOTE: Front-to-back ordering for opaque with ascending sort means near should have the
                                //       lowest sort key and getting further away should increase. As we have
                                //       -z in front of the camera, values in view space decrease away from the
                                //       camera. Flipping the sign of mesh_z results in the correct front-to-back ordering
                                distance: -mesh_z,
                            });
                        }
                        AlphaMode::Mask(_) => {
                            alpha_mask_phase.add(AlphaMask3d {
                                entity: visible_entity,
                                draw_function: draw_alpha_mask_pbr,
                                pipeline: pipeline_id,
                                // NOTE: Front-to-back ordering for alpha mask with ascending sort means near should have the
                                //       lowest sort key and getting further away should increase. As we have
                                //       -z in front of the camera, values in view space decrease away from the
                                //       camera. Flipping the sign of mesh_z results in the correct front-to-back ordering
                                distance: -mesh_z,
                            });
                        }
                        AlphaMode::Blend => {
                            transparent_phase.add(Transparent3d {
                                entity: visible_entity,
                                draw_function: draw_transparent_pbr,
                                pipeline: pipeline_id,
                                // NOTE: Back-to-front ordering for transparent with ascending sort means far should have the
                                //       lowest sort key and getting closer should increase. As we have
                                //       -z in front of the camera, the largest distance is -far with values increasing toward the
                                //       camera. As such we can just use mesh_z as the distance
                                distance: mesh_z,
                            });
                        }
                    }
                }
            }
        }
    }
}

pub type DrawPbr = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetStandardMaterialBindGroup<1>,
    SetTransformBindGroup<2>,
    DrawMesh,
);

pub struct SetMeshViewBindGroup<const I: usize>;
impl<T: PhaseItem, const I: usize> RenderCommand<T> for SetMeshViewBindGroup<I> {
    type Param = SQuery<(
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<PbrViewBindGroup>,
    )>;
    #[inline]
    fn render<'w>(
        view: Entity,
        _item: &T,
        view_query: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let (view_uniform, view_lights, pbr_view_bind_group) = view_query.get(view).unwrap();
        pass.set_bind_group(
            I,
            &pbr_view_bind_group.value,
            &[view_uniform.offset, view_lights.offset],
        );
    }
}

pub struct SetTransformBindGroup<const I: usize>;
impl<T: EntityPhaseItem + PhaseItem, const I: usize> RenderCommand<T> for SetTransformBindGroup<I> {
    type Param = (
        SRes<TransformBindGroup>,
        SQuery<Read<DynamicUniformIndex<MeshUniform>>>,
    );
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: &T,
        (transform_bind_group, mesh_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let transform_index = mesh_query.get(item.entity()).unwrap();
        pass.set_bind_group(
            I,
            &transform_bind_group.into_inner().value,
            &[transform_index.index()],
        );
    }
}

pub struct SetStandardMaterialBindGroup<const I: usize>;
impl<T: EntityPhaseItem + PhaseItem, const I: usize> RenderCommand<T>
    for SetStandardMaterialBindGroup<I>
{
    type Param = (
        SRes<RenderAssets<StandardMaterial>>,
        SQuery<Read<Handle<StandardMaterial>>>,
    );
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: &T,
        (materials, handle_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let handle = handle_query.get(item.entity()).unwrap();
        let materials = materials.into_inner();
        let material = materials.get(handle).unwrap();

        pass.set_bind_group(I, &material.bind_group, &[]);
    }
}

pub struct DrawMesh;
impl<T: EntityPhaseItem + PhaseItem> RenderCommand<T> for DrawMesh {
    type Param = (SRes<RenderAssets<Mesh>>, SQuery<Read<Handle<Mesh>>>);
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: &T,
        (meshes, mesh_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let mesh_handle = mesh_query.get(item.entity()).unwrap();
        let gpu_mesh = meshes.into_inner().get(mesh_handle).unwrap();
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        if let Some(index_info) = &gpu_mesh.index_info {
            pass.set_index_buffer(index_info.buffer.slice(..), 0, index_info.index_format);
            pass.draw_indexed(0..index_info.count, 0, 0..1);
        } else {
            panic!("non-indexed drawing not supported yet")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PbrPipelineKey;
    #[test]
    fn pbr_key_msaa_samples() {
        for i in 1..=64 {
            assert_eq!(PbrPipelineKey::from_msaa_samples(i).msaa_samples(), i);
        }
    }
}
