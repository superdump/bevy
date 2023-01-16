use bevy_app::Plugin;
use bevy_asset::{load_internal_asset, Handle, HandleId, HandleUntyped};
use bevy_core_pipeline::core_2d::Transparent2d;
use bevy_ecs::{
    prelude::*,
    query::ROQueryItem,
    system::{lifetimeless::*, SystemParamItem, SystemState},
};
use bevy_math::{Mat4, Vec2, Vec4};
use bevy_reflect::{Reflect, TypeUuid, Uuid};
use bevy_render::{
    extract_component::{ComponentVecUniforms, MAX_REASONABLE_UNIFORM_BUFFER_BINDING_SIZE},
    globals::{GlobalsBuffer, GlobalsUniform},
    mesh::{GpuBufferInfo, Mesh, MeshVertexBufferLayout},
    render_asset::RenderAssets,
    render_phase::{
        BatchedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem, RenderCommand,
        RenderCommandResult, RenderPhase, TrackedRenderPass,
    },
    render_resource::*,
    renderer::{render_system, RenderDevice, RenderQueue},
    texture::{
        BevyDefault, DefaultImageSampler, GpuImage, Image, ImageSampler, TextureFormatPixelInfo,
    },
    view::{
        ComputedVisibility, ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset,
        ViewUniforms, VisibleEntities,
    },
    Extract, RenderApp, RenderStage,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{hashbrown, PassHash};

use crate::{Material2dBindingMeta, Material2dHandle};

/// Component for rendering with meshes in the 2d pipeline, usually with a [2d material](crate::Material2d) such as [`ColorMaterial`](crate::ColorMaterial).
///
/// It wraps a [`Handle<Mesh>`] to differentiate from the 3d pipelines which use the handles directly as components
#[derive(Default, Clone, Component, Debug, Reflect, PartialEq, Eq)]
#[reflect(Component)]
pub struct Mesh2dHandle(pub Handle<Mesh>);

impl From<Handle<Mesh>> for Mesh2dHandle {
    fn from(handle: Handle<Mesh>) -> Self {
        Self(handle)
    }
}

#[derive(Default)]
pub struct Mesh2dRenderPlugin;

pub const MESH2D_VERTEX_OUTPUT: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 7646632476603252194);
pub const MESH2D_VIEW_TYPES_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 12677582416765805110);
pub const MESH2D_VIEW_BINDINGS_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 6901431444735842434);
pub const MESH2D_TYPES_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 8994673400261890424);
pub const MESH2D_BINDINGS_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 8983617858458862856);
pub const MESH2D_FUNCTIONS_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 4976379308250389413);
pub const MESH2D_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 2971387252468633715);

impl Plugin for Mesh2dRenderPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        load_internal_asset!(
            app,
            MESH2D_VERTEX_OUTPUT,
            "mesh2d_vertex_output.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH2D_VIEW_TYPES_HANDLE,
            "mesh2d_view_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH2D_VIEW_BINDINGS_HANDLE,
            "mesh2d_view_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH2D_TYPES_HANDLE,
            "mesh2d_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH2D_BINDINGS_HANDLE,
            "mesh2d_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH2D_FUNCTIONS_HANDLE,
            "mesh2d_functions.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, MESH2D_SHADER_HANDLE, "mesh2d.wgsl", Shader::from_wgsl);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            let limits = render_app
                .world
                .get_resource::<RenderDevice>()
                .expect(
                    "RenderDevice Resource must exist before a UniformComponentVecPlugin is added",
                )
                .limits();

            render_app
                // NOTE: ComponentVecUniforms<Mesh2dUniform> must be inserted before
                // the Mesh2dPipeline resource as Mesh2dPipeline initialization uses
                // information from it
                .insert_resource(ComponentVecUniforms::<Mesh2dUniform>::new(
                    (limits
                        .max_uniform_buffer_binding_size
                        .min(MAX_REASONABLE_UNIFORM_BUFFER_BINDING_SIZE)
                        as u64
                        / Mesh2dUniform::min_size().get()) as usize,
                    limits.min_uniform_buffer_offset_alignment,
                ))
                .init_resource::<Mesh2dPipeline>()
                .init_resource::<ExtractedMeshes2d>()
                .init_resource::<SpecializedMeshPipelines<Mesh2dPipeline>>()
                .add_system_to_stage(RenderStage::Prepare, prepare_mesh2d_uniforms)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    queue_mesh2d_bind_group.after(prepare_mesh2d_uniforms),
                )
                .add_system_to_stage(RenderStage::Prepare, queue_mesh2d_view_bind_groups);
        }
    }
}

/// Data necessary to be equal for two draw commands to be mergeable
///
/// This is based on the following assumptions:
/// - Only entities with prepared assets (pipelines, materials, meshes) are
///   queued to phases
/// - View bindings are constant across a phase for a given draw function as
///   phases are per-view
/// - `prepare_mesh_uniforms` is the only system that performs this batching
///   and has sole responsibility for preparing the per-object data. As such
///   the mesh binding and dynamic offsets are assumed to only be variable as a
///   result of the `prepare_mesh_uniforms` system, e.g. due to having to split
///   data across separate uniform bindings within the same buffer due to the
///   maximum uniform buffer binding size.
#[derive(Default, PartialEq, Eq)]
struct BatchMeta<'mat> {
    /// The pipeline id encompasses all pipeline configuration including vertex
    /// buffers and layouts, shaders and their specializations, bind group
    /// layouts, etc.
    pipeline_id: Option<CachedRenderPipelineId>,
    /// The draw function id defines the RenderCommands that are called to
    /// set the pipeline and bindings, and make the draw command
    draw_function_id: Option<DrawFunctionId>,
    dynamic_offset: u32,
    mesh_handle: Option<HandleId>,
    /// The material binding meta includes the material bind group id and
    /// dynamic offsets.
    material2d_binding_meta: Option<&'mat Material2dBindingMeta>,
}

// impl<'mat> BatchMeta<'mat> {
//     fn matches(&self, other: &BatchMeta<'mat>, consider_material: bool) -> bool {
//         self.pipeline_id == other.pipeline_id
//             && self.draw_function_id == other.draw_function_id
//             && self.mesh_handle == other.mesh_handle
//             && self.dynamic_offset == other.dynamic_offset
//             && (!consider_material || self.material2d_binding_meta == other.material2d_binding_meta)
//     }
// }

#[derive(Default)]
struct BatchState<'mat> {
    meta: BatchMeta<'mat>,
    /// The base index in the object data binding's array
    index: u32,
    /// The number of entities in the batch
    count: u32,
    item_index: usize,
}

#[inline]
fn update_batch_data<I: BatchedPhaseItem>(item: &mut I, batch: &BatchState) {
    let &BatchState {
        meta: BatchMeta {
            dynamic_offset: offset,
            ..
        },
        index,
        count,
        ..
    } = batch;
    *item.batch_dynamic_offset_mut() = Some(offset);
    *item.batch_range_mut() = Some(index..index + count);
}

fn process_phase<I: CachedRenderPipelinePhaseItem + BatchedPhaseItem>(
    batches: &mut Vec<(Entity, (Material2dHandle, Mesh2dHandle))>,
    object_data_buffer: &mut ComponentVecUniforms<Mesh2dUniform>,
    // object_query: &mut ObjectQuery,
    extracted_meshes: &mut ExtractedMeshes2d,
    phase: &mut RenderPhase<I>,
    consider_material: bool,
) {
    let mut batch = BatchState::default();
    // dbg!(phase.items.len());
    // let flags = MeshFlags::NONE.bits;
    let mut i = 0;
    while i < phase.items.len() {
        let item = &mut phase.items[i];
        let Some(extracted_mesh) = extracted_meshes.meshes.get(&item.entity().to_bits()) else {continue;};
        let transform = extracted_mesh.transform.affine();
        // let indices = object_data_buffer.push(Mesh2dUniform {
        //     transform: Mat4::from(transform),
        //     inverse_transpose_model: Mat4::from(transform.inverse()).transpose(),
        //     flags,
        // });
        let indices = object_data_buffer.push(Mesh2dUniform {
            local_to_world: [
                transform.matrix3.x_axis.extend(transform.translation.x),
                transform.matrix3.y_axis.extend(transform.translation.y),
                transform.matrix3.z_axis.extend(transform.translation.z),
            ],
        });
        // let batch_meta = BatchMeta {
        //     pipeline_id: Some(item.cached_pipeline()),
        //     draw_function_id: Some(item.draw_function()),
        //     material2d_binding_meta: Some(&extracted_mesh.material2d_binding_meta),
        //     mesh_handle: Some(extracted_mesh.mesh_handle_id),
        //     dynamic_offset: indices.offset(),
        // };
        if indices.offset() != batch.meta.dynamic_offset
            || Some(extracted_mesh.mesh_handle_id) != batch.meta.mesh_handle
            || Some(&extracted_mesh.material2d_binding_meta) != batch.meta.material2d_binding_meta
            || Some(item.draw_function()) != batch.meta.draw_function_id
            || Some(item.cached_pipeline()) != batch.meta.pipeline_id
        {
            let pipeline_id = item.cached_pipeline();
            let draw_function_id = item.draw_function();
            // !batch_meta.matches(&batch.meta, consider_material) {
            if batch.count > 0 {
                update_batch_data(&mut phase.items[batch.item_index], &batch);
                let entity = phase.items[batch.item_index].entity();
                let extracted_mesh = extracted_meshes.meshes.get(&entity.to_bits()).unwrap();
                batches.push((
                    entity,
                    (
                        Material2dHandle(extracted_mesh.material_handle_id),
                        Mesh2dHandle(Handle::weak(extracted_mesh.mesh_handle_id)),
                    ),
                ));
            }

            batch.meta = BatchMeta {
                pipeline_id: Some(pipeline_id),
                draw_function_id: Some(draw_function_id),
                material2d_binding_meta: Some(&extracted_mesh.material2d_binding_meta),
                mesh_handle: Some(extracted_mesh.mesh_handle_id),
                dynamic_offset: indices.offset(),
            };
            batch.index = indices.index();
            batch.count = 0;
            batch.item_index = i;
        }
        batch.count += 1;
        i += 1;
    }
    if !phase.items.is_empty() {
        update_batch_data(&mut phase.items[batch.item_index], &batch);
        let entity = phase.items[batch.item_index].entity();
        let extracted_mesh = extracted_meshes.meshes.get(&entity.to_bits()).unwrap();
        batches.push((
            entity,
            (
                Material2dHandle(extracted_mesh.material_handle_id),
                Mesh2dHandle(Handle::weak(extracted_mesh.mesh_handle_id)),
            ),
        ));
    }
}

type ObjectQuery<'w, 's, 'mat, 'mesh, 'data> = Query<
    'w,
    's,
    (
        Option<&'mat Material2dBindingMeta>,
        &'mesh Mesh2dHandle,
        &'data Mesh2dUniform,
    ),
>;

/// This system prepares all components of the corresponding component type.
/// They are transformed into uniforms and stored in the [`ComponentVecUniforms`] resource.
fn prepare_mesh2d_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    object_data_buffer: ResMut<ComponentVecUniforms<Mesh2dUniform>>,
    // mut object_query: ObjectQuery,
    extracted_meshes: ResMut<ExtractedMeshes2d>,
    mut views: Query<(&VisibleEntities, &mut RenderPhase<Transparent2d>)>,
    mut previous_len: Local<usize>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy_utils::tracing::info_span!("vec_alloc").entered();
    let mut batches = Vec::with_capacity(*previous_len);
    #[cfg(feature = "trace")]
    drop(_span);

    let object_data_buffer = object_data_buffer.into_inner();
    object_data_buffer.clear();

    #[cfg(feature = "trace")]
    let _span = bevy_utils::tracing::info_span!("sort_views").entered();
    let mut views = views.iter_mut().collect::<Vec<_>>();
    views.sort_unstable_by(|a, b| a.0.len().cmp(&b.0.len()));
    #[cfg(feature = "trace")]
    drop(_span);

    let extracted_meshes = extracted_meshes.into_inner();
    for (_, transparent) in views.into_iter() {
        #[cfg(feature = "trace")]
        let _span = bevy_utils::tracing::info_span!("process_phase").entered();
        process_phase(
            &mut batches,
            object_data_buffer,
            // &mut object_query,
            extracted_meshes,
            transparent.into_inner(),
            true,
        );
        #[cfg(feature = "trace")]
        drop(_span);
    }

    *previous_len = batches.len();
    commands.insert_or_spawn_batch(batches);

    #[cfg(feature = "trace")]
    let _span = bevy_utils::tracing::info_span!("write_buffer").entered();
    object_data_buffer.write_buffer(&render_device, &render_queue);
    #[cfg(feature = "trace")]
    drop(_span);
}

#[derive(Component, ShaderType, Clone, Copy)]
pub struct Mesh2dUniform {
    /// Affine transform packed as column-major 3x3 in the xyz elements
    /// and translation in the w elements
    pub local_to_world: [Vec4; 3],
}
// #[derive(Component, ShaderType, Clone, Copy)]
// pub struct Mesh2dUniform {
//     pub transform: Mat4,
//     pub inverse_transpose_model: Mat4,
//     pub flags: u32,
// }

// NOTE: These must match the bit flags in bevy_sprite/src/mesh2d/mesh2d.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    struct MeshFlags: u32 {
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

#[derive(Component, Clone)]
pub struct ExtractedMaterialMesh2d {
    pub entity: Entity,
    pub transform: GlobalTransform,
    /// PERF: storing a `HandleId` instead of `Handle<Image>` enables some optimizations (`ExtractedSprite` becomes `Copy` and doesn't need to be dropped)
    pub mesh_handle_id: HandleId,
    /// PERF: storing a `HandleId` instead of `Handle<Image>` enables some optimizations (`ExtractedSprite` becomes `Copy` and doesn't need to be dropped)
    pub material_handle_id: HandleId,
    pub material2d_binding_meta: Material2dBindingMeta,
}

#[derive(Default, Resource)]
pub struct ExtractedMeshes2d {
    pub meshes: hashbrown::HashMap<u64, ExtractedMaterialMesh2d, PassHash>,
}

#[derive(Resource, Clone)]
pub struct Mesh2dPipeline {
    pub view_layout: BindGroupLayout,
    pub mesh_layout: BindGroupLayout,
    // This dummy white texture is to be used in place of optional textures
    pub dummy_white_gpu_image: GpuImage,
    pub mesh2d_uniform_array_len: u32,
}

impl FromWorld for Mesh2dPipeline {
    fn from_world(world: &mut World) -> Self {
        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<DefaultImageSampler>,
            Res<ComponentVecUniforms<Mesh2dUniform>>,
        )> = SystemState::new(world);
        let (render_device, default_sampler, mesh2d_uniforms) = system_state.get_mut(world);
        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GlobalsUniform::min_size()),
                    },
                    count: None,
                },
            ],
            label: Some("mesh2d_view_layout"),
        });

        let mesh_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(mesh2d_uniforms.size()),
                },
                count: None,
            }],
            label: Some("mesh2d_layout"),
        });

        let mesh2d_uniform_array_len = (render_device
            .limits()
            .max_uniform_buffer_binding_size
            .min(MAX_REASONABLE_UNIFORM_BUFFER_BINDING_SIZE)
            as u64
            / Mesh2dUniform::min_size().get()) as u32;

        // A 1x1x1 'all 1.0' texture to use as a dummy texture to use in place of optional StandardMaterial textures
        let dummy_white_gpu_image = {
            let image = Image::new_fill(
                Extent3d::default(),
                TextureDimension::D2,
                &[255u8; 4],
                TextureFormat::bevy_default(),
            );
            let texture = render_device.create_texture(&image.texture_descriptor);
            let sampler = match image.sampler_descriptor {
                ImageSampler::Default => (**default_sampler).clone(),
                ImageSampler::Descriptor(descriptor) => render_device.create_sampler(&descriptor),
            };

            let format_size = image.texture_descriptor.format.pixel_size();
            let render_queue = world.resource_mut::<RenderQueue>();
            render_queue.write_texture(
                ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
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
                texture_format: image.texture_descriptor.format,
                sampler,
                size: Vec2::new(
                    image.texture_descriptor.size.width as f32,
                    image.texture_descriptor.size.height as f32,
                ),
            }
        };
        Mesh2dPipeline {
            view_layout,
            mesh_layout,
            dummy_white_gpu_image,
            mesh2d_uniform_array_len,
        }
    }
}

impl Mesh2dPipeline {
    pub fn get_image_texture<'a>(
        &'a self,
        gpu_images: &'a RenderAssets<Image>,
        handle_option: &Option<Handle<Image>>,
    ) -> Option<(&'a TextureView, &'a Sampler)> {
        if let Some(handle) = handle_option {
            let gpu_image = gpu_images.get(handle)?;
            Some((&gpu_image.texture_view, &gpu_image.sampler))
        } else {
            Some((
                &self.dummy_white_gpu_image.texture_view,
                &self.dummy_white_gpu_image.sampler,
            ))
        }
    }
}

bitflags::bitflags! {
    #[repr(transparent)]
    // NOTE: Apparently quadro drivers support up to 64x MSAA.
    // MSAA uses the highest 3 bits for the MSAA log2(sample count) to support up to 128x MSAA.
    // FIXME: make normals optional?
    pub struct Mesh2dPipelineKey: u32 {
        const NONE                        = 0;
        const HDR                         = (1 << 0);
        const TONEMAP_IN_SHADER           = (1 << 1);
        const DEBAND_DITHER               = (1 << 2);
        const MSAA_RESERVED_BITS          = Self::MSAA_MASK_BITS << Self::MSAA_SHIFT_BITS;
        const PRIMITIVE_TOPOLOGY_RESERVED_BITS = Self::PRIMITIVE_TOPOLOGY_MASK_BITS << Self::PRIMITIVE_TOPOLOGY_SHIFT_BITS;
    }
}

impl Mesh2dPipelineKey {
    const MSAA_MASK_BITS: u32 = 0b111;
    const MSAA_SHIFT_BITS: u32 = 32 - Self::MSAA_MASK_BITS.count_ones();
    const PRIMITIVE_TOPOLOGY_MASK_BITS: u32 = 0b111;
    const PRIMITIVE_TOPOLOGY_SHIFT_BITS: u32 = Self::MSAA_SHIFT_BITS - 3;

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits =
            (msaa_samples.trailing_zeros() & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        Self::from_bits(msaa_bits).unwrap()
    }

    pub fn from_hdr(hdr: bool) -> Self {
        if hdr {
            Mesh2dPipelineKey::HDR
        } else {
            Mesh2dPipelineKey::NONE
        }
    }

    pub fn msaa_samples(&self) -> u32 {
        1 << ((self.bits >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS)
    }

    pub fn from_primitive_topology(primitive_topology: PrimitiveTopology) -> Self {
        let primitive_topology_bits = ((primitive_topology as u32)
            & Self::PRIMITIVE_TOPOLOGY_MASK_BITS)
            << Self::PRIMITIVE_TOPOLOGY_SHIFT_BITS;
        Self::from_bits(primitive_topology_bits).unwrap()
    }

    pub fn primitive_topology(&self) -> PrimitiveTopology {
        let primitive_topology_bits =
            (self.bits >> Self::PRIMITIVE_TOPOLOGY_SHIFT_BITS) & Self::PRIMITIVE_TOPOLOGY_MASK_BITS;
        match primitive_topology_bits {
            x if x == PrimitiveTopology::PointList as u32 => PrimitiveTopology::PointList,
            x if x == PrimitiveTopology::LineList as u32 => PrimitiveTopology::LineList,
            x if x == PrimitiveTopology::LineStrip as u32 => PrimitiveTopology::LineStrip,
            x if x == PrimitiveTopology::TriangleList as u32 => PrimitiveTopology::TriangleList,
            x if x == PrimitiveTopology::TriangleStrip as u32 => PrimitiveTopology::TriangleStrip,
            _ => PrimitiveTopology::default(),
        }
    }
}

impl SpecializedMeshPipeline for Mesh2dPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut shader_defs = Vec::new();
        let mut vertex_attributes = Vec::new();

        if layout.contains(Mesh::ATTRIBUTE_POSITION) {
            shader_defs.push("VERTEX_POSITIONS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_POSITION.at_shader_location(0));
        }

        if layout.contains(Mesh::ATTRIBUTE_NORMAL) {
            shader_defs.push("VERTEX_NORMALS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_NORMAL.at_shader_location(1));
        }

        if layout.contains(Mesh::ATTRIBUTE_UV_0) {
            shader_defs.push("VERTEX_UVS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(2));
        }

        if layout.contains(Mesh::ATTRIBUTE_TANGENT) {
            shader_defs.push("VERTEX_TANGENTS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_TANGENT.at_shader_location(3));
        }

        if layout.contains(Mesh::ATTRIBUTE_COLOR) {
            shader_defs.push("VERTEX_COLORS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_COLOR.at_shader_location(4));
        }

        if key.contains(Mesh2dPipelineKey::TONEMAP_IN_SHADER) {
            shader_defs.push("TONEMAP_IN_SHADER".into());

            // Debanding is tied to tonemapping in the shader, cannot run without it.
            if key.contains(Mesh2dPipelineKey::DEBAND_DITHER) {
                shader_defs.push("DEBAND_DITHER".into());
            }
        }

        shader_defs.push(ShaderDefVal::UInt(
            "MESHES_2D_UNIFORM_ARRAY_LEN".to_string(),
            self.mesh2d_uniform_array_len,
        ));

        let vertex_buffer_layout = layout.get_layout(&vertex_attributes)?;

        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        Ok(RenderPipelineDescriptor {
            vertex: VertexState {
                shader: MESH2D_SHADER_HANDLE.typed::<Shader>(),
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: MESH2D_SHADER_HANDLE.typed::<Shader>(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout: Some(vec![self.view_layout.clone(), self.mesh_layout.clone()]),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: key.primitive_topology(),
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("transparent_mesh2d_pipeline".into()),
        })
    }
}

#[derive(Resource)]
pub struct Mesh2dBindGroup {
    pub value: BindGroup,
}

pub fn queue_mesh2d_bind_group(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    render_device: Res<RenderDevice>,
    mesh2d_uniforms: Res<ComponentVecUniforms<Mesh2dUniform>>,
) {
    if let Some(binding) = mesh2d_uniforms.binding() {
        commands.insert_resource(Mesh2dBindGroup {
            value: render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: binding,
                }],
                label: Some("mesh2d_bind_group"),
                layout: &mesh2d_pipeline.mesh_layout,
            }),
        });
    }
}

#[derive(Component)]
pub struct Mesh2dViewBindGroup {
    pub value: BindGroup,
}

pub fn queue_mesh2d_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<Entity, With<ExtractedView>>,
    globals_buffer: Res<GlobalsBuffer>,
) {
    if let (Some(view_binding), Some(globals)) = (
        view_uniforms.uniforms.binding(),
        globals_buffer.buffer.binding(),
    ) {
        for entity in &views {
            let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: view_binding.clone(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: globals.clone(),
                    },
                ],
                label: Some("mesh2d_view_bind_group"),
                layout: &mesh2d_pipeline.view_layout,
            });

            commands.entity(entity).insert(Mesh2dViewBindGroup {
                value: view_bind_group,
            });
        }
    }
}

pub struct SetMesh2dViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetMesh2dViewBindGroup<I> {
    type Param = ();
    type ViewWorldQuery = (Read<ViewUniformOffset>, Read<Mesh2dViewBindGroup>);
    type ItemWorldQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        (view_uniform, mesh2d_view_bind_group): ROQueryItem<'w, Self::ViewWorldQuery>,
        _view: (),
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &mesh2d_view_bind_group.value, &[view_uniform.offset]);

        RenderCommandResult::Success
    }
}

pub struct SetMesh2dBindGroup<const I: usize>;
impl<P: BatchedPhaseItem, const I: usize> RenderCommand<P> for SetMesh2dBindGroup<I> {
    type Param = SRes<Mesh2dBindGroup>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _: ROQueryItem<'_, Self::ItemWorldQuery>,
        mesh2d_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            &mesh2d_bind_group.into_inner().value,
            &[item.batch_dynamic_offset().unwrap()],
        );
        RenderCommandResult::Success
    }
}

pub struct DrawMesh2d;
impl<P: BatchedPhaseItem> RenderCommand<P> for DrawMesh2d {
    type Param = SRes<RenderAssets<Mesh>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<Mesh2dHandle>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        mesh_handle: ROQueryItem<'w, Self::ItemWorldQuery>,
        meshes: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        if let Some(gpu_mesh) = meshes.into_inner().get(&mesh_handle.0) {
            pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            let batch_range = item.batch_range().as_ref().unwrap();
            match &gpu_mesh.buffer_info {
                GpuBufferInfo::Indexed {
                    buffer,
                    index_format,
                    count,
                } => {
                    pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                    pass.draw_indexed(0..*count, 0, batch_range.clone());
                }
                GpuBufferInfo::NonIndexed { vertex_count } => {
                    pass.draw(0..*vertex_count, batch_range.clone());
                }
            }
            RenderCommandResult::SuccessfulDraw(batch_range.len())
        } else {
            RenderCommandResult::Failure
        }
    }
}
