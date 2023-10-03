use std::ops::Range;

use bevy_app::Plugin;
use bevy_asset::{load_internal_asset, AssetId, Handle};

use bevy_core_pipeline::{
    clear_color::{self, ClearColor, ClearColorConfig},
    core_2d::{BatchQueue, BatchStruct, Camera2d, DrawOperations, DrawStream, DynamicOffsets},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    prelude::*,
    query::{QueryItem, ROQueryItem},
    system::{lifetimeless::*, SystemParamItem, SystemState},
};
use bevy_math::{Affine3, Vec2, Vec4};
use bevy_reflect::Reflect;
use bevy_render::{
    batching::{write_batched_instance_buffer, GetBatchData, NoAutomaticBatching},
    camera::ExtractedCamera,
    globals::{GlobalsBuffer, GlobalsUniform},
    mesh::{GpuBufferInfo, Mesh, MeshVertexBufferLayout},
    render_asset::{
        prepare_assets, InnerRenderAssetKey, RenderAssetKey, RenderAssetKeySender,
        RenderAssetKeyUpdates, RenderAssets,
    },
    render_phase::{PhaseItem, RenderCommand, RenderCommandResult, TrackedRenderPass},
    render_resource::*,
    renderer::{render_system, RenderContext, RenderDevice, RenderQueue},
    texture::{
        BevyDefault, DefaultImageSampler, GpuImage, Image, ImageSampler, TextureFormatPixelInfo,
    },
    view::{
        ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms, ViewVisibility,
    },
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{
    slotmap::{Key, KeyData},
    EntityHashMap, HashMap,
};

use crate::{ColorMaterial, Material2dBindGroupId, RenderMaterial2dInstances, RenderMaterials2d};

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

pub const MESH2D_VERTEX_OUTPUT: Handle<Shader> = Handle::weak_from_u128(7646632476603252194);
pub const MESH2D_VIEW_TYPES_HANDLE: Handle<Shader> = Handle::weak_from_u128(12677582416765805110);
pub const MESH2D_VIEW_BINDINGS_HANDLE: Handle<Shader> = Handle::weak_from_u128(6901431444735842434);
pub const MESH2D_TYPES_HANDLE: Handle<Shader> = Handle::weak_from_u128(8994673400261890424);
pub const MESH2D_BINDINGS_HANDLE: Handle<Shader> = Handle::weak_from_u128(8983617858458862856);
pub const MESH2D_FUNCTIONS_HANDLE: Handle<Shader> = Handle::weak_from_u128(4976379308250389413);
pub const MESH2D_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(2971387252468633715);

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
            MESH2D_FUNCTIONS_HANDLE,
            "mesh2d_functions.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, MESH2D_SHADER_HANDLE, "mesh2d.wgsl", Shader::from_wgsl);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<RenderMesh2dInstances>()
                .init_resource::<SpecializedMeshPipelines<Mesh2dPipeline>>()
                .add_systems(ExtractSchedule, extract_mesh2d)
                .add_systems(
                    Render,
                    (
                        batch_and_prepare_batch_queue.in_set(RenderSet::PrepareResources),
                        update_render_world_render_mesh_asset_keys
                            .in_set(RenderSet::PrepareAssets)
                            .after(prepare_assets::<Mesh>),
                        write_batched_instance_buffer::<Mesh2dPipeline>
                            .in_set(RenderSet::PrepareResourcesFlush),
                        prepare_mesh2d_bind_group.in_set(RenderSet::PrepareBindGroups),
                        prepare_mesh2d_view_bind_groups.in_set(RenderSet::PrepareBindGroups),
                        render_from_draw_streams
                            .in_set(RenderSet::Render)
                            .after(PipelineCache::process_pipeline_queue_system)
                            .before(render_system),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut bevy_app::App) {
        let mut mesh_bindings_shader_defs = Vec::with_capacity(1);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            if let Some(per_object_buffer_batch_size) = GpuArrayBuffer::<Mesh2dUniform>::batch_size(
                render_app.world.resource::<RenderDevice>(),
            ) {
                mesh_bindings_shader_defs.push(ShaderDefVal::UInt(
                    "PER_OBJECT_BUFFER_BATCH_SIZE".into(),
                    per_object_buffer_batch_size,
                ));
            }

            render_app
                .insert_resource(GpuArrayBuffer::<Mesh2dUniform>::new(
                    render_app.world.resource::<RenderDevice>(),
                ))
                .init_resource::<Mesh2dPipeline>();
        }

        // Load the mesh_bindings shader module here as it depends on runtime information about
        // whether storage buffers are supported, or the maximum uniform buffer binding size.
        load_internal_asset!(
            app,
            MESH2D_BINDINGS_HANDLE,
            "mesh2d_bindings.wgsl",
            Shader::from_wgsl_with_defs,
            mesh_bindings_shader_defs
        );
    }
}

#[derive(Component)]
pub struct Mesh2dTransforms {
    pub transform: Affine3,
    pub flags: u32,
}

#[derive(ShaderType, Clone)]
pub struct Mesh2dUniform {
    // Affine 4x3 matrix transposed to 3x4
    pub transform: [Vec4; 3],
    // 3x3 matrix packed in mat2x4 and f32 as:
    //   [0].xyz, [1].x,
    //   [1].yz, [2].xy
    //   [2].z
    pub inverse_transpose_model_a: [Vec4; 2],
    pub inverse_transpose_model_b: f32,
    pub flags: u32,
}

impl From<&Mesh2dTransforms> for Mesh2dUniform {
    fn from(mesh_transforms: &Mesh2dTransforms) -> Self {
        let (inverse_transpose_model_a, inverse_transpose_model_b) =
            mesh_transforms.transform.inverse_transpose_3x3();
        Self {
            transform: mesh_transforms.transform.to_transpose(),
            inverse_transpose_model_a,
            inverse_transpose_model_b,
            flags: mesh_transforms.flags,
        }
    }
}

// NOTE: These must match the bit flags in bevy_sprite/src/mesh2d/mesh2d.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    struct MeshFlags: u32 {
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

pub struct RenderMesh2dInstance {
    pub transforms: Mesh2dTransforms,
    pub mesh_asset_id: Option<AssetId<Mesh>>,
    pub mesh_asset_key: Option<RenderAssetKey<Mesh>>,
    pub material_bind_group_id: Material2dBindGroupId,
    pub automatic_batching: bool,
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct RenderMesh2dInstances(EntityHashMap<Entity, RenderMesh2dInstance>);

#[derive(Component)]
pub struct Mesh2d;

pub fn extract_mesh2d(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    query: Extract<
        Query<(
            Entity,
            &ViewVisibility,
            &GlobalTransform,
            Ref<Mesh2dHandle>,
            Option<&RenderAssetKey<Mesh>>,
            Has<NoAutomaticBatching>,
        )>,
    >,
) {
    render_mesh_instances.clear();
    let mut entities = Vec::with_capacity(*previous_len);

    for (entity, view_visibility, transform, handle, mesh_asset_key, no_automatic_batching) in
        &query
    {
        if !view_visibility.get() {
            continue;
        }
        // FIXME: Remove this - it is just a workaround to enable rendering to work as
        // render commands require an entity to exist at the moment.
        entities.push((entity, Mesh2d));
        render_mesh_instances.insert(
            entity,
            RenderMesh2dInstance {
                transforms: Mesh2dTransforms {
                    transform: (&transform.affine()).into(),
                    flags: MeshFlags::empty().bits(),
                },
                mesh_asset_id: if mesh_asset_key.is_none() || handle.is_changed() {
                    Some(handle.0.id())
                } else {
                    None
                },
                mesh_asset_key: mesh_asset_key.cloned(),
                material_bind_group_id: Material2dBindGroupId::default(),
                automatic_batching: !no_automatic_batching,
            },
        );
    }
    *previous_len = entities.len();
    commands.insert_or_spawn_batch(entities);
}

pub fn update_render_world_render_mesh_asset_keys(
    mut key_updates: ResMut<RenderAssetKeyUpdates<Mesh>>,
    mut render_mesh2d_instances: ResMut<RenderMesh2dInstances>,
    mut key_map: Local<HashMap<AssetId<Mesh>, RenderAssetKey<Mesh>>>,
    render_asset_key_sender: Res<RenderAssetKeySender<Mesh>>,
) {
    let mut map: HashMap<InnerRenderAssetKey, Vec<Entity>> = HashMap::new();
    key_map.extend(key_updates.drain(..));
    for (&entity, mesh_instance) in render_mesh2d_instances.iter_mut() {
        if mesh_instance.mesh_asset_key.is_some() {
            continue;
        }
        mesh_instance.mesh_asset_key = key_map
            .get(mesh_instance.mesh_asset_id.as_ref().unwrap())
            .cloned();
        if let Some(key) = mesh_instance.mesh_asset_key {
            map.entry(key.inner).or_default().push(entity);
        }
    }
    let to_send = map
        .into_iter()
        .map(|(k, v)| (v, RenderAssetKey::<Mesh>::new(k)))
        .collect::<Vec<_>>();
    if !to_send.is_empty() {
        match render_asset_key_sender.try_send(to_send) {
            Ok(_) => {}
            Err(_) => panic!("Failed to send"),
        }
    }
}

#[derive(Resource, Clone)]
pub struct Mesh2dPipeline {
    pub view_layout: BindGroupLayout,
    pub mesh_layout: BindGroupLayout,
    // This dummy white texture is to be used in place of optional textures
    pub dummy_white_gpu_image: GpuImage,
    pub per_object_buffer_batch_size: Option<u32>,
}

impl FromWorld for Mesh2dPipeline {
    fn from_world(world: &mut World) -> Self {
        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<RenderQueue>,
            Res<DefaultImageSampler>,
        )> = SystemState::new(world);
        let (render_device, render_queue, default_sampler) = system_state.get_mut(world);
        let render_device = render_device.into_inner();
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
            entries: &[GpuArrayBuffer::<Mesh2dUniform>::binding_layout(
                0,
                ShaderStages::VERTEX_FRAGMENT,
                render_device,
            )],
            label: Some("mesh2d_layout"),
        });
        // A 1x1x1 'all 1.0' texture to use as a dummy texture to use in place of optional StandardMaterial textures
        let dummy_white_gpu_image = {
            let image = Image::default();
            let texture = render_device.create_texture(&image.texture_descriptor);
            let sampler = match image.sampler_descriptor {
                ImageSampler::Default => (**default_sampler).clone(),
                ImageSampler::Descriptor(descriptor) => render_device.create_sampler(&descriptor),
            };

            let format_size = image.texture_descriptor.format.pixel_size();
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
                    bytes_per_row: Some(image.texture_descriptor.size.width * format_size as u32),
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
                mip_level_count: image.texture_descriptor.mip_level_count,
            }
        };
        Mesh2dPipeline {
            view_layout,
            mesh_layout,
            dummy_white_gpu_image,
            per_object_buffer_batch_size: GpuArrayBuffer::<Mesh2dUniform>::batch_size(
                render_device,
            ),
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
            let gpu_image = gpu_images.get_with_asset_id(handle)?;
            Some((&gpu_image.texture_view, &gpu_image.sampler))
        } else {
            Some((
                &self.dummy_white_gpu_image.texture_view,
                &self.dummy_white_gpu_image.sampler,
            ))
        }
    }
}

impl GetBatchData for Mesh2dPipeline {
    type Param = SRes<RenderMesh2dInstances>;
    type Query = Entity;
    type QueryFilter = With<Mesh2d>;
    type CompareData = (Material2dBindGroupId, RenderAssetKey<Mesh>);
    type BufferData = Mesh2dUniform;

    fn get_batch_data(
        mesh_instances: &SystemParamItem<Self::Param>,
        entity: &QueryItem<Self::Query>,
    ) -> (Self::BufferData, Option<Self::CompareData>) {
        let mesh_instance = mesh_instances
            .get(entity)
            .expect("Failed to find render mesh2d instance");
        (
            (&mesh_instance.transforms).into(),
            mesh_instance.automatic_batching.then_some((
                mesh_instance.material_bind_group_id,
                mesh_instance
                    .mesh_asset_key
                    .expect("mesh_asset_key was None"),
            )),
        )
    }
}

/// Batch the items in a render phase. This means comparing metadata needed to draw each phase item
/// and trying to combine the draws into a batch.
pub fn batch_and_prepare_batch_queue(
    gpu_array_buffer: ResMut<GpuArrayBuffer<Mesh2dUniform>>,
    mut views: Query<(&BatchQueue, &mut DrawStream)>,
    render_mesh2d_instances: Res<RenderMesh2dInstances>,
    dynamic_offsets: ResMut<DynamicOffsets>,
) {
    let gpu_array_buffer = gpu_array_buffer.into_inner();
    let dynamic_offsets = dynamic_offsets.into_inner();

    let mut substream = Vec::new();

    for (batch_queue, mut draw_stream) in &mut views {
        draw_stream.clear();

        if batch_queue.is_empty() {
            continue;
        }
        // dbg!(batch_queue.len());

        let (mut prev_batch, mut prev_dynamic_offset_range, mut prev_instance_range): (
            Option<&BatchStruct>,
            Option<Range<u16>>,
            Range<u32>,
        ) = (
            None,
            None,
            gpu_array_buffer.len() as u32..gpu_array_buffer.len() as u32,
        );

        let mut draw_ops = DrawOperations::empty();
        for i in 1..batch_queue.len() {
            let batch = &batch_queue[i];
            let mesh_instance = render_mesh2d_instances.get(&batch.entity).unwrap();
            let buffer_index =
                gpu_array_buffer.push(Mesh2dUniform::from(&mesh_instance.transforms));

            let index = buffer_index.index.get();
            let dynamic_offset_range = if let Some(dynamic_offset) = buffer_index.dynamic_offset {
                let dynamic_offset_index = dynamic_offsets.len() as u16;
                dynamic_offsets.push(dynamic_offset.get());
                dynamic_offset_index..dynamic_offset_index + 1
            } else {
                0..0
            };
            let instance_range = index..index + 1;

            if let Some(prev_batch) = prev_batch {
                if prev_batch.pipeline_id != batch.pipeline_id {
                    draw_ops |= DrawOperations::PIPELINE_ID;
                    substream.push(batch.pipeline_id.id() as u32);
                }
                if prev_batch.material_bind_group_id != batch.material_bind_group_id {
                    draw_ops |= DrawOperations::MATERIAL_BIND_GROUP_ID;
                    substream.push(batch.material_bind_group_id.0.get());
                }
                if prev_batch.material_bind_group_dynamic_offsets
                    != batch.material_bind_group_dynamic_offsets
                {
                    draw_ops |= DrawOperations::MATERIAL_BIND_GROUP_DYNAMIC_OFFSETS;
                    substream.push(
                        ((batch.material_bind_group_dynamic_offsets.start as u32) << 16)
                            | batch.material_bind_group_dynamic_offsets.end as u32,
                    );
                }
                if prev_batch.mesh_buffers != batch.mesh_buffers {
                    draw_ops |= DrawOperations::MESH_BUFFERS_ID;
                    let generational_index = batch.mesh_buffers.inner.data().as_ffi();
                    substream.push((generational_index >> 32) as u32);
                    substream.push((generational_index & ((1 << 32) - 1)) as u32);
                }
                if let Some(prev_dynamic_offset_range) = prev_dynamic_offset_range.as_ref() {
                    if *prev_dynamic_offset_range != dynamic_offset_range
                        && !dynamic_offset_range.is_empty()
                    {
                        draw_ops |= DrawOperations::MESH_BIND_GROUP_DYNAMIC_OFFSETS;
                        substream.push(
                            ((dynamic_offset_range.start as u32) << 16)
                                | dynamic_offset_range.end as u32,
                        );
                    }
                }
            } else {
                // Encode the initial state
                draw_ops |= DrawOperations::PIPELINE_ID;
                substream.push(batch.pipeline_id.id() as u32);
                draw_ops |= DrawOperations::MATERIAL_BIND_GROUP_ID;
                substream.push(batch.material_bind_group_id.0.get());
                draw_ops |= DrawOperations::MATERIAL_BIND_GROUP_DYNAMIC_OFFSETS;
                substream.push(
                    ((batch.material_bind_group_dynamic_offsets.start as u32) << 16)
                        | batch.material_bind_group_dynamic_offsets.end as u32,
                );
                draw_ops |= DrawOperations::MESH_BUFFERS_ID;
                let generational_index = batch.mesh_buffers.inner.data().as_ffi();
                substream.push((generational_index >> 32) as u32);
                substream.push((generational_index & ((1 << 32) - 1)) as u32);
                if !dynamic_offset_range.is_empty() {
                    draw_ops |= DrawOperations::MESH_BIND_GROUP_DYNAMIC_OFFSETS;
                    substream.push(
                        ((dynamic_offset_range.start as u32) << 16)
                            | dynamic_offset_range.end as u32,
                    );
                }

                prev_batch = Some(batch);
                prev_dynamic_offset_range = Some(dynamic_offset_range.clone());
                prev_instance_range = instance_range.clone();
            }

            if draw_ops.is_empty() {
                prev_instance_range.end = instance_range.end;
            } else {
                draw_ops |= DrawOperations::INSTANCE_RANGE;
                substream.push(prev_instance_range.start);
                substream.push(prev_instance_range.end);

                draw_stream.push(prev_batch.unwrap().entity.generation());
                draw_stream.push(prev_batch.unwrap().entity.index());
                draw_stream.push(draw_ops.bits());
                draw_stream.extend(substream.drain(..));
                draw_ops = DrawOperations::empty();

                prev_batch = Some(batch);
                prev_dynamic_offset_range = Some(dynamic_offset_range);
                prev_instance_range = instance_range;
            }
        }
        if !draw_ops.is_empty() {
            draw_ops |= DrawOperations::INSTANCE_RANGE;
            substream.push(prev_instance_range.start);
            substream.push(prev_instance_range.end);

            draw_stream.push(draw_ops.bits());
            draw_stream.extend(substream.drain(..));
        }
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    // NOTE: Apparently quadro drivers support up to 64x MSAA.
    // MSAA uses the highest 3 bits for the MSAA log2(sample count) to support up to 128x MSAA.
    // FIXME: make normals optional?
    pub struct Mesh2dPipelineKey: u32 {
        const NONE                              = 0;
        const HDR                               = (1 << 0);
        const TONEMAP_IN_SHADER                 = (1 << 1);
        const DEBAND_DITHER                     = (1 << 2);
        const MSAA_RESERVED_BITS                = Self::MSAA_MASK_BITS << Self::MSAA_SHIFT_BITS;
        const PRIMITIVE_TOPOLOGY_RESERVED_BITS  = Self::PRIMITIVE_TOPOLOGY_MASK_BITS << Self::PRIMITIVE_TOPOLOGY_SHIFT_BITS;
        const TONEMAP_METHOD_RESERVED_BITS      = Self::TONEMAP_METHOD_MASK_BITS << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_NONE               = 0 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_REINHARD           = 1 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_REINHARD_LUMINANCE = 2 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_ACES_FITTED        = 3 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_AGX                = 4 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM = 5 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_TONY_MC_MAPFACE    = 6 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_BLENDER_FILMIC     = 7 << Self::TONEMAP_METHOD_SHIFT_BITS;
    }
}

impl Mesh2dPipelineKey {
    const MSAA_MASK_BITS: u32 = 0b111;
    const MSAA_SHIFT_BITS: u32 = 32 - Self::MSAA_MASK_BITS.count_ones();
    const PRIMITIVE_TOPOLOGY_MASK_BITS: u32 = 0b111;
    const PRIMITIVE_TOPOLOGY_SHIFT_BITS: u32 = Self::MSAA_SHIFT_BITS - 3;
    const TONEMAP_METHOD_MASK_BITS: u32 = 0b111;
    const TONEMAP_METHOD_SHIFT_BITS: u32 =
        Self::PRIMITIVE_TOPOLOGY_SHIFT_BITS - Self::TONEMAP_METHOD_MASK_BITS.count_ones();

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits =
            (msaa_samples.trailing_zeros() & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        Self::from_bits_retain(msaa_bits)
    }

    pub fn from_hdr(hdr: bool) -> Self {
        if hdr {
            Mesh2dPipelineKey::HDR
        } else {
            Mesh2dPipelineKey::NONE
        }
    }

    pub fn msaa_samples(&self) -> u32 {
        1 << ((self.bits() >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS)
    }

    pub fn from_primitive_topology(primitive_topology: PrimitiveTopology) -> Self {
        let primitive_topology_bits = ((primitive_topology as u32)
            & Self::PRIMITIVE_TOPOLOGY_MASK_BITS)
            << Self::PRIMITIVE_TOPOLOGY_SHIFT_BITS;
        Self::from_bits_retain(primitive_topology_bits)
    }

    pub fn primitive_topology(&self) -> PrimitiveTopology {
        let primitive_topology_bits = (self.bits() >> Self::PRIMITIVE_TOPOLOGY_SHIFT_BITS)
            & Self::PRIMITIVE_TOPOLOGY_MASK_BITS;
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

            let method = key.intersection(Mesh2dPipelineKey::TONEMAP_METHOD_RESERVED_BITS);

            match method {
                Mesh2dPipelineKey::TONEMAP_METHOD_NONE => {
                    shader_defs.push("TONEMAP_METHOD_NONE".into());
                }
                Mesh2dPipelineKey::TONEMAP_METHOD_REINHARD => {
                    shader_defs.push("TONEMAP_METHOD_REINHARD".into());
                }
                Mesh2dPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE => {
                    shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
                }
                Mesh2dPipelineKey::TONEMAP_METHOD_ACES_FITTED => {
                    shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into());
                }
                Mesh2dPipelineKey::TONEMAP_METHOD_AGX => {
                    shader_defs.push("TONEMAP_METHOD_AGX".into());
                }
                Mesh2dPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM => {
                    shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
                }
                Mesh2dPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC => {
                    shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
                }
                Mesh2dPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE => {
                    shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
                }
                _ => {}
            }
            // Debanding is tied to tonemapping in the shader, cannot run without it.
            if key.contains(Mesh2dPipelineKey::DEBAND_DITHER) {
                shader_defs.push("DEBAND_DITHER".into());
            }
        }

        let vertex_buffer_layout = layout.get_layout(&vertex_attributes)?;

        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        Ok(RenderPipelineDescriptor {
            vertex: VertexState {
                shader: MESH2D_SHADER_HANDLE,
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: MESH2D_SHADER_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout: vec![self.view_layout.clone(), self.mesh_layout.clone()],
            push_constant_ranges: Vec::new(),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
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

pub fn prepare_mesh2d_bind_group(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    render_device: Res<RenderDevice>,
    mesh2d_uniforms: Res<GpuArrayBuffer<Mesh2dUniform>>,
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

pub fn prepare_mesh2d_view_bind_groups(
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
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetMesh2dBindGroup<I> {
    type Param = SRes<Mesh2dBindGroup>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: (),
        mesh2d_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mut dynamic_offsets: [u32; 1] = Default::default();
        let mut offset_count = 0;
        if let Some(dynamic_offset) = item.dynamic_offset() {
            dynamic_offsets[offset_count] = dynamic_offset.get();
            offset_count += 1;
        }
        pass.set_bind_group(
            I,
            &mesh2d_bind_group.into_inner().value,
            &dynamic_offsets[..offset_count],
        );
        RenderCommandResult::Success
    }
}

pub struct DrawMesh2d;
impl<P: PhaseItem> RenderCommand<P> for DrawMesh2d {
    type Param = (SRes<RenderAssets<Mesh>>, SRes<RenderMesh2dInstances>);
    type ViewWorldQuery = ();
    type ItemWorldQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: (),
        (meshes, render_mesh2d_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let meshes = meshes.into_inner();
        let render_mesh2d_instances = render_mesh2d_instances.into_inner();

        let Some(RenderMesh2dInstance {
            mesh_asset_key: Some(mesh_asset_key),
            ..
        }) = render_mesh2d_instances.get(&item.entity())
        else {
            return RenderCommandResult::Failure;
        };
        let Some(gpu_mesh) = meshes.get_with_key(*mesh_asset_key) else {
            return RenderCommandResult::Failure;
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));

        let batch_range = item.batch_range();
        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        pass.set_push_constants(
            ShaderStages::VERTEX,
            0,
            &(batch_range.start as i32).to_le_bytes(),
        );
        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, batch_range.clone());
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..gpu_mesh.vertex_count, batch_range.clone());
            }
        }
        RenderCommandResult::Success
    }
}

// FIXME run this
pub fn render_from_draw_streams(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    clear_color: Res<ClearColor>,
    pipeline_cache: Res<PipelineCache>,
    mesh_bind_group: Res<Mesh2dBindGroup>,
    materials: Res<RenderMaterials2d<ColorMaterial>>,
    material_instances: Res<RenderMaterial2dInstances<ColorMaterial>>,
    meshes: Res<RenderAssets<Mesh>>,
    dynamic_offsets: Res<DynamicOffsets>,
    views: Query<(
        &ExtractedCamera,
        &ViewTarget,
        &Camera2d,
        &DrawStream,
        &Mesh2dViewBindGroup,
        &ViewUniformOffset,
    )>,
) {
    let render_device = render_device.into_inner();
    let pipeline_cache = pipeline_cache.into_inner();
    let materials = materials.into_inner();
    let material_instances = material_instances.into_inner();
    let dynamic_offsets = dynamic_offsets.into_inner();
    let meshes = meshes.into_inner();
    let mesh_bind_group = mesh_bind_group.into_inner();

    let mut render_context = RenderContext::new(render_device.clone());

    for (camera, target, camera_2d, draw_stream, view_bind_group, view_bind_group_dynamic_offset) in
        views.iter()
    {
        #[cfg(feature = "trace")]
        let _main_pass_2d = info_span!("main_pass_2d").entered();

        if draw_stream.is_empty() {
            continue;
        }

        // Cannot use command_encoder() as we need to split the borrow on self
        let command_encoder = render_context.command_encoder();
        let descriptor = RenderPassDescriptor {
            label: Some("main_pass_2d"),
            color_attachments: &[Some(target.get_color_attachment(Operations {
                load: match camera_2d.clear_color {
                    ClearColorConfig::Default => LoadOp::Clear(clear_color.0.into()),
                    ClearColorConfig::Custom(color) => LoadOp::Clear(color.into()),
                    ClearColorConfig::None => LoadOp::Load,
                },
                store: true,
            }))],
            depth_stencil_attachment: None,
        };
        let mut render_pass = command_encoder.begin_render_pass(&descriptor);

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_viewport(
                viewport.physical_position.x as f32,
                viewport.physical_position.y as f32,
                viewport.physical_size.x as f32,
                viewport.physical_size.y as f32,
                viewport.depth.start,
                viewport.depth.end,
            );
        }

        // Bind the view bind group
        render_pass.set_bind_group(
            0,
            &view_bind_group.value,
            &[view_bind_group_dynamic_offset.offset],
        );

        let has_storage_buffers = render_device.limits().max_storage_buffers_per_shader_stage > 0;
        if has_storage_buffers {
            render_pass.set_bind_group(2, &mesh_bind_group.value, &[]);
        }

        let mut material_bind_group_id = None;
        let mut material_bind_group_dynamic_offsets = 0..0;
        let mut material_bind_group_rebind = false;

        let mut indexed = false;
        let mut vertex_count = 0;

        let mut i = 0;
        // dbg!(draw_stream.len());
        while i < draw_stream.len() {
            let entity =
                Entity::from_bits((draw_stream[i] as u64) << 32 | draw_stream[i + 1] as u64);
            i += 2;
            let draw_ops = DrawOperations::from_bits(draw_stream[i]).unwrap();
            i += 1;

            if draw_ops.contains(DrawOperations::PIPELINE_ID) {
                let pipeline_id = draw_stream[i];
                i += 1;

                if let Some(pipeline) =
                    pipeline_cache.get_render_pipeline(CachedRenderPipelineId(pipeline_id as usize))
                {
                    render_pass.set_pipeline(pipeline);
                }
            }

            if draw_ops.contains(DrawOperations::MATERIAL_BIND_GROUP_ID) {
                material_bind_group_id = Some(draw_stream[i]);
                i += 1;
                material_bind_group_rebind = true;
            }

            if draw_ops.contains(DrawOperations::MATERIAL_BIND_GROUP_DYNAMIC_OFFSETS) {
                let dynamic_offsets_range_encoded = draw_stream[i];
                i += 1;
                material_bind_group_dynamic_offsets = ((dynamic_offsets_range_encoded >> 16)
                    as usize)
                    ..((dynamic_offsets_range_encoded & ((1 << 16) - 1)) as usize);
                material_bind_group_rebind = true;
            }

            'material_bind_group: {
                if material_bind_group_rebind {
                    let Some(material_instance) = material_instances.get(&entity) else {
                        break 'material_bind_group;
                    };
                    let Some(material_key) = material_instance.key else {
                        break 'material_bind_group;
                    };
                    let Some(material2d) = materials.get_with_key(material_key) else {
                        break 'material_bind_group;
                    };
                    render_pass.set_bind_group(
                        1,
                        &material2d.bind_group,
                        if material_bind_group_dynamic_offsets.is_empty() {
                            &[]
                        } else {
                            &dynamic_offsets[material_bind_group_dynamic_offsets.clone()]
                        },
                    );
                }
            }

            if draw_ops.contains(DrawOperations::MESH_BUFFERS_ID) {
                let mesh_asset_key = RenderAssetKey::<Mesh>::new(InnerRenderAssetKey::from(
                    KeyData::from_ffi(((draw_stream[i] as u64) << 32) | draw_stream[i + 1] as u64),
                ));
                i += 2;

                let Some(gpu_mesh) = meshes.get_with_key(mesh_asset_key) else {
                    panic!("BLARGH");
                };

                render_pass.set_vertex_buffer(0, *gpu_mesh.vertex_buffer.slice(..));

                match &gpu_mesh.buffer_info {
                    GpuBufferInfo::Indexed {
                        buffer,
                        index_format,
                        count,
                    } => {
                        render_pass.set_index_buffer(*buffer.slice(..), *index_format);
                        indexed = true;
                        vertex_count = *count;
                    }
                    GpuBufferInfo::NonIndexed => {
                        indexed = false;
                        vertex_count = gpu_mesh.vertex_count;
                    }
                }
            }

            if draw_ops.contains(DrawOperations::MESH_BIND_GROUP_DYNAMIC_OFFSETS) {
                let dynamic_offset_range_encoded = draw_stream[i];
                i += 1;
                let dynamic_offset_range = (((dynamic_offset_range_encoded >> 16) & ((1 << 16) - 1))
                    as usize)
                    ..((dynamic_offset_range_encoded & ((1 << 16) - 1)) as usize);
                if !has_storage_buffers {
                    render_pass.set_bind_group(
                        2,
                        &mesh_bind_group.value,
                        &dynamic_offsets[dynamic_offset_range],
                    );
                }
            }

            let instance_range = if draw_ops.contains(DrawOperations::INSTANCE_RANGE) {
                let range = draw_stream[i]..draw_stream[i + 1];
                i += 2;
                range
            } else {
                0..1
            };

            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            render_pass.set_push_constants(
                ShaderStages::VERTEX,
                0,
                &(instance_range.start as i32).to_le_bytes(),
            );

            if indexed {
                render_pass.draw_indexed(0..vertex_count, 0, instance_range.clone());
            } else {
                render_pass.draw(0..vertex_count, instance_range.clone());
            }
        }
        assert_eq!(i, draw_stream.len());
    }

    // WebGL2 quirk: if ending with a render pass with a custom viewport, the viewport isn't
    // reset for the next render pass so add an empty render pass without a custom viewport
    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    if camera.viewport.is_some() {
        #[cfg(feature = "trace")]
        let _reset_viewport_pass_2d = info_span!("reset_viewport_pass_2d").entered();
        let pass_descriptor = RenderPassDescriptor {
            label: Some("reset_viewport_pass_2d"),
            color_attachments: &[Some(target.get_color_attachment(Operations {
                load: LoadOp::Load,
                store: true,
            }))],
            depth_stencil_attachment: None,
        };

        render_context
            .command_encoder()
            .begin_render_pass(&pass_descriptor);
    }

    {
        #[cfg(feature = "trace")]
        let _span = info_span!("submit_graph_commands").entered();
        render_queue.0.submit(render_context.finish());
    }
}
