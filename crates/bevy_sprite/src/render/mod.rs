use std::ops::Range;

use crate::{
    texture_atlas::{TextureAtlas, TextureAtlasSprite},
    Sprite, SPRITE_SHADER_HANDLE,
};
use bevy_asset::{AssetEvent, Assets, Handle, HandleId};
use bevy_core_pipeline::{
    core_2d::Transparent2d,
    tonemapping::{DebandDither, Tonemapping},
};
use bevy_ecs::{
    prelude::*,
    storage::SparseSet,
    system::{lifetimeless::*, SystemParamItem, SystemState},
};
use bevy_math::{Affine3A, Quat, Rect, Vec2, Vec4};
use bevy_render::{
    color::Color,
    render_asset::RenderAssets,
    render_phase::{
        DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult, RenderPhase, SetItemPipeline,
        TrackedRenderPass,
    },
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    texture::{
        BevyDefault, DefaultImageSampler, GpuImage, Image, ImageSampler, TextureFormatPixelInfo,
    },
    view::{
        ComputedVisibility, ExtractedView, Msaa, ViewTarget, ViewUniform, ViewUniformOffset,
        ViewUniforms, VisibleEntities,
    },
    Extract,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{FloatOrd, HashMap, Uuid};
use bytemuck::{Pod, Zeroable};
use fixedbitset::FixedBitSet;

#[derive(Resource)]
pub struct SpritePipeline {
    view_layout: BindGroupLayout,
    material_layout: BindGroupLayout,
    sprite_layout: BindGroupLayout,
    pub dummy_white_gpu_image: GpuImage,
}

impl FromWorld for SpritePipeline {
    fn from_world(world: &mut World) -> Self {
        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<DefaultImageSampler>,
            Res<RenderQueue>,
        )> = SystemState::new(world);
        let (render_device, default_sampler, render_queue) = system_state.get_mut(world);

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            }],
            label: Some("sprite_view_layout"),
        });

        let material_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("sprite_material_layout"),
        });

        let sprite_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[GpuArrayBuffer::<GpuQuad>::binding_layout(
                0,
                ShaderStages::VERTEX_FRAGMENT,
                &render_device,
            )],
            label: Some("sprite_layout"),
        });

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

        SpritePipeline {
            view_layout,
            material_layout,
            sprite_layout,
            dummy_white_gpu_image,
        }
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    // NOTE: Apparently quadro drivers support up to 64x MSAA.
    // MSAA uses the highest 3 bits for the MSAA log2(sample count) to support up to 128x MSAA.
    pub struct SpritePipelineKey: u32 {
        const NONE                              = 0;
        const COLORED                           = (1 << 0);
        const HDR                               = (1 << 1);
        const TONEMAP_IN_SHADER                 = (1 << 2);
        const DEBAND_DITHER                     = (1 << 3);
        const MSAA_RESERVED_BITS                = Self::MSAA_MASK_BITS << Self::MSAA_SHIFT_BITS;
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

impl SpritePipelineKey {
    const MSAA_MASK_BITS: u32 = 0b111;
    const MSAA_SHIFT_BITS: u32 = 32 - Self::MSAA_MASK_BITS.count_ones();
    const TONEMAP_METHOD_MASK_BITS: u32 = 0b111;
    const TONEMAP_METHOD_SHIFT_BITS: u32 =
        Self::MSAA_SHIFT_BITS - Self::TONEMAP_METHOD_MASK_BITS.count_ones();

    #[inline]
    pub const fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits =
            (msaa_samples.trailing_zeros() & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        Self::from_bits_retain(msaa_bits)
    }

    #[inline]
    pub const fn msaa_samples(&self) -> u32 {
        1 << ((self.bits() >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS)
    }

    #[inline]
    pub const fn from_colored(colored: bool) -> Self {
        if colored {
            SpritePipelineKey::COLORED
        } else {
            SpritePipelineKey::NONE
        }
    }

    #[inline]
    pub const fn from_hdr(hdr: bool) -> Self {
        if hdr {
            SpritePipelineKey::HDR
        } else {
            SpritePipelineKey::NONE
        }
    }
}

impl SpecializedRenderPipeline for SpritePipeline {
    type Key = SpritePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();
        if key.contains(SpritePipelineKey::TONEMAP_IN_SHADER) {
            shader_defs.push("TONEMAP_IN_SHADER".into());

            let method = key.intersection(SpritePipelineKey::TONEMAP_METHOD_RESERVED_BITS);

            if method == SpritePipelineKey::TONEMAP_METHOD_NONE {
                shader_defs.push("TONEMAP_METHOD_NONE".into());
            } else if method == SpritePipelineKey::TONEMAP_METHOD_REINHARD {
                shader_defs.push("TONEMAP_METHOD_REINHARD".into());
            } else if method == SpritePipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            } else if method == SpritePipelineKey::TONEMAP_METHOD_ACES_FITTED {
                shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into());
            } else if method == SpritePipelineKey::TONEMAP_METHOD_AGX {
                shader_defs.push("TONEMAP_METHOD_AGX".into());
            } else if method == SpritePipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
            {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            } else if method == SpritePipelineKey::TONEMAP_METHOD_BLENDER_FILMIC {
                shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
            } else if method == SpritePipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE {
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }

            // Debanding is tied to tonemapping in the shader, cannot run without it.
            if key.contains(SpritePipelineKey::DEBAND_DITHER) {
                shader_defs.push("DEBAND_DITHER".into());
            }
        }

        let format = match key.contains(SpritePipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: SPRITE_SHADER_HANDLE.typed::<Shader>(),
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: SPRITE_SHADER_HANDLE.typed::<Shader>(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout: vec![
                self.view_layout.clone(),
                self.material_layout.clone(),
                self.sprite_layout.clone(),
            ],
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("sprite_pipeline".into()),
            push_constant_ranges: Vec::new(),
        }
    }
}

#[derive(Component, Clone, Copy)]
pub struct ExtractedSprite {
    pub transform: GlobalTransform,
    pub color: Color,
    /// Select an area of the texture
    pub rect: Option<Rect>,
    /// Change the on-screen size of the sprite
    pub custom_size: Option<Vec2>,
    /// Handle to the [`Image`] of this sprite
    /// PERF: storing a `HandleId` instead of `Handle<Image>` enables some optimizations (`ExtractedSprite` becomes `Copy` and doesn't need to be dropped)
    pub image_handle_id: HandleId,
    pub flip_x: bool,
    pub flip_y: bool,
    pub anchor: Vec2,
}

#[derive(Resource, Default)]
pub struct ExtractedSprites {
    pub sprites: SparseSet<Entity, ExtractedSprite>,
}

#[derive(Resource, Default)]
pub struct SpriteAssetEvents {
    pub images: Vec<AssetEvent<Image>>,
}

pub fn extract_sprite_events(
    mut events: ResMut<SpriteAssetEvents>,
    mut image_events: Extract<EventReader<AssetEvent<Image>>>,
) {
    let SpriteAssetEvents { ref mut images } = *events;
    images.clear();

    for image in image_events.iter() {
        // AssetEvent: !Clone
        images.push(match image {
            AssetEvent::Created { handle } => AssetEvent::Created {
                handle: handle.clone_weak(),
            },
            AssetEvent::Modified { handle } => AssetEvent::Modified {
                handle: handle.clone_weak(),
            },
            AssetEvent::Removed { handle } => AssetEvent::Removed {
                handle: handle.clone_weak(),
            },
        });
    }
}

pub fn extract_sprites(
    mut extracted_sprites: ResMut<ExtractedSprites>,
    texture_atlases: Extract<Res<Assets<TextureAtlas>>>,
    sprite_query: Extract<
        Query<(
            Entity,
            &ComputedVisibility,
            &Sprite,
            &GlobalTransform,
            &Handle<Image>,
        )>,
    >,
    atlas_query: Extract<
        Query<(
            Entity,
            &ComputedVisibility,
            &TextureAtlasSprite,
            &GlobalTransform,
            &Handle<TextureAtlas>,
        )>,
    >,
) {
    extracted_sprites.sprites.clear();
    for (entity, visibility, sprite, transform, handle) in sprite_query.iter() {
        if !visibility.is_visible() {
            continue;
        }
        // PERF: we don't check in this function that the `Image` asset is ready, since it should be in most cases and hashing the handle is expensive
        extracted_sprites.sprites.insert(
            entity,
            ExtractedSprite {
                color: sprite.color,
                transform: *transform,
                rect: sprite.rect,
                // Pass the custom size
                custom_size: sprite.custom_size,
                flip_x: sprite.flip_x,
                flip_y: sprite.flip_y,
                image_handle_id: handle.id(),
                anchor: sprite.anchor.as_vec(),
            },
        );
    }
    for (entity, visibility, atlas_sprite, transform, texture_atlas_handle) in atlas_query.iter() {
        if !visibility.is_visible() {
            continue;
        }
        if let Some(texture_atlas) = texture_atlases.get(texture_atlas_handle) {
            let rect = Some(
                *texture_atlas
                    .textures
                    .get(atlas_sprite.index)
                    .unwrap_or_else(|| {
                        panic!(
                            "Sprite index {:?} does not exist for texture atlas handle {:?}.",
                            atlas_sprite.index,
                            texture_atlas_handle.id(),
                        )
                    }),
            );
            extracted_sprites.sprites.insert(
                entity,
                ExtractedSprite {
                    color: atlas_sprite.color,
                    transform: *transform,
                    // Select the area in the texture atlas
                    rect,
                    // Pass the custom size
                    custom_size: atlas_sprite.custom_size,
                    flip_x: atlas_sprite.flip_x,
                    flip_y: atlas_sprite.flip_y,
                    image_handle_id: texture_atlas.texture.id(),
                    anchor: atlas_sprite.anchor.as_vec(),
                },
            );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SpriteVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ColoredSpriteVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Resource)]
pub struct SpriteMeta {
    view_bind_group: Option<BindGroup>,
    sprite_bind_group: Option<BindGroup>,
    sprite_index_buffer: BufferVec<u32>,
}

impl Default for SpriteMeta {
    fn default() -> Self {
        Self {
            view_bind_group: None,
            sprite_bind_group: None,
            sprite_index_buffer: BufferVec::<u32>::new(BufferUsages::INDEX),
        }
    }
}

#[derive(Component, Eq, PartialEq, Clone)]
pub struct SpriteBatch {
    image_handle_id: HandleId,
    range: Range<u32>,
}

#[derive(Resource, Default)]
pub struct ImageBindGroups {
    values: HashMap<Handle<Image>, BindGroup>,
}

#[allow(clippy::too_many_arguments)]
pub fn queue_sprites(
    mut view_entities: Local<FixedBitSet>,
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    sprite_pipeline: Res<SpritePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SpritePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    msaa: Res<Msaa>,
    extracted_sprites: Res<ExtractedSprites>,
    mut views: Query<(
        &mut RenderPhase<Transparent2d>,
        &VisibleEntities,
        &ExtractedView,
        Option<&Tonemapping>,
        Option<&DebandDither>,
    )>,
) {
    let msaa_key = SpritePipelineKey::from_msaa_samples(msaa.samples());

    let draw_sprite_function = draw_functions.read().id::<DrawSprite>();

    for (mut transparent_phase, visible_entities, view, tonemapping, dither) in &mut views {
        let mut view_key = SpritePipelineKey::from_hdr(view.hdr) | msaa_key;

        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= SpritePipelineKey::TONEMAP_IN_SHADER;
                view_key |= match tonemapping {
                    Tonemapping::None => SpritePipelineKey::TONEMAP_METHOD_NONE,
                    Tonemapping::Reinhard => SpritePipelineKey::TONEMAP_METHOD_REINHARD,
                    Tonemapping::ReinhardLuminance => {
                        SpritePipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE
                    }
                    Tonemapping::AcesFitted => SpritePipelineKey::TONEMAP_METHOD_ACES_FITTED,
                    Tonemapping::AgX => SpritePipelineKey::TONEMAP_METHOD_AGX,
                    Tonemapping::SomewhatBoringDisplayTransform => {
                        SpritePipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
                    }
                    Tonemapping::TonyMcMapface => SpritePipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
                    Tonemapping::BlenderFilmic => SpritePipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
                };
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= SpritePipelineKey::DEBAND_DITHER;
            }
        }

        let pipeline = pipelines.specialize(
            &pipeline_cache,
            &sprite_pipeline,
            view_key | SpritePipelineKey::from_colored(false),
        );
        let colored_pipeline = pipelines.specialize(
            &pipeline_cache,
            &sprite_pipeline,
            view_key | SpritePipelineKey::from_colored(true),
        );

        view_entities.clear();
        view_entities.extend(visible_entities.entities.iter().map(|e| e.index() as usize));

        transparent_phase
            .items
            .reserve(extracted_sprites.sprites.len());

        for (entity, extracted_sprite) in extracted_sprites.sprites.iter() {
            if !view_entities.contains(entity.index() as usize) {
                continue;
            }

            // These items will be sorted by depth with other phase items
            let sort_key = FloatOrd(extracted_sprite.transform.translation().z);

            // Add the item to the render phase
            if extracted_sprite.color != Color::WHITE {
                transparent_phase.add(Transparent2d {
                    draw_function: draw_sprite_function,
                    pipeline: colored_pipeline,
                    entity: *entity,
                    sort_key,
                    // batch size will be calculated in prepare_sprites
                    batch_size: 0,
                });
            } else {
                transparent_phase.add(Transparent2d {
                    draw_function: draw_sprite_function,
                    pipeline,
                    entity: *entity,
                    sort_key,
                    // batch size will be calculated in prepare_sprites
                    batch_size: 0,
                });
            }
        }
    }
}

#[derive(Clone, ShaderType)]
pub struct GpuQuad {
    // Affine 4x3
    pub col0_tx: Vec4,
    pub col1_ty: Vec4,
    pub col2_tz: Vec4,
    pub color: [f32; 4],
    pub uv_offset_scale: Vec4,
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_sprites(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut sprite_meta: ResMut<SpriteMeta>,
    mut sprites: ResMut<GpuArrayBuffer<GpuQuad>>,
    view_uniforms: Res<ViewUniforms>,
    sprite_pipeline: Res<SpritePipeline>,
    mut image_bind_groups: ResMut<ImageBindGroups>,
    gpu_images: Res<RenderAssets<Image>>,
    extracted_sprites: Res<ExtractedSprites>,
    mut phases: Query<&mut RenderPhase<Transparent2d>>,
    events: Res<SpriteAssetEvents>,
) {
    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Created { .. } => None,
            AssetEvent::Modified { handle } | AssetEvent::Removed { handle } => {
                image_bind_groups.values.remove(handle)
            }
        };
    }

    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let mut batches: Vec<(Entity, (SpriteBatch, GpuArrayBufferIndex<GpuQuad>))> =
            Vec::with_capacity(*previous_len);

        // Clear the GpuArrayBuffer
        sprites.clear();

        sprite_meta.view_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: view_binding,
            }],
            label: Some("sprite_view_bind_group"),
            layout: &sprite_pipeline.view_layout,
        }));

        // Index buffer indices
        let mut index = 0;

        let image_bind_groups = &mut *image_bind_groups;

        let mut total_sprites = 0;
        for mut transparent_phase in &mut phases {
            total_sprites += transparent_phase.items.len();
            let mut batch_item_index = 0;
            let mut batch_image_size = Vec2::ZERO;
            let mut batch_image_handle = HandleId::Id(Uuid::nil(), u64::MAX);
            let mut batch_dynamic_offset = None;

            // Iterate through the phase items and detect when successive sprites that can be batched.
            // Spawn an entity with a `SpriteBatch` component for each possible batch.
            // Compatible items share the same entity.
            for item_index in 0..transparent_phase.items.len() {
                let item = &transparent_phase.items[item_index];
                let Some(extracted_sprite) = extracted_sprites.sprites.get(item.entity) else { continue };

                let batch_image_changed = batch_image_handle != extracted_sprite.image_handle_id;
                if batch_image_changed {
                    let Some(gpu_image) =
                    gpu_images.get(&Handle::weak(extracted_sprite.image_handle_id)) else { continue };

                    batch_image_size = Vec2::new(gpu_image.size.x, gpu_image.size.y);
                    batch_image_handle = extracted_sprite.image_handle_id;
                    image_bind_groups
                        .values
                        .entry(Handle::weak(batch_image_handle))
                        .or_insert_with(|| {
                            render_device.create_bind_group(&BindGroupDescriptor {
                                entries: &[
                                    BindGroupEntry {
                                        binding: 0,
                                        resource: BindingResource::TextureView(
                                            &gpu_image.texture_view,
                                        ),
                                    },
                                    BindGroupEntry {
                                        binding: 1,
                                        resource: BindingResource::Sampler(&gpu_image.sampler),
                                    },
                                ],
                                label: Some("sprite_material_bind_group"),
                                layout: &sprite_pipeline.material_layout,
                            })
                        });
                }

                // By default, the size of the quad is the size of the texture
                let mut quad_size = batch_image_size;

                // Calculate vertex data for this item
                let mut uv_offset_scale: Vec4;

                // If a rect is specified, adjust UVs and the size of the quad
                if let Some(rect) = extracted_sprite.rect {
                    let rect_size = rect.size();
                    uv_offset_scale = Vec4::new(
                        rect.min.x / batch_image_size.x,
                        rect.max.y / batch_image_size.y,
                        rect_size.x / batch_image_size.x,
                        -rect_size.y / batch_image_size.y,
                    );
                    quad_size = rect_size;
                } else {
                    uv_offset_scale = Vec4::new(0.0, 1.0, 1.0, -1.0);
                }

                if extracted_sprite.flip_x {
                    uv_offset_scale.x += uv_offset_scale.z;
                    uv_offset_scale.z *= -1.0;
                }
                if extracted_sprite.flip_y {
                    uv_offset_scale.y += uv_offset_scale.w;
                    uv_offset_scale.w *= -1.0;
                }

                // Override the size if a custom one is specified
                if let Some(custom_size) = extracted_sprite.custom_size {
                    quad_size = custom_size;
                }

                let transform = extracted_sprite.transform.affine()
                    * Affine3A::from_scale_rotation_translation(
                        quad_size.extend(1.0),
                        Quat::IDENTITY,
                        (quad_size * (-extracted_sprite.anchor - Vec2::splat(0.5))).extend(0.0),
                    );

                // Store the vertex data and add the item to the render phase
                let sprite_index = sprites.push(GpuQuad {
                    col0_tx: transform.matrix3.x_axis.extend(transform.translation.x),
                    col1_ty: transform.matrix3.y_axis.extend(transform.translation.y),
                    col2_tz: transform.matrix3.z_axis.extend(transform.translation.z),
                    color: extracted_sprite.color.as_linear_rgba_f32(),
                    uv_offset_scale,
                });

                if batch_image_changed || batch_dynamic_offset != sprite_index.dynamic_offset {
                    batch_item_index = item_index;
                    batch_dynamic_offset = sprite_index.dynamic_offset;

                    let new_batch = SpriteBatch {
                        image_handle_id: batch_image_handle,
                        range: index..index,
                    };

                    batches.push((item.entity, (new_batch.clone(), sprite_index)));
                }

                transparent_phase.items[batch_item_index].batch_size += 1;
                batches.last_mut().unwrap().1 .0.range.end += 6;
                index += 6;
            }
        }
        sprites.write_buffer(&render_device, &render_queue);
        if let Some(resource) = sprites.binding() {
            sprite_meta.sprite_bind_group =
                Some(render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource,
                    }],
                    label: Some("sprite_material_bind_group"),
                    layout: &sprite_pipeline.sprite_layout,
                }));
        }

        let total_indices = total_sprites * 6;
        let current_indices = sprite_meta.sprite_index_buffer.len();
        if total_indices > current_indices {
            sprite_meta
                .sprite_index_buffer
                .values
                .reserve(total_indices - current_indices);
            for i in (current_indices / 6)..total_sprites {
                let base = (i * 4) as u32;
                sprite_meta.sprite_index_buffer.push(base + 2);
                sprite_meta.sprite_index_buffer.push(base);
                sprite_meta.sprite_index_buffer.push(base + 1);
                sprite_meta.sprite_index_buffer.push(base + 1);
                sprite_meta.sprite_index_buffer.push(base + 3);
                sprite_meta.sprite_index_buffer.push(base + 2);
            }
            sprite_meta
                .sprite_index_buffer
                .write_buffer(&render_device, &render_queue);
        }

        *previous_len = batches.len();
        commands.insert_or_spawn_batch(batches);
    }
}

pub type DrawSprite = (
    SetItemPipeline,
    SetSpriteViewBindGroup<0>,
    SetSpriteTextureBindGroup<1>,
    SetSpriteInstanceBindGroup<2>,
    DrawSpriteBatch,
);

pub struct SetSpriteViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSpriteViewBindGroup<I> {
    type Param = SRes<SpriteMeta>;
    type ViewWorldQuery = Read<ViewUniformOffset>;
    type ItemWorldQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'_ ViewUniformOffset,
        _entity: (),
        sprite_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            sprite_meta.into_inner().view_bind_group.as_ref().unwrap(),
            &[view_uniform.offset],
        );
        RenderCommandResult::Success
    }
}

pub struct SetSpriteTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSpriteTextureBindGroup<I> {
    type Param = SRes<ImageBindGroups>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<SpriteBatch>;

    fn render<'w>(
        _item: &P,
        _view: (),
        sprite_batch: &'_ SpriteBatch,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let image_bind_groups = image_bind_groups.into_inner();

        pass.set_bind_group(
            I,
            image_bind_groups
                .values
                .get(&Handle::weak(sprite_batch.image_handle_id))
                .unwrap(),
            &[],
        );
        RenderCommandResult::Success
    }
}

pub struct SetSpriteInstanceBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSpriteInstanceBindGroup<I> {
    type Param = SRes<SpriteMeta>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<GpuArrayBufferIndex<GpuQuad>>;

    fn render<'w>(
        _item: &P,
        _view: (),
        gpu_array_buffer_index: &'_ GpuArrayBufferIndex<GpuQuad>,
        sprite_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mut dynamic_offsets: [u32; 1] = Default::default();
        let mut dynamic_offset_count = 0;
        if let Some(dynamic_offset) = gpu_array_buffer_index.dynamic_offset {
            dynamic_offsets[dynamic_offset_count] = dynamic_offset;
            dynamic_offset_count += 1;
        }
        pass.set_bind_group(
            I,
            sprite_meta.into_inner().sprite_bind_group.as_ref().unwrap(),
            &dynamic_offsets[0..dynamic_offset_count],
        );
        RenderCommandResult::Success
    }
}

pub struct DrawSpriteBatch;
impl<P: PhaseItem> RenderCommand<P> for DrawSpriteBatch {
    type Param = SRes<SpriteMeta>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<SpriteBatch>;

    fn render<'w>(
        _item: &P,
        _view: (),
        sprite_batch: &'_ SpriteBatch,
        sprite_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let sprite_meta = sprite_meta.into_inner();
        pass.set_index_buffer(
            sprite_meta.sprite_index_buffer.buffer().unwrap().slice(..),
            0,
            IndexFormat::Uint32,
        );
        pass.draw_indexed(sprite_batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}
