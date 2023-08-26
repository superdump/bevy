use std::mem::size_of;

use bytemuck::{cast_slice, Pod};
use wgpu::{
    BindGroupLayoutEntry, BindingResource, BindingType, Extent3d, ImageDataLayout, ShaderStages,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

use crate::renderer::{RenderDevice, RenderQueue};

use super::{GpuArrayBufferIndex, Texture, TextureView};

pub const DATA_ARRAY_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

pub struct DataArrayTexture<T: Default + Pod> {
    label: Option<String>,
    values: Vec<T>,
    count: usize,
    texture: Option<Texture>,
    texture_view: Option<TextureView>,
    max_extents: Extent3d,
    extents: Extent3d,
    changed: bool,
}

impl<T: Default + Pod> DataArrayTexture<T> {
    pub fn new(device: &RenderDevice) -> Self {
        let limits = device.limits();
        Self {
            label: None,
            values: Vec::new(),
            count: 0,
            texture: None,
            texture_view: None,
            max_extents: Extent3d {
                width: limits.max_texture_dimension_2d,
                height: limits.max_texture_dimension_2d,
                depth_or_array_layers: limits.max_texture_array_layers,
            },
            extents: Extent3d {
                width: 0,
                height: 0,
                depth_or_array_layers: 0,
            },
            changed: true,
        }
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.count = 0;
    }

    pub fn push(&mut self, value: T) -> GpuArrayBufferIndex<T> {
        let index = self.count as u32;
        if self.count < self.values.len() {
            self.values[self.count] = value;
        } else {
            self.values.push(value);
        }
        self.count += 1;
        GpuArrayBufferIndex::<T>::new(index, None)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    fn internal_size(&self) -> usize {
        self.values.len() * size_of::<T>()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.len() * size_of::<T>()
    }

    #[inline]
    fn extents_to_capacity(extents: Extent3d) -> usize {
        (extents.width
            * DATA_ARRAY_TEXTURE_FORMAT.block_size(None).unwrap()
            * extents.height
            * extents.depth_or_array_layers) as usize
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        Self::extents_to_capacity(self.extents)
    }

    #[inline]
    fn size_from_capacity(&self, capacity: usize) -> Extent3d {
        let capacity = capacity as u32;
        let width = self.max_extents.width;
        let row_bytes = width * DATA_ARRAY_TEXTURE_FORMAT.block_size(None).unwrap();
        let mut height = self.max_extents.height;
        let max_layer_bytes = row_bytes * height;
        let depth_or_array_layers = ((capacity + max_layer_bytes - 1) / max_layer_bytes).max(1);
        if depth_or_array_layers == 1 {
            height = ((capacity + row_bytes - 1) / row_bytes).max(1);
        }
        Extent3d {
            width,
            height,
            depth_or_array_layers,
        }
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        let capacity = self.capacity();
        let size = self.size();
        if capacity < size || self.changed {
            self.extents = self.size_from_capacity(size);
            self.texture = Some(device.create_texture(&TextureDescriptor {
                label: self.label.as_deref(),
                size: self.extents,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DATA_ARRAY_TEXTURE_FORMAT,
                usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
            self.texture_view = self.texture.as_ref().map(|texture| {
                texture.create_view(&TextureViewDescriptor {
                    label: self.label.as_deref(),
                    format: Some(DATA_ARRAY_TEXTURE_FORMAT),
                    dimension: Some(TextureViewDimension::D2Array),
                    aspect: TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(self.extents.depth_or_array_layers),
                })
            });
        }
        // Pad the buffer to be large enough
        let buffer_size_bytes = self.internal_size();
        let texture_size_bytes =
            Self::extents_to_capacity(self.texture.as_ref().map(|texture| texture.size()).unwrap());
        if buffer_size_bytes < texture_size_bytes {
            self.values.resize_with(
                (texture_size_bytes + size_of::<T>() - 1) / size_of::<T>(),
                Default::default,
            );
        }
        self.texture.as_ref().map(|texture| {
            queue.write_texture(
                texture.as_image_copy(),
                &cast_slice(&self.values)[..texture_size_bytes],
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: DATA_ARRAY_TEXTURE_FORMAT
                        .block_size(None)
                        .map(|block_size| block_size * texture.size().width),
                    rows_per_image: Some(texture.size().height),
                },
                texture.size(),
            );
        });
    }

    pub fn binding_layout(
        binding: u32,
        visibility: ShaderStages,
        _device: &RenderDevice,
    ) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }
    }

    pub fn binding(&self) -> Option<BindingResource> {
        self.texture_view
            .as_ref()
            .map(|texture_view| BindingResource::TextureView(texture_view))
    }
}
