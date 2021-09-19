use crate::{
    render_resource::Buffer,
    renderer::{RenderDevice, RenderQueue},
};
use crevice::std140::{self, AsStd140, DynamicUniform, Std140};
use std::num::NonZeroU64;
use wgpu::{BindingResource, BufferBinding, BufferDescriptor, BufferUsage};

pub struct AlignedBufferVec<T: AsStd140> {
    values: Vec<T>,
    scratch: Vec<u8>,
    buffer_usage: BufferUsage,
    buffer: Option<Buffer>,
    capacity: usize,
    item_size: usize,
}

impl<T: AsStd140> Default for AlignedBufferVec<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            scratch: Vec::new(),
            buffer_usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
            buffer: None,
            capacity: 0,
            item_size: (T::std140_size_static() + <T as AsStd140>::Std140Type::ALIGNMENT - 1)
                & !(<T as AsStd140>::Std140Type::ALIGNMENT - 1),
        }
    }
}

impl<T: AsStd140> AlignedBufferVec<T> {
    pub fn new(buffer_usage: BufferUsage) -> Self {
        Self {
            buffer_usage,
            ..Default::default()
        }
    }

    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.buffer()?,
            offset: 0,
            size: Some(NonZeroU64::new(self.item_size as u64).unwrap()),
        }))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn push(&mut self, value: T) -> usize {
        let len = self.values.len();
        if len < self.capacity {
            self.values.push(value);
            len
        } else {
            panic!(
                "Cannot push value because capacity of {} has been reached",
                self.capacity
            );
        }
    }

    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) {
        if capacity > self.capacity {
            self.capacity = capacity;
            let size = self.item_size * capacity;
            self.scratch.resize(size, 0);
            self.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: None,
                size: size as wgpu::BufferAddress,
                usage: BufferUsage::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            }));
        }
    }

    pub fn reserve_and_clear(&mut self, capacity: usize, device: &RenderDevice) {
        self.clear();
        self.reserve(capacity, device);
    }

    pub fn write_buffer(&mut self, queue: &RenderQueue) {
        if let Some(buffer) = &self.buffer {
            let range = 0..self.item_size * self.values.len();
            let mut writer = std140::Writer::new(&mut self.scratch[range.clone()]);
            writer.write(self.values.as_slice()).unwrap();
            queue.write_buffer(buffer, 0, &self.scratch[range]);
        }
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }
}

pub struct DynamicUniformVec<T: AsStd140> {
    uniform_vec: AlignedBufferVec<DynamicUniform<T>>,
}

impl<T: AsStd140> Default for DynamicUniformVec<T> {
    fn default() -> Self {
        Self {
            uniform_vec: Default::default(),
        }
    }
}

impl<T: AsStd140> DynamicUniformVec<T> {
    #[inline]
    pub fn uniform_buffer(&self) -> Option<&Buffer> {
        self.uniform_vec.buffer()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        self.uniform_vec.binding()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.uniform_vec.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.uniform_vec.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.uniform_vec.capacity()
    }

    #[inline]
    pub fn push(&mut self, value: T) -> u32 {
        (self.uniform_vec.push(DynamicUniform(value)) * self.uniform_vec.item_size) as u32
    }

    #[inline]
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) {
        self.uniform_vec.reserve(capacity, device);
    }

    #[inline]
    pub fn reserve_and_clear(&mut self, capacity: usize, device: &RenderDevice) {
        self.uniform_vec.reserve_and_clear(capacity, device);
    }

    #[inline]
    pub fn write_buffer(&mut self, queue: &RenderQueue) {
        self.uniform_vec.write_buffer(queue);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.uniform_vec.clear();
    }
}
