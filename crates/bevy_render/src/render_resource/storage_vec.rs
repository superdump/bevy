use std::num::NonZeroU64;

use bevy_crevice::std430::{self, AsStd430, Std430};
use bevy_utils::tracing::warn;
use wgpu::{BindingResource, BufferBinding, BufferDescriptor, BufferUsages};

use crate::renderer::{RenderDevice, RenderQueue};

use super::Buffer;

pub struct StorageVec<T: AsStd430> {
    values: Vec<T>,
    scratch: Vec<u8>,
    storage_buffer: Option<Buffer>,
    capacity: usize,
    item_size: usize,
}

impl<T: AsStd430> Default for StorageVec<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            scratch: Vec::new(),
            storage_buffer: None,
            capacity: 0,
            // FIXME: Is this correct or even necessary? Maybe only need T::std430_size_static()?
            item_size: (T::std430_size_static() + <T as AsStd430>::Output::ALIGNMENT - 1)
                & !(<T as AsStd430>::Output::ALIGNMENT - 1),
        }
    }
}

impl<T: AsStd430> StorageVec<T> {
    #[inline]
    pub fn storage_buffer(&self) -> Option<&Buffer> {
        self.storage_buffer.as_ref()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.storage_buffer()?,
            offset: 0,
            size: Some(NonZeroU64::new((self.item_size * self.values.len()) as u64).unwrap()),
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
        let index = self.values.len();
        self.values.push(value);
        index
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.values[index]
    }

    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) -> bool {
        if capacity > self.capacity {
            self.capacity = capacity;
            let size = self.item_size * capacity;
            self.scratch.resize(size, 0);
            self.storage_buffer = Some(device.create_buffer(&BufferDescriptor {
                label: None,
                size: size as wgpu::BufferAddress,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            true
        } else {
            false
        }
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.values.is_empty() {
            return;
        }
        self.reserve(self.values.len(), device);
        if let Some(storage_buffer) = &self.storage_buffer {
            let range = 0..self.item_size * self.values.len();
            let mut writer = std430::Writer::new(&mut self.scratch[range.clone()]);
            // NOTE: Failing to write should be non-fatal. It would likely cause visual
            // artifacts but a warning message should suffice.
            writer
                .write(self.values.as_slice())
                .map_err(|e| warn!("{:?}", e))
                .ok();
            queue.write_buffer(storage_buffer, 0, &self.scratch[range]);
        }
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}
