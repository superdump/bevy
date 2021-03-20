use crate::{
    pipeline::{
        IndexFormat, PipelineCompiler, PipelineDescriptor, PipelineLayout, PipelineSpecialization,
    },
    renderer::{
        AssetRenderResourceBindings, BindGroup, BindGroupId, BufferId, RenderResource,
        RenderResourceBinding, RenderResourceBindings, RenderResourceContext, SharedBuffers,
    },
    shader::Shader,
};
use bevy_asset::{Asset, Assets, Handle};
use bevy_ecs::{
    reflect::ReflectComponent,
    system::{Query, Res, ResMut, SystemParam},
};
use bevy_reflect::Reflect;
use std::{marker::PhantomData, ops::Range, sync::Arc};
use thiserror::Error;

/// A queued command for the renderer
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RenderCommand {
    SetPipeline {
        pipeline: Handle<PipelineDescriptor>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: BufferId,
        offset: u64,
    },
    SetIndexBuffer {
        buffer: BufferId,
        offset: u64,
        index_format: IndexFormat,
    },
    SetBindGroup {
        index: u32,
        bind_group: BindGroupId,
        dynamic_uniform_indices: Option<Arc<[u32]>>,
    },
    DrawIndexed {
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    },
    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
}

#[derive(Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct Visible {
    pub is_visible: bool,
    // TODO: consider moving this to materials
    pub is_transparent: bool,
}

impl Default for Visible {
    fn default() -> Self {
        Visible {
            is_visible: true,
            is_transparent: false,
        }
    }
}

/// A component that indicates how to draw an entity.
#[derive(Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct Draw<P: Send + Sync + 'static> {
    #[reflect(ignore)]
    pub render_commands: Vec<RenderCommand>,
    #[reflect(ignore)]
    pub marker: PhantomData<P>,
}

impl<P: Send + Sync + 'static> Default for Draw<P> {
    fn default() -> Self {
        Self {
            render_commands: Default::default(),
            marker: Default::default(),
        }
    }
}

impl<P: Send + Sync + 'static> Draw<P> {
    pub fn clear_render_commands(&mut self) {
        self.render_commands.clear();
    }

    pub fn set_pipeline(&mut self, pipeline: &Handle<PipelineDescriptor>) {
        self.render_command(RenderCommand::SetPipeline {
            pipeline: pipeline.clone_weak(),
        });
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferId, offset: u64) {
        self.render_command(RenderCommand::SetVertexBuffer {
            slot,
            buffer,
            offset,
        });
    }

    pub fn set_index_buffer(&mut self, buffer: BufferId, offset: u64, index_format: IndexFormat) {
        self.render_command(RenderCommand::SetIndexBuffer {
            buffer,
            offset,
            index_format,
        });
    }

    pub fn set_bind_group(&mut self, index: u32, bind_group: &BindGroup) {
        self.render_command(RenderCommand::SetBindGroup {
            index,
            bind_group: bind_group.id,
            dynamic_uniform_indices: bind_group.dynamic_uniform_indices.clone(),
        });
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.render_command(RenderCommand::DrawIndexed {
            base_vertex,
            indices,
            instances,
        });
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.render_command(RenderCommand::Draw {
            vertices,
            instances,
        });
    }

    #[inline]
    pub fn render_command(&mut self, render_command: RenderCommand) {
        self.render_commands.push(render_command);
    }
}

#[derive(Debug, Error)]
pub enum DrawError {
    #[error("pipeline does not exist")]
    NonExistentPipeline,
    #[error("no pipeline set")]
    NoPipelineSet,
    #[error("pipeline has no layout")]
    PipelineHasNoLayout,
    #[error("failed to get a buffer for the given `RenderResource`")]
    BufferAllocationFailure,
    #[error("the given asset does not have any render resources")]
    MissingAssetRenderResources,
}

#[derive(SystemParam)]
pub struct DrawContext<'a> {
    pub pipelines: ResMut<'a, Assets<PipelineDescriptor>>,
    pub shaders: ResMut<'a, Assets<Shader>>,
    pub asset_render_resource_bindings: ResMut<'a, AssetRenderResourceBindings>,
    pub pipeline_compiler: ResMut<'a, PipelineCompiler>,
    pub render_resource_context: Res<'a, Box<dyn RenderResourceContext>>,
    pub shared_buffers: ResMut<'a, SharedBuffers>,
    #[system_param(ignore)]
    pub current_pipeline: Option<Handle<PipelineDescriptor>>,
}

impl<'a> DrawContext<'a> {
    pub fn get_uniform_buffer<T: RenderResource>(
        &mut self,
        render_resource: &T,
    ) -> Result<RenderResourceBinding, DrawError> {
        self.shared_buffers
            .get_uniform_buffer(&**self.render_resource_context, render_resource)
            .ok_or(DrawError::BufferAllocationFailure)
    }

    pub fn set_pipeline<P: Send + Sync + 'static>(
        &mut self,
        draw: &mut Draw<P>,
        pipeline_handle: &Handle<PipelineDescriptor>,
        specialization: &PipelineSpecialization,
    ) -> Result<(), DrawError> {
        let specialized_pipeline = if let Some(specialized_pipeline) = self
            .pipeline_compiler
            .get_specialized_pipeline(pipeline_handle, specialization)
        {
            specialized_pipeline
        } else {
            self.pipeline_compiler.compile_pipeline(
                &**self.render_resource_context,
                &mut self.pipelines,
                &mut self.shaders,
                pipeline_handle,
                specialization,
            )
        };

        draw.set_pipeline(&specialized_pipeline);
        self.current_pipeline = Some(specialized_pipeline.clone_weak());
        Ok(())
    }

    pub fn get_pipeline_descriptor(&self) -> Result<&PipelineDescriptor, DrawError> {
        self.current_pipeline
            .as_ref()
            .and_then(|handle| self.pipelines.get(handle))
            .ok_or(DrawError::NoPipelineSet)
    }

    pub fn get_pipeline_layout(&self) -> Result<&PipelineLayout, DrawError> {
        self.get_pipeline_descriptor().and_then(|descriptor| {
            descriptor
                .get_layout()
                .ok_or(DrawError::PipelineHasNoLayout)
        })
    }

    pub fn set_asset_bind_groups<T: Asset, P: Send + Sync + 'static>(
        &mut self,
        draw: &mut Draw<P>,
        asset_handle: &Handle<T>,
    ) -> Result<(), DrawError> {
        if let Some(asset_bindings) = self
            .asset_render_resource_bindings
            .get_mut_untyped(&asset_handle.clone_weak_untyped())
        {
            Self::set_bind_groups_from_bindings_internal(
                &self.current_pipeline,
                &self.pipelines,
                &**self.render_resource_context,
                None,
                draw,
                &mut [asset_bindings],
            )
        } else {
            Err(DrawError::MissingAssetRenderResources)
        }
    }

    pub fn set_bind_groups_from_bindings<P: Send + Sync + 'static>(
        &mut self,
        draw: &mut Draw<P>,
        render_resource_bindings: &mut [&mut RenderResourceBindings],
    ) -> Result<(), DrawError> {
        Self::set_bind_groups_from_bindings_internal(
            &self.current_pipeline,
            &self.pipelines,
            &**self.render_resource_context,
            Some(&mut self.asset_render_resource_bindings),
            draw,
            render_resource_bindings,
        )
    }

    fn set_bind_groups_from_bindings_internal<P: Send + Sync + 'static>(
        current_pipeline: &Option<Handle<PipelineDescriptor>>,
        pipelines: &Assets<PipelineDescriptor>,
        render_resource_context: &dyn RenderResourceContext,
        mut asset_render_resource_bindings: Option<&mut AssetRenderResourceBindings>,
        draw: &mut Draw<P>,
        render_resource_bindings: &mut [&mut RenderResourceBindings],
    ) -> Result<(), DrawError> {
        let pipeline = current_pipeline.as_ref().ok_or(DrawError::NoPipelineSet)?;
        let pipeline_descriptor = pipelines
            .get(pipeline)
            .ok_or(DrawError::NonExistentPipeline)?;
        let layout = pipeline_descriptor
            .get_layout()
            .ok_or(DrawError::PipelineHasNoLayout)?;
        'bind_group_descriptors: for bind_group_descriptor in layout.bind_groups.iter() {
            for bindings in render_resource_bindings.iter_mut() {
                if let Some(bind_group) =
                    bindings.update_bind_group(bind_group_descriptor, render_resource_context)
                {
                    draw.set_bind_group(bind_group_descriptor.index, bind_group);
                    continue 'bind_group_descriptors;
                }
            }

            // if none of the given RenderResourceBindings have the current bind group, try their
            // assets
            let asset_render_resource_bindings =
                if let Some(value) = asset_render_resource_bindings.as_mut() {
                    value
                } else {
                    continue 'bind_group_descriptors;
                };
            for bindings in render_resource_bindings.iter_mut() {
                for (asset_handle, _) in bindings.iter_assets() {
                    let asset_bindings = if let Some(asset_bindings) =
                        asset_render_resource_bindings.get_mut_untyped(asset_handle)
                    {
                        asset_bindings
                    } else {
                        continue;
                    };

                    if let Some(bind_group) = asset_bindings
                        .update_bind_group(bind_group_descriptor, render_resource_context)
                    {
                        draw.set_bind_group(bind_group_descriptor.index, bind_group);
                        continue 'bind_group_descriptors;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn create_bind_group_resource(
        &self,
        index: u32,
        bind_group: &BindGroup,
    ) -> Result<(), DrawError> {
        let pipeline = self
            .current_pipeline
            .as_ref()
            .ok_or(DrawError::NoPipelineSet)?;
        let pipeline_descriptor = self
            .pipelines
            .get(pipeline)
            .ok_or(DrawError::NonExistentPipeline)?;
        let layout = pipeline_descriptor
            .get_layout()
            .ok_or(DrawError::PipelineHasNoLayout)?;
        let bind_group_descriptor = &layout.bind_groups[index as usize];
        self.render_resource_context
            .create_bind_group(bind_group_descriptor.id, bind_group);
        Ok(())
    }

    pub fn set_vertex_buffers_from_bindings<P: Send + Sync + 'static>(
        &self,
        draw: &mut Draw<P>,
        render_resource_bindings: &[&RenderResourceBindings],
    ) -> Result<(), DrawError> {
        for bindings in render_resource_bindings.iter() {
            if let Some((index_buffer, index_format)) = bindings.index_buffer {
                draw.set_index_buffer(index_buffer, 0, index_format);
            }
            if let Some(main_vertex_buffer) = bindings.vertex_attribute_buffer {
                draw.set_vertex_buffer(0, main_vertex_buffer, 0);
            }
        }
        Ok(())
    }
}

pub trait Drawable<P: Send + Sync + 'static> {
    fn draw(&mut self, draw: &mut Draw<P>, context: &mut DrawContext) -> Result<(), DrawError>;
}

pub fn clear_draw_system<P: Send + Sync + 'static>(mut query: Query<&mut Draw<P>>) {
    for mut draw in query.iter_mut() {
        draw.clear_render_commands();
    }
}
