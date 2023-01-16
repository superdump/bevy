use std::ops::Range;

use super::{UiImageBindGroups, UiMeta};
use crate::{prelude::UiCameraConfig, DefaultCameraView, ExtractedUiNode};
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_render::{
    render_graph::*,
    render_phase::*,
    render_resource::{CachedRenderPipelineId, LoadOp, Operations, RenderPassDescriptor},
    renderer::*,
    view::*,
};
use bevy_utils::FloatOrd;

pub struct UiPassNode {
    ui_view_query: QueryState<
        (
            &'static RenderPhase<TransparentUi>,
            &'static ViewTarget,
            Option<&'static UiCameraConfig>,
        ),
        With<ExtractedView>,
    >,
    default_camera_view_query: QueryState<&'static DefaultCameraView>,
}

impl UiPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            ui_view_query: world.query_filtered(),
            default_camera_view_query: world.query(),
        }
    }
}

impl Node for UiPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(UiPassNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.ui_view_query.update_archetypes(world);
        self.default_camera_view_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let input_view_entity = graph.get_input_entity(Self::IN_VIEW)?;

        let Ok((transparent_phase, target, camera_ui)) =
                self.ui_view_query.get_manual(world, input_view_entity)
             else {
                return Ok(());
            };
        if transparent_phase.items.is_empty() {
            return Ok(());
        }
        // Don't render UI for cameras where it is explicitly disabled
        if matches!(camera_ui, Some(&UiCameraConfig { show_ui: false })) {
            return Ok(());
        }

        // use the "default" view entity if it is defined
        let view_entity = if let Ok(default_view) = self
            .default_camera_view_query
            .get_manual(world, input_view_entity)
        {
            default_view.0
        } else {
            input_view_entity
        };
        let pass_descriptor = RenderPassDescriptor {
            label: Some("ui_pass"),
            color_attachments: &[Some(target.get_unsampled_color_attachment(Operations {
                load: LoadOp::Load,
                store: true,
            }))],
            depth_stencil_attachment: None,
        };

        let render_pass = render_context
            .command_encoder
            .begin_render_pass(&pass_descriptor);
        let mut render_pass = TrackedRenderPass::new(render_pass);

        transparent_phase.render(&mut render_pass, world, view_entity);

        Ok(())
    }
}

pub struct TransparentUi {
    pub sort_key: <Self as PhaseItem>::SortKey,
    pub entity: Entity,
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub batch_range: Option<Range<u32>>,
    pub dynamic_offset: Option<u32>,
}

impl PhaseItem for TransparentUi {
    type SortKey = (usize, FloatOrd);

    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl CachedRenderPipelinePhaseItem for TransparentUi {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

impl BatchedPhaseItem for TransparentUi {
    fn batch_range(&self) -> &Option<std::ops::Range<u32>> {
        &self.batch_range
    }

    fn batch_range_mut(&mut self) -> &mut Option<std::ops::Range<u32>> {
        &mut self.batch_range
    }

    fn batch_dynamic_offset(&self) -> &Option<u32> {
        &self.dynamic_offset
    }

    fn batch_dynamic_offset_mut(&mut self) -> &mut Option<u32> {
        &mut self.dynamic_offset
    }
}

pub type DrawUi = (
    SetItemPipeline,
    SetUiViewBindGroup<0>,
    SetUiTextureBindGroup<1>,
    DrawUiNode,
);

pub struct SetUiViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiViewBindGroup<I> {
    type Param = SRes<UiMeta>;
    type ViewWorldQuery = Read<ViewUniformOffset>;
    type ItemWorldQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: (),
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            ui_meta.into_inner().view_bind_group.as_ref().unwrap(),
            &[view_uniform.offset],
        );
        RenderCommandResult::Success
    }
}
pub struct SetUiTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiTextureBindGroup<I> {
    type Param = SRes<UiImageBindGroups>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<ExtractedUiNode>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        uinode: &'w ExtractedUiNode,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let image_bind_groups = image_bind_groups.into_inner();
        pass.set_bind_group(I, image_bind_groups.values.get(&uinode.image).unwrap(), &[]);
        RenderCommandResult::Success
    }
}
pub struct DrawUiNode;
impl<P: BatchedPhaseItem> RenderCommand<P> for DrawUiNode {
    type Param = SRes<UiMeta>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _: (),
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_vertex_buffer(0, ui_meta.into_inner().vertices.buffer().unwrap().slice(..));
        let batch_range = item.batch_range().as_ref().unwrap();
        pass.draw(batch_range.clone(), 0..1);
        RenderCommandResult::SuccessfulDraw(batch_range.len() / 6)
    }
}
