use crate::{ClearColor, Opaque3dPhase, Transparent3dPhase};
use bevy_ecs::prelude::*;
use bevy_render2::{
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_phase::{DrawFunctions, RenderPhase, TrackedRenderPass},
    render_resource::{
        LoadOp, Operations, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
        RenderPassDescriptor,
    },
    renderer::RenderContext,
    view::ExtractedView,
};

pub struct MainPass3dNode {
    query: QueryState<
        (
            &'static RenderPhase<Opaque3dPhase>,
            &'static RenderPhase<Transparent3dPhase>,
        ),
        With<ExtractedView>,
    >,
}

impl MainPass3dNode {
    pub const IN_COLOR_ATTACHMENT: &'static str = "color_attachment";
    pub const IN_DEPTH: &'static str = "depth";
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for MainPass3dNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![
            SlotInfo::new(MainPass3dNode::IN_COLOR_ATTACHMENT, SlotType::TextureView),
            SlotInfo::new(MainPass3dNode::IN_DEPTH, SlotType::TextureView),
            SlotInfo::new(MainPass3dNode::IN_VIEW, SlotType::Entity),
        ]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let color_attachment_texture = graph.get_input_texture(Self::IN_COLOR_ATTACHMENT)?;
        let clear_color = world.get_resource::<ClearColor>().unwrap();
        let depth_texture = graph.get_input_texture(Self::IN_DEPTH)?;

        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let draw_functions = world.get_resource::<DrawFunctions>().unwrap();

        let (opaque_phase, transparent_phase) = self
            .query
            .get_manual(world, view_entity)
            .expect("view entity should exist");

        let mut draw_functions = draw_functions.write();

        {
            // Run the opaque pass, sorted front-to-back
            // NOTE: Scoped to drop the mutable borrow of render_context
            let opaque_pass_descriptor = RenderPassDescriptor {
                label: Some("main_opaque_pass_3d"),
                color_attachments: &[RenderPassColorAttachment {
                    view: color_attachment_texture,
                    resolve_target: None,
                    // NOTE: The opaque pass clears and initializes the color
                    //       buffer as well as writing to it.
                    ops: Operations {
                        load: LoadOp::Clear(clear_color.0.into()),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_texture,
                    // NOTE: The opaque pass clears and writes to the depth buffer.
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };
            let opaque_render_pass = render_context
                .command_encoder
                .begin_render_pass(&opaque_pass_descriptor);
            let mut opaque_tracked_pass = TrackedRenderPass::new(opaque_render_pass);
            for drawable in opaque_phase.drawn_things.iter() {
                let draw_function = draw_functions.get_mut(drawable.draw_function).unwrap();
                draw_function.draw(
                    world,
                    &mut opaque_tracked_pass,
                    view_entity,
                    drawable.draw_key,
                    drawable.sort_key,
                );
            }
        }

        {
            // Run the transparent pass, sorted back-to-front
            // NOTE: Scoped to drop the mutable borrow of render_context
            let transparent_pass_descriptor = RenderPassDescriptor {
                label: Some("main_transparent_pass_3d"),
                color_attachments: &[RenderPassColorAttachment {
                    view: color_attachment_texture,
                    resolve_target: None,
                    // NOTE: For the transparent pass we load the color and overwrite it
                    ops: Operations {
                        load: LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_texture,
                    // NOTE: For the transparent pass we load the depth buffer but do not write to it.
                    //       As the opaque pass is run first, opaque meshes can occlude transparent ones.
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            };
            let transparent_render_pass = render_context
                .command_encoder
                .begin_render_pass(&transparent_pass_descriptor);
            let mut transparent_tracked_pass = TrackedRenderPass::new(transparent_render_pass);
            for drawable in transparent_phase.drawn_things.iter() {
                let draw_function = draw_functions.get_mut(drawable.draw_function).unwrap();
                draw_function.draw(
                    world,
                    &mut transparent_tracked_pass,
                    view_entity,
                    drawable.draw_key,
                    drawable.sort_key,
                );
            }
        }

        Ok(())
    }
}
