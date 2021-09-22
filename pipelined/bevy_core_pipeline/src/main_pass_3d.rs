use crate::{AlphaMask3d, ClearColor, Opaque3d, Transparent3d};
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
            &'static RenderPhase<Opaque3d>,
            &'static RenderPhase<AlphaMask3d>,
            &'static RenderPhase<Transparent3d>,
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

        let (opaque_phase, alpha_mask_phase, transparent_phase) = self
            .query
            .get_manual(world, view_entity)
            .expect("view entity should exist");

        {
            // Run the opaque pass, sorted front-to-back
            // NOTE: Scoped to drop the mutable borrow of render_context
            let pass_descriptor = RenderPassDescriptor {
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
                    // NOTE: The opaque pass loads the depth prepass and does not store
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            };
            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            let draw_functions = world.get_resource::<DrawFunctions<Opaque3d>>().unwrap();
            let mut draw_functions = draw_functions.write();
            for item in opaque_phase.items.iter() {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        {
            // Run the alpha mask pass, sorted front-to-back
            // NOTE: Scoped to drop the mutable borrow of render_context
            let pass_descriptor = RenderPassDescriptor {
                label: Some("main_alpha_mask_pass_3d"),
                color_attachments: &[RenderPassColorAttachment {
                    view: color_attachment_texture,
                    resolve_target: None,
                    // NOTE: The alpha_mask pass loads the color buffer as well as overwriting it where appropriate.
                    ops: Operations {
                        load: LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_texture,
                    // NOTE: The alpha mask pass loads the depth prepass and does not store
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            };
            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            let draw_functions = world.get_resource::<DrawFunctions<AlphaMask3d>>().unwrap();
            let mut draw_functions = draw_functions.write();
            for item in alpha_mask_phase.items.iter() {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        {
            // Run the transparent pass, sorted back-to-front
            // NOTE: Scoped to drop the mutable borrow of render_context
            let pass_descriptor = RenderPassDescriptor {
                label: Some("main_transparent_pass_3d"),
                color_attachments: &[RenderPassColorAttachment {
                    view: color_attachment_texture,
                    resolve_target: None,
                    // NOTE: The transparent pass loads the color buffer as well as overwriting it where appropriate.
                    ops: Operations {
                        load: LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_texture,
                    // NOTE: For the transparent pass we load the depth buffer but do not write to it.
                    //       As the opaque and alpha mask passes run first, opaque meshes can occlude
                    //       transparent ones.
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            };
            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            let draw_functions = world
                .get_resource::<DrawFunctions<Transparent3d>>()
                .unwrap();
            let mut draw_functions = draw_functions.write();
            for item in transparent_phase.items.iter() {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        Ok(())
    }
}
