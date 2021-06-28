mod bundle;
mod light;
mod material;
mod render;

pub use bundle::*;
pub use light::*;
pub use material::*;
pub use render::*;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_render2::{
    core_pipeline,
    render_graph::RenderGraph,
    render_phase::{sort_phase_system, DrawFunctions},
    RenderStage,
};

pub mod draw_3d_graph {
    pub mod node {
        pub const SHADOW_PASS: &'static str = "shadow_pass";
        pub const DEPTH_NORMAL_PASS: &'static str = "depth_normal_pass";
        pub const SSAO_PASS: &'static str = "ssao_pass";
    }
}

#[derive(Default)]
pub struct PbrPlugin;

impl Plugin for PbrPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(StandardMaterialPlugin)
            .init_resource::<AmbientLight>()
            .init_resource::<SsaoConfig>()
            .add_startup_system(render::ssao_pass::load_blue_noise.system());

        let render_app = app.sub_app_mut(0);
        render_app
            .add_system_to_stage(RenderStage::Extract, render::extract_meshes.system())
            .add_system_to_stage(
                RenderStage::Extract,
                render::depth_normal_prepass::extract_depth_normal_prepass_camera_phase.system(),
            )
            .add_system_to_stage(
                RenderStage::Extract,
                render::ssao_pass::extract_ssao.system(),
            )
            .add_system_to_stage(RenderStage::Extract, render::extract_lights.system())
            .add_system_to_stage(RenderStage::Prepare, render::prepare_meshes.system())
            .add_system_to_stage(
                RenderStage::Prepare,
                render::depth_normal_prepass::prepare_view_depth_normals.system(),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                render::ssao_pass::prepare_ssao.system(),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                // this is added as an exclusive system because it contributes new views. it must run (and have Commands applied)
                // _before_ the `prepare_views()` system is run. ideally this becomes a normal system when "stageless" features come out
                render::prepare_lights.exclusive_system(),
            )
            .add_system_to_stage(RenderStage::Queue, render::queue_meshes.system())
            .add_system_to_stage(
                RenderStage::Queue,
                render::depth_normal_prepass::queue_meshes.system(),
            )
            .add_system_to_stage(RenderStage::Queue, render::ssao_pass::queue_meshes.system())
            .add_system_to_stage(RenderStage::Queue, render::light::queue_meshes.system())
            .add_system_to_stage(
                RenderStage::PhaseSort,
                sort_phase_system::<DepthNormalPhase>.system(),
            )
            .add_system_to_stage(
                RenderStage::PhaseSort,
                sort_phase_system::<SsaoPhase>.system(),
            )
            .add_system_to_stage(
                RenderStage::PhaseSort,
                sort_phase_system::<ShadowPhase>.system(),
            )
            // FIXME: Hack to ensure RenderCommandQueue is initialized when PbrShaders is being initialized
            // .init_resource::<RenderCommandQueue>()
            .init_resource::<PbrShaders>()
            .init_resource::<DepthNormalShaders>()
            .init_resource::<SsaoShaders>()
            .init_resource::<ShadowShaders>()
            .init_resource::<MeshMeta>()
            .init_resource::<LightMeta>()
            .init_resource::<SsaoMeta>();

        let draw_pbr = DrawPbr::new(&mut render_app.world);
        let draw_depth_normal_mesh = DrawDepthNormalMesh::new(&mut render_app.world);
        let depth_normal_pass_node = DepthNormalPassNode::new(&mut render_app.world);
        let draw_ssao = DrawSsao::new(&mut render_app.world);
        let ssao_pass_node = SsaoPassNode::new(&mut render_app.world);
        let draw_shadow_mesh = DrawShadowMesh::new(&mut render_app.world);
        let shadow_pass_node = ShadowPassNode::new(&mut render_app.world);
        let render_world = render_app.world.cell();
        let draw_functions = render_world.get_resource::<DrawFunctions>().unwrap();
        draw_functions.write().add(draw_pbr);
        draw_functions.write().add(draw_depth_normal_mesh);
        draw_functions.write().add(draw_ssao);
        draw_functions.write().add(draw_shadow_mesh);
        let mut graph = render_world.get_resource_mut::<RenderGraph>().unwrap();
        graph.add_node("pbr", PbrNode);
        graph
            .add_node_edge("pbr", core_pipeline::node::MAIN_PASS_DEPENDENCIES)
            .unwrap();

        let draw_3d_graph = graph
            .get_sub_graph_mut(core_pipeline::draw_3d_graph::NAME)
            .unwrap();

        // Depth and view normal pre-pass
        draw_3d_graph.add_node(
            draw_3d_graph::node::DEPTH_NORMAL_PASS,
            depth_normal_pass_node,
        );
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::DEPTH_NORMAL_PASS,
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::DEPTH_NORMAL_PASS,
                DepthNormalPassNode::IN_VIEW,
            )
            .unwrap();

        // Ssao pass
        draw_3d_graph.add_node(draw_3d_graph::node::SSAO_PASS, ssao_pass_node);
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::DEPTH_NORMAL_PASS,
                draw_3d_graph::node::SSAO_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::SSAO_PASS,
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::SSAO_PASS,
                SsaoPassNode::IN_VIEW,
            )
            .unwrap();

        // Shadows
        draw_3d_graph.add_node(draw_3d_graph::node::SHADOW_PASS, shadow_pass_node);
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::SHADOW_PASS,
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::SHADOW_PASS,
                ShadowPassNode::IN_VIEW,
            )
            .unwrap();
    }
}
