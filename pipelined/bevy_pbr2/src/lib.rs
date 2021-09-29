mod alpha;
mod bundle;
mod light;
mod material;
mod render;

pub use alpha::*;
pub use bundle::*;
pub use light::*;
pub use material::*;
pub use render::*;

use bevy_app::prelude::*;
use bevy_asset::Handle;
use bevy_core_pipeline::{AlphaMask3d, Opaque3d, Transparent3d};
use bevy_ecs::prelude::*;
use bevy_render2::{
    render_component::{ExtractComponentPlugin, UniformComponentPlugin},
    render_graph::RenderGraph,
    render_phase::{sort_phase_system, AddRenderCommand, DrawFunctions},
    view::VisibilitySystems,
    RenderApp, RenderStage,
};
use bevy_transform::TransformSystem;

pub mod draw_3d_graph {
    pub mod node {
        pub const DEPTH_PREPASS: &str = "depth_prepass";
        pub const SHADOW_PASS: &str = "shadow_pass";
    }
}

#[derive(Default)]
pub struct PbrPlugin;

impl Plugin for PbrPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(StandardMaterialPlugin)
            .add_plugin(ExtractComponentPlugin::<Handle<StandardMaterial>>::default())
            .add_plugin(UniformComponentPlugin::<MeshUniform>::default())
            .init_resource::<AmbientLight>()
            .init_resource::<DirectionalLightShadowMap>()
            .init_resource::<PointLightShadowMap>()
            .init_resource::<AmbientLight>()
            .init_resource::<VisiblePointLights>()
            .add_system_to_stage(
                CoreStage::PostUpdate,
                // NOTE: Clusters need to have been added before update_clusters is run so
                //       add as an exclusive system
                render::add_clusters
                    .exclusive_system()
                    .label(LightSystems::AddClusters)
                    .after(TransformSystem::TransformPropagate),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                render::update_clusters
                    .label(LightSystems::UpdateClusters)
                    .after(TransformSystem::TransformPropagate)
                    .after(LightSystems::AddClusters),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                render::assign_lights_to_clusters
                    .label(LightSystems::AssignLightsToClusters)
                    .after(TransformSystem::TransformPropagate)
                    .after(LightSystems::UpdateClusters),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                render::update_directional_light_frusta
                    .label(LightSystems::UpdateDirectionalLightFrusta)
                    .after(TransformSystem::TransformPropagate),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                render::update_point_light_frusta
                    .label(LightSystems::UpdatePointLightFrusta)
                    .after(TransformSystem::TransformPropagate)
                    .after(LightSystems::AssignLightsToClusters),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                render::check_light_mesh_visibility
                    .label(LightSystems::CheckLightVisibility)
                    .after(TransformSystem::TransformPropagate)
                    .after(VisibilitySystems::CalculateBounds)
                    .after(LightSystems::UpdateDirectionalLightFrusta)
                    .after(LightSystems::UpdatePointLightFrusta)
                    // NOTE: This MUST be scheduled AFTER the core renderer visibility check
                    //       because that resets entity ComputedVisibility for the first view
                    //       which would override any results from this otherwise
                    .after(VisibilitySystems::CheckVisibility),
            );

        {
            let render_app = app.sub_app(RenderApp);
            render_app
                .add_system_to_stage(RenderStage::Extract, render::extract_meshes)
                .add_system_to_stage(
                    RenderStage::Extract,
                    render::extract_clusters.label(LightSystems::ExtractClusters),
                )
                .add_system_to_stage(
                    RenderStage::Extract,
                    render::extract_lights.label(LightSystems::ExtractLights),
                )
                .add_system_to_stage(
                    RenderStage::Prepare,
                    // this is added as an exclusive system because it contributes new views. it must run (and have Commands applied)
                    // _before_ the `prepare_views()` system is run. ideally this becomes a normal system when "stageless" features come out
                    render::prepare_lights
                        .exclusive_system()
                        .label(LightSystems::PrepareLights),
                )
                .add_system_to_stage(
                    RenderStage::Prepare,
                    // FIXME: Is this true?
                    // this is added as an exclusive system because it contributes new views. it must run (and have Commands applied)
                    // _before_ the `prepare_views()` system is run. ideally this becomes a normal system when "stageless" features come out
                    render::prepare_clusters
                        .exclusive_system()
                        .label(LightSystems::PrepareClusters)
                        .after(LightSystems::PrepareLights),
                )
                .add_system_to_stage(RenderStage::Queue, render::queue_meshes)
                .add_system_to_stage(
                    RenderStage::Queue,
                    render::queue_shadows.label(LightSystems::QueueShadows),
                )
                .add_system_to_stage(RenderStage::Queue, render::queue_shadow_view_bind_group)
                .add_system_to_stage(RenderStage::Queue, render::queue_transform_bind_group)
                .add_system_to_stage(RenderStage::PhaseSort, sort_phase_system::<Shadow>)
                .init_resource::<PbrShaders>()
                .init_resource::<ShadowShaders>()
                .init_resource::<DrawFunctions<Shadow>>()
                .init_resource::<LightMeta>()
                .init_resource::<GlobalLightMeta>();

            let draw_shadow_mesh = DrawShadowMesh::new(&mut render_app.world);
            let shadow_pass_node = ShadowPassNode::new(&mut render_app.world);
            render_app.add_render_command::<Opaque3d, DrawPbr>();
            render_app.add_render_command::<AlphaMask3d, DrawPbr>();
            render_app.add_render_command::<Transparent3d, DrawPbr>();
            let render_world = render_app.world.cell();
            let draw_functions = render_world
                .get_resource::<DrawFunctions<Shadow>>()
                .unwrap();
            draw_functions.write().add(draw_shadow_mesh);
            let mut graph = render_world.get_resource_mut::<RenderGraph>().unwrap();
            let draw_3d_graph = graph
                .get_sub_graph_mut(bevy_core_pipeline::draw_3d_graph::NAME)
                .unwrap();
            draw_3d_graph.add_node(draw_3d_graph::node::SHADOW_PASS, shadow_pass_node);
            draw_3d_graph
                .add_node_edge(
                    draw_3d_graph::node::SHADOW_PASS,
                    bevy_core_pipeline::draw_3d_graph::node::MAIN_PASS,
                )
                .unwrap();
            draw_3d_graph
                .add_slot_edge(
                    draw_3d_graph.input_node().unwrap().id,
                    bevy_core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                    draw_3d_graph::node::SHADOW_PASS,
                    ShadowPassNode::IN_VIEW,
                )
                .unwrap();
        }
        app.add_plugin(DepthPrepassPlugin);
    }
}
