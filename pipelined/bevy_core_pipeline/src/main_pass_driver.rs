use crate::ViewDepthTexture;
use bevy_ecs::world::World;
use bevy_render2::{
    camera::{CameraPlugin, ExtractedCamera, ExtractedCameraNames},
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType, SlotValue},
    renderer::RenderContext,
    view::ExtractedWindows,
};

pub struct MainPassDriverNode;

impl Node for MainPassDriverNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new("hdr_target", SlotType::TextureView)]
    } 

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let extracted_cameras = world.get_resource::<ExtractedCameraNames>().unwrap();
        let extracted_windows = world.get_resource::<ExtractedWindows>().unwrap();

        if let Some(camera_2d) = extracted_cameras.entities.get(CameraPlugin::CAMERA_2D) {
            let extracted_camera = world.entity(*camera_2d).get::<ExtractedCamera>().unwrap();
            let extracted_window = extracted_windows.get(&extracted_camera.window_id).unwrap();
            let swap_chain_texture = extracted_window.swap_chain_frame.as_ref().unwrap().clone();
            graph.run_sub_graph(
                crate::draw_2d_graph::NAME,
                vec![
                    SlotValue::Entity(*camera_2d),
                    SlotValue::TextureView(swap_chain_texture),
                ],
            )?;
        }

        if let Some(camera_3d) = extracted_cameras.entities.get(CameraPlugin::CAMERA_3D) {
            let depth_texture = world.entity(*camera_3d).get::<ViewDepthTexture>().unwrap();
            let hdr_target = graph.get_input_texture("hdr_target").unwrap().clone();
            graph.run_sub_graph(
                crate::draw_3d_graph::NAME,
                vec![
                    SlotValue::Entity(*camera_3d),
                    SlotValue::TextureView(hdr_target),
                    SlotValue::TextureView(depth_texture.view.clone()),
                ],
            )?;
        }

        Ok(())
    }
}
