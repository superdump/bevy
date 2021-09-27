use std::{collections::HashMap, sync::Mutex};

use bevy_ecs::prelude::World;
use bevy_render2::{
    camera::{CameraPlugin, ExtractedCamera, ExtractedCameraNames},
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_resource::{Extent3d, TextureDescriptor, TextureDimension, TextureUsage, TextureView},
    renderer::{RenderContext, RenderDevice},
    view::ExtractedWindows,
};
use bevy_window::WindowId;

struct HdrTexture {
    width: u32,
    height: u32,
    view: TextureView,
}

impl HdrTexture {
    pub fn new(ctx: &mut RenderContext, width: u32, height: u32) -> Self {
        let texture = ctx.render_device.create_texture(&TextureDescriptor {
            label: Some("hdr_target"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: crate::HDR_FORMAT,
            dimension: TextureDimension::D2,
            sample_count: 1,
            mip_level_count: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
        });

        HdrTexture {
            width,
            height,
            view: texture.create_view(&Default::default()),
        }
    }
}

/// Outputs an HDR texture with size equal to the window
/// currently targeted by the [`CameraPlugin::CAMERA_3D`].
#[derive(Default)]
pub struct HdrTextureNode {
    // NOTE: it might not be worth it cache the textures
    textures: Mutex<HashMap<WindowId, HdrTexture>>,
    empty_texture: Option<TextureView>,
}

impl HdrTextureNode {
    pub const HDR_TARGET: &'static str = "hdr_target";
}

impl Node for HdrTextureNode {
    fn output(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::HDR_TARGET, SlotType::TextureView)]
    }

    fn update(&mut self, world: &mut World) {
        if self.empty_texture.is_none() {
            let render_device = world.get_resource::<RenderDevice>().unwrap();

            let texture = render_device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                format: crate::HDR_FORMAT,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            });

            self.empty_texture = Some(texture.create_view(&Default::default()));
        }
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let extracted_cameras = world.get_resource::<ExtractedCameraNames>().unwrap();
        let extracted_windows = world.get_resource::<ExtractedWindows>().unwrap();

        if let Some(camera_3d) = extracted_cameras.entities.get(CameraPlugin::CAMERA_3D) {
            let extracted_camera = world.entity(*camera_3d).get::<ExtractedCamera>().unwrap();
            let extracted_window = extracted_windows.get(&extracted_camera.window_id).unwrap();

            let mut textures = self.textures.lock().unwrap();

            let hdr_texture = textures.entry(extracted_window.id).or_insert_with(|| {
                HdrTexture::new(
                    render_context,
                    extracted_window.physical_width,
                    extracted_window.physical_height,
                )
            });

            if hdr_texture.width != extracted_window.physical_width
                || hdr_texture.height != extracted_window.physical_height
            {
                *hdr_texture = HdrTexture::new(
                    render_context,
                    extracted_window.physical_width,
                    extracted_window.physical_height,
                );
            }

            graph.set_output(Self::HDR_TARGET, hdr_texture.view.clone())?;
        } else {
            graph.set_output(Self::HDR_TARGET, self.empty_texture.clone().unwrap())?;
        }

        Ok(())
    }
}
