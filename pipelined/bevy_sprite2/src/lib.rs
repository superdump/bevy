mod bundle;
mod rect;
mod render;
mod sprite;

pub use bundle::*;
pub use rect::*;
pub use render::*;
pub use sprite::*;

use bevy_app::prelude::*;
use bevy_ecs::prelude::IntoSystem;
use bevy_render2::{
    core_pipeline, render_graph::RenderGraph, render_phase::DrawFunctions, RenderStage,
};

pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        bundle::PipelinedSpriteBundle,
        Sprite, SpriteResizeMode,
    };
}

#[derive(Default)]
pub struct SpritePlugin;

impl Plugin for SpritePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Sprite>();
        let render_app = app.sub_app_mut(0);
        render_app
            .add_system_to_stage(RenderStage::Extract, render::extract_sprites.system())
            .add_system_to_stage(RenderStage::Prepare, render::prepare_sprites.system())
            .add_system_to_stage(RenderStage::Queue, queue_sprites.system())
            .init_resource::<SpriteShaders>()
            .init_resource::<SpriteMeta>();
        let draw_sprite = DrawSprite::new(&mut render_app.world);
        render_app
            .world
            .get_resource::<DrawFunctions>()
            .unwrap()
            .write()
            .add(draw_sprite);
        let render_world = app.sub_app_mut(0).world.cell();
        let mut graph = render_world.get_resource_mut::<RenderGraph>().unwrap();
        graph.add_node("sprite", SpriteNode);
        graph
            .add_node_edge("sprite", core_pipeline::node::MAIN_PASS_DEPENDENCIES)
            .unwrap();
    }
}
