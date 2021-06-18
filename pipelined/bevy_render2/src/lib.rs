pub mod camera;
pub mod color;
pub mod core_pipeline;
pub mod mesh;
pub mod pass;
pub mod pipeline;
pub mod render_command;
pub mod render_graph;
pub mod render_phase;
pub mod render_resource;
pub mod renderer;
pub mod shader;
pub mod texture;
pub mod view;

pub use once_cell;

use crate::{
    camera::CameraPlugin,
    mesh::MeshPlugin,
    render_command::RenderCommandPlugin,
    render_graph::RenderGraph,
    render_phase::DrawFunctions,
    renderer::RenderResources,
    texture::TexturePlugin,
    view::{ViewPlugin, WindowRenderPlugin},
};
use bevy_app::{App, Plugin, StartupStage};
use bevy_ecs::prelude::*;
use bevy_utils::tracing::warn;

pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        camera::*,
        color::Color,
        mesh::{shape, Mesh},
        pass::ClearColor,
        shader::Shader,
        texture::Texture,
    };
}

#[derive(Default)]
pub struct RenderPlugin;

/// The names of the default App stages
#[derive(Debug, Hash, PartialEq, Eq, Clone, StageLabel)]
pub enum RenderStage {
    /// Extract data from "app world" and insert it into "render world". This step should be kept
    /// as short as possible to increase the "pipelining potential" for running the next frame
    /// while rendering the current frame.
    Extract,

    /// Prepare render resources from extracted data.
    Prepare,

    /// Create Bind Groups that depend on Prepare data and queue up draw calls to run during the Render stage.
    Queue,

    // TODO: This could probably be moved in favor of a system ordering abstraction in Render or Queue
    /// Sort RenderPhases here 
    PhaseSort,

    /// Actual rendering happens here. In most cases, only the render backend should insert resources here
    Render,

    /// Cleanup render resources here.
    Cleanup,
}

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system_to_stage(
            StartupStage::PreStartup,
            check_for_render_resource_context.system(),
        );
        let mut render_app = App::empty();
        let mut extract_stage = SystemStage::parallel();
        // don't apply buffers when the stage finishes running
        // extract stage runs on the app world, but the buffers are applied to the render world
        extract_stage.set_apply_buffers(false);
        render_app
            .add_stage(RenderStage::Extract, extract_stage)
            .add_stage(RenderStage::Prepare, SystemStage::parallel())
            .add_stage(RenderStage::Queue, SystemStage::parallel())
            .add_stage(RenderStage::PhaseSort, SystemStage::parallel())
            .add_stage(RenderStage::Render, SystemStage::parallel())
            .add_stage(RenderStage::Cleanup, SystemStage::parallel())
            .init_resource::<RenderGraph>()
            .init_resource::<DrawFunctions>();

        app.add_sub_app(render_app, |app_world, render_app| {
            // reserve all existing app entities for use in render_app
            // they can only be spawned using `get_or_spawn()`
            let meta_len = app_world.entities().meta.len();
            render_app
                .world
                .entities()
                .reserve_entities(meta_len as u32);
            // flushing as "invalid" ensures that app world entities aren't added as "empty archetype" entities by default
            // these entities cannot be accessed without spawning directly onto them
            // this _only_ works as expected because clear_entities() is called at the end of every frame.
            render_app.world.entities_mut().flush_as_invalid();

            // extract
            extract(app_world, render_app);

            // prepare
            let prepare = render_app
                .schedule
                .get_stage_mut::<SystemStage>(&RenderStage::Prepare)
                .unwrap();
            prepare.run(&mut render_app.world);

            // queue
            let queue = render_app
                .schedule
                .get_stage_mut::<SystemStage>(&RenderStage::Queue)
                .unwrap();
            queue.run(&mut render_app.world);

            // phase sort
            let phase_sort = render_app
                .schedule
                .get_stage_mut::<SystemStage>(&RenderStage::PhaseSort)
                .unwrap();
            phase_sort.run(&mut render_app.world);

            // render
            let render = render_app
                .schedule
                .get_stage_mut::<SystemStage>(&RenderStage::Render)
                .unwrap();
            render.run(&mut render_app.world);

            // cleanup
            let cleanup = render_app
                .schedule
                .get_stage_mut::<SystemStage>(&RenderStage::Cleanup)
                .unwrap();
            cleanup.run(&mut render_app.world);

            render_app.world.clear_entities();
        });

        app.add_plugin(RenderCommandPlugin)
            .add_plugin(WindowRenderPlugin)
            .add_plugin(CameraPlugin)
            .add_plugin(ViewPlugin)
            .add_plugin(MeshPlugin)
            .add_plugin(TexturePlugin);
    }
}

fn extract(app_world: &mut World, render_app: &mut App) {
    let extract = render_app
        .schedule
        .get_stage_mut::<SystemStage>(&RenderStage::Extract)
        .unwrap();
    extract.run(app_world);
    extract.apply_buffers(&mut render_app.world);
}

fn check_for_render_resource_context(context: Option<Res<RenderResources>>) {
    if context.is_none() {
        warn!(
            "bevy_render couldn't find a render backend. Perhaps try adding the bevy_wgpu feature/plugin!"
        );
    }
}
