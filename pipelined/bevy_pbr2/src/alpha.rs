use bevy_app::{App, Plugin};
use bevy_core_pipeline::node;
use bevy_ecs::{
    prelude::{Commands, Entity, Query, Res, ResMut, World},
    reflect::ReflectComponent,
};
use bevy_reflect::Reflect;
use bevy_render2::{
    render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext},
    render_resource::DynamicUniformVec,
    renderer::{RenderContext, RenderDevice},
    RenderApp, RenderStage,
};
use crevice::std140::AsStd140;

// FIXME: This should probably be part of bevy_render2!
/// Alpha mode
#[derive(Debug, Reflect, Clone, PartialEq)]
#[reflect(Component)]
pub enum AlphaMode {
    Opaque,
    /// An alpha cutoff must be supplied where alpha values >= the cutoff
    /// will be fully opaque and < will be fully transparent
    Mask(f32),
    Blend,
}

impl Eq for AlphaMode {}

impl Default for AlphaMode {
    fn default() -> Self {
        AlphaMode::Opaque
    }
}

// NOTE: These must match the bit flags in bevy_pbr2/src/render/pbr.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    struct AlphaModeFlags: u32 {
        const OPAQUE                     = 0;
        const MASK                       = (1 << 0);
        const BLEND                      = (1 << 1);
        const UNINITIALIZED              = 0xFFFF;
    }
}

pub struct AlphaModePlugin;

impl AlphaModePlugin {
    pub const ALPHA_MODE_NODE: &'static str = "alpha_mode";
}

impl Plugin for AlphaModePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AlphaMode>();

        let render_app = app.sub_app(RenderApp);
        render_app
            .init_resource::<AlphaModeMeta>()
            .add_system_to_stage(RenderStage::Extract, extract_alpha_mode)
            .add_system_to_stage(RenderStage::Prepare, prepare_alpha_mode);

        let mut graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();
        graph.add_node(AlphaModePlugin::ALPHA_MODE_NODE, AlphaModeNode);
        graph
            .add_node_edge(
                AlphaModePlugin::ALPHA_MODE_NODE,
                node::MAIN_PASS_DEPENDENCIES,
            )
            .unwrap();
    }
}

pub type ExtractedAlphaMode = AlphaMode;

fn extract_alpha_mode(mut commands: Commands, query: Query<(Entity, &AlphaMode)>) {
    for (entity, alpha_mode) in query.iter() {
        commands
            .get_or_spawn(entity)
            .insert(alpha_mode.clone() as ExtractedAlphaMode);
    }
}

#[derive(Clone, AsStd140)]
pub struct AlphaModeUniform {
    pub mode: u32,
    pub cutoff: f32,
}

#[derive(Default)]
pub struct AlphaModeMeta {
    pub uniforms: DynamicUniformVec<AlphaModeUniform>,
}

pub struct AlphaModeUniformOffset {
    pub offset: u32,
}

fn prepare_alpha_mode(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut alpha_mode_meta: ResMut<AlphaModeMeta>,
    query: Query<(Entity, &AlphaMode)>,
) {
    let len = query.iter().len();
    alpha_mode_meta
        .uniforms
        .reserve_and_clear(len, &render_device);
    for (entity, alpha_mode) in query.iter() {
        // NOTE: 0.5 is from the glTF default - do we want this?
        let mut cutoff = 0.5;
        let mode = match alpha_mode {
            AlphaMode::Opaque => AlphaModeFlags::OPAQUE.bits,
            AlphaMode::Mask(c) => {
                cutoff = *c;
                AlphaModeFlags::MASK.bits
            }
            AlphaMode::Blend => AlphaModeFlags::BLEND.bits,
        };

        commands
            .get_or_spawn(entity)
            .insert(AlphaModeUniformOffset {
                offset: alpha_mode_meta
                    .uniforms
                    .push(AlphaModeUniform { mode, cutoff }),
            });
    }

    alpha_mode_meta
        .uniforms
        .write_to_staging_buffer(&render_device);
}

pub struct AlphaModeNode;

impl Node for AlphaModeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let alpha_mode_meta = world.get_resource::<AlphaModeMeta>().unwrap();
        alpha_mode_meta
            .uniforms
            .write_to_uniform_buffer(&mut render_context.command_encoder);
        Ok(())
    }
}
