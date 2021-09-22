use bevy_app::Plugin;
use bevy_asset::Handle;
use bevy_core::FloatOrd;
use bevy_ecs::{
    prelude::*,
    system::{
        lifetimeless::{Read, SQuery, SRes},
        SystemParamItem,
    },
};
use bevy_render2::{
    camera::{ActiveCameras, CameraPlugin},
    mesh::Mesh,
    render_asset::RenderAssets,
    render_component::DynamicUniformIndex,
    render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType},
    render_phase::{
        sort_phase_system, AddRenderCommand, DrawFunctionId, DrawFunctions, EntityPhaseItem,
        PhaseItem, RenderCommand, RenderPhase, TrackedRenderPass,
    },
    render_resource::{BindGroup, RenderPipeline},
    renderer::{RenderContext, RenderDevice},
    shader::Shader,
    texture::Image,
    view::{ExtractedView, ViewUniformOffset, ViewUniforms, VisibleEntities},
    RenderApp, RenderStage,
};
use bevy_utils::HashMap;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferSize,
    CompareFunction, DepthBiasState, DepthStencilState, Face, FragmentState, FrontFace,
    IndexFormat, InputStepMode, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor,
    PolygonMode, PrimitiveState, PrimitiveTopology, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, ShaderModule, ShaderStage, StencilFaceState,
    StencilState, TextureFormat, TextureSampleType, TextureViewDimension, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexState,
};

use crate::{
    draw_3d_graph, image_handle_to_view_sampler, AlphaMode, MeshUniform, PbrShaders,
    StandardMaterial, TransformBindGroup,
};

pub struct DepthPrepassShaders {
    pub shader_module: ShaderModule,
    pub opaque_prepass_pipeline: RenderPipeline,
    pub alpha_mask_prepass_pipeline: RenderPipeline,
    pub view_layout: BindGroupLayout,
    pub material_layout: BindGroupLayout,
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for DepthPrepassShaders {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let pbr_shaders = world.get_resource::<PbrShaders>().unwrap();
        let shader = Shader::from_wgsl(include_str!("depth_prepass.wgsl"));
        let shader_module = render_device.create_shader_module(&shader);

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        // TODO: change this to ViewUniform::std140_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(144),
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let material_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // TODO: change this to StandardMaterialUniformData::std140_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(64),
                    },
                    count: None,
                },
                // Base Color Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Base Color Texture Sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let opaque_prepass_pipeline_layout =
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                push_constant_ranges: &[],
                bind_group_layouts: &[&view_layout, &material_layout, &pbr_shaders.mesh_layout],
            });

        let opaque_prepass_pipeline =
            render_device.create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                vertex: VertexState {
                    buffers: &[VertexBufferLayout {
                        array_stride: 32,
                        step_mode: InputStepMode::Vertex,
                        attributes: &[
                            // Position (GOTCHA! Vertex_Position isn't first in the buffer due to how Mesh sorts attributes (alphabetically))
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 0,
                            },
                        ],
                    }],
                    module: &shader_module,
                    entry_point: "vertex_opaque",
                },
                fragment: None,
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Greater,
                    stencil: StencilState {
                        front: StencilFaceState::IGNORE,
                        back: StencilFaceState::IGNORE,
                        read_mask: 0,
                        write_mask: 0,
                    },
                    bias: DepthBiasState {
                        constant: 0,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                layout: Some(&opaque_prepass_pipeline_layout),
                multisample: MultisampleState::default(),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(Face::Back),
                    polygon_mode: PolygonMode::Fill,
                    clamp_depth: false,
                    conservative: false,
                },
            });

        let alpha_mask_prepass_pipeline_layout =
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                push_constant_ranges: &[],
                bind_group_layouts: &[&view_layout, &material_layout, &pbr_shaders.mesh_layout],
            });

        let alpha_mask_prepass_pipeline =
            render_device.create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                vertex: VertexState {
                    buffers: &[VertexBufferLayout {
                        array_stride: 32,
                        step_mode: InputStepMode::Vertex,
                        attributes: &[
                            // Position (GOTCHA! Vertex_Position isn't first in the buffer due to how Mesh sorts attributes (alphabetically))
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 0,
                            },
                            // Uv
                            VertexAttribute {
                                format: VertexFormat::Float32x2,
                                offset: 24,
                                shader_location: 1,
                            },
                        ],
                    }],
                    module: &shader_module,
                    entry_point: "vertex_alpha_mask",
                },
                fragment: Some(FragmentState {
                    module: &shader_module,
                    entry_point: "fragment_alpha_mask",
                    targets: &[],
                }),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Greater,
                    stencil: StencilState {
                        front: StencilFaceState::IGNORE,
                        back: StencilFaceState::IGNORE,
                        read_mask: 0,
                        write_mask: 0,
                    },
                    bias: DepthBiasState {
                        constant: 0,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                layout: Some(&alpha_mask_prepass_pipeline_layout),
                multisample: MultisampleState::default(),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(Face::Back),
                    polygon_mode: PolygonMode::Fill,
                    clamp_depth: false,
                    conservative: false,
                },
            });

        DepthPrepassShaders {
            shader_module,
            opaque_prepass_pipeline,
            alpha_mask_prepass_pipeline,
            view_layout,
            material_layout,
        }
    }
}

pub struct OpaqueDepth3d {
    pub distance: f32,
    pub entity: Entity,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for OpaqueDepth3d {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for OpaqueDepth3d {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }
}

pub struct AlphaMaskDepth3d {
    pub distance: f32,
    pub entity: Entity,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for AlphaMaskDepth3d {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for AlphaMaskDepth3d {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum Systems {
    ExtractDepthPhases,
    QueueDepthPrepassMeshes,
}

pub struct DepthPrepassPlugin;
impl Plugin for DepthPrepassPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        let render_app = app.sub_app(RenderApp);
        render_app
            .add_system_to_stage(
                RenderStage::Extract,
                extract_depth_phases.label(Systems::ExtractDepthPhases),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_depth_prepass_meshes.label(Systems::QueueDepthPrepassMeshes),
            )
            .add_system_to_stage(RenderStage::PhaseSort, sort_phase_system::<OpaqueDepth3d>)
            .add_system_to_stage(
                RenderStage::PhaseSort,
                sort_phase_system::<AlphaMaskDepth3d>,
            )
            .init_resource::<DepthPrepassMaterialMeta>()
            .init_resource::<DepthPrepassShaders>()
            .init_resource::<DrawFunctions<OpaqueDepth3d>>()
            .init_resource::<DrawFunctions<AlphaMaskDepth3d>>();

        let depth_prepass_node = DepthPrepassNode::new(&mut render_app.world);
        render_app.add_render_command::<OpaqueDepth3d, DrawDepth>();
        render_app.add_render_command::<AlphaMaskDepth3d, DrawDepth>();
        let render_world = render_app.world.cell();
        let mut graph = render_world.get_resource_mut::<RenderGraph>().unwrap();
        let draw_3d_graph = graph
            .get_sub_graph_mut(bevy_core_pipeline::draw_3d_graph::NAME)
            .unwrap();

        draw_3d_graph.add_node(draw_3d_graph::node::DEPTH_PREPASS, depth_prepass_node);
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::DEPTH_PREPASS,
                bevy_core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                bevy_core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::DEPTH_PREPASS,
                DepthPrepassNode::IN_VIEW,
            )
            .unwrap();
        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                bevy_core_pipeline::draw_3d_graph::input::DEPTH,
                draw_3d_graph::node::DEPTH_PREPASS,
                DepthPrepassNode::IN_DEPTH,
            )
            .unwrap();
    }
}

pub fn extract_depth_phases(mut commands: Commands, active_cameras: Res<ActiveCameras>) {
    if let Some(camera_3d) = active_cameras.get(CameraPlugin::CAMERA_3D) {
        if let Some(entity) = camera_3d.entity {
            commands.get_or_spawn(entity).insert_bundle((
                RenderPhase::<OpaqueDepth3d>::default(),
                RenderPhase::<AlphaMaskDepth3d>::default(),
            ));
        }
    }
}

pub struct DepthPrepassViewBindGroup {
    pub value: BindGroup,
}

pub type DepthPrepassMaterialMeta = HashMap<Handle<StandardMaterial>, BindGroup>;

pub fn queue_depth_prepass_meshes(
    mut commands: Commands,
    opaque_depth_draw_functions: Res<DrawFunctions<OpaqueDepth3d>>,
    alpha_mask_depth_draw_functions: Res<DrawFunctions<AlphaMaskDepth3d>>,
    render_device: Res<RenderDevice>,
    view_uniforms: Res<ViewUniforms>,
    mut depth_prepass_material_meta: ResMut<DepthPrepassMaterialMeta>,
    gpu_images: Res<RenderAssets<Image>>,
    render_materials: Res<RenderAssets<StandardMaterial>>,
    standard_material_meshes: Query<(&Handle<StandardMaterial>, &MeshUniform), With<Handle<Mesh>>>,
    depth_prepass_shaders: Res<DepthPrepassShaders>,
    pbr_shaders: Res<PbrShaders>,
    mut views: Query<(
        Entity,
        &ExtractedView,
        &VisibleEntities,
        &mut RenderPhase<OpaqueDepth3d>,
        &mut RenderPhase<AlphaMaskDepth3d>,
    )>,
) {
    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        for (entity, view, visible_entities, mut opaque_depth_phase, mut alpha_mask_depth_phase) in
            views.iter_mut()
        {
            let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: view_binding.clone(),
                }],
                label: None,
                layout: &depth_prepass_shaders.view_layout,
            });

            commands.entity(entity).insert(DepthPrepassViewBindGroup {
                value: view_bind_group,
            });

            let (draw_opaque_depth, draw_alpha_mask_depth) = (
                opaque_depth_draw_functions
                    .read()
                    .get_id::<DrawDepth>()
                    .unwrap(),
                alpha_mask_depth_draw_functions
                    .read()
                    .get_id::<DrawDepth>()
                    .unwrap(),
            );

            let inverse_view_matrix = view.transform.compute_matrix().inverse();
            let inverse_view_row_2 = inverse_view_matrix.row(2);

            for visible_entity in &visible_entities.entities {
                if let Ok((material_handle, mesh_uniform)) =
                    standard_material_meshes.get(visible_entity.entity)
                {
                    if let Some(material) = render_materials.get(material_handle) {
                        if !depth_prepass_material_meta.contains_key(material_handle) {
                            if let Some((base_color_texture_view, base_color_sampler)) =
                                image_handle_to_view_sampler(
                                    &*pbr_shaders,
                                    &*gpu_images,
                                    &material.base_color_texture,
                                )
                            {
                                let bind_group =
                                    render_device.create_bind_group(&BindGroupDescriptor {
                                        entries: &[
                                            BindGroupEntry {
                                                binding: 0,
                                                resource: material.buffer.as_entire_binding(),
                                            },
                                            BindGroupEntry {
                                                binding: 1,
                                                resource: BindingResource::TextureView(
                                                    base_color_texture_view,
                                                ),
                                            },
                                            BindGroupEntry {
                                                binding: 2,
                                                resource: BindingResource::Sampler(
                                                    base_color_sampler,
                                                ),
                                            },
                                        ],
                                        label: None,
                                        layout: &depth_prepass_shaders.material_layout,
                                    });
                                depth_prepass_material_meta
                                    .insert(material_handle.clone(), bind_group);
                            }
                        }

                        // NOTE: row 2 of the inverse view matrix dotted with column 3 of the model matrix
                        //       gives the z component of translation of the mesh in view space
                        let mesh_z = inverse_view_row_2.dot(mesh_uniform.transform.col(3));

                        // NOTE: Front-to-back ordering for opaque and alpha mask with ascending sort means near should have the
                        //       lowest sort key and getting further away should increase. As we have
                        //       -z in front fo the camera, values in view space decrease away from the
                        //       camera. Flipping the sign of mesh_z results in the correct front-to-back ordering
                        let distance = -mesh_z;
                        match material.alpha_mode {
                            AlphaMode::Opaque => {
                                opaque_depth_phase.add(OpaqueDepth3d {
                                    distance,
                                    entity: visible_entity.entity,
                                    draw_function: draw_opaque_depth,
                                });
                            }
                            AlphaMode::Mask(_) => {
                                alpha_mask_depth_phase.add(AlphaMaskDepth3d {
                                    distance,
                                    entity: visible_entity.entity,
                                    draw_function: draw_alpha_mask_depth,
                                });
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

pub struct DepthPrepassNode {
    query: QueryState<
        (
            &'static RenderPhase<OpaqueDepth3d>,
            &'static RenderPhase<AlphaMaskDepth3d>,
        ),
        With<ExtractedView>,
    >,
}

impl DepthPrepassNode {
    pub const IN_DEPTH: &'static str = "depth";
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for DepthPrepassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![
            SlotInfo::new(DepthPrepassNode::IN_DEPTH, SlotType::TextureView),
            SlotInfo::new(DepthPrepassNode::IN_VIEW, SlotType::Entity),
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
        let depth_texture = graph.get_input_texture(Self::IN_DEPTH)?;

        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;

        let (opaque_depth_phase, alpha_mask_depth_phase) = self
            .query
            .get_manual(world, view_entity)
            .expect("view entity should exist");

        {
            // Run the opaque pass, sorted front-to-back
            // NOTE: Scoped to drop the mutable borrow of render_context

            let draw_functions = world
                .get_resource::<DrawFunctions<OpaqueDepth3d>>()
                .unwrap();
            let mut draw_functions = draw_functions.write();

            let pass_descriptor = RenderPassDescriptor {
                label: Some("opaque_depth_prepass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_texture,
                    // NOTE: The opaque depth prepass clears and writes to the depth buffer.
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };
            let pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut tracked_pass = TrackedRenderPass::new(pass);
            for item in opaque_depth_phase.items.iter() {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        {
            // Run the alpha_mask depth prepass, sorted front-to-back
            // NOTE: Scoped to drop the mutable borrow of render_context

            let draw_functions = world
                .get_resource::<DrawFunctions<AlphaMaskDepth3d>>()
                .unwrap();
            let mut draw_functions = draw_functions.write();

            let pass_descriptor = RenderPassDescriptor {
                label: Some("alpha_mask_depth_prepass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_texture,
                    // NOTE: The alpha_mask pass loads and writes to the depth buffer.
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };
            let pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut tracked_pass = TrackedRenderPass::new(pass);
            for item in alpha_mask_depth_phase.items.iter() {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        Ok(())
    }
}

pub type DrawDepth = (
    SetDepthPrepassPipeline,
    SetMeshViewBindGroup<0>,
    SetDepthPrepassMaterialBindGroup<1>,
    SetTransformBindGroup<2>,
    DrawMesh,
);

pub struct SetDepthPrepassPipeline;
impl RenderCommand<OpaqueDepth3d> for SetDepthPrepassPipeline {
    type Param = SRes<DepthPrepassShaders>;
    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: &OpaqueDepth3d,
        depth_prepass_shaders: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        pass.set_render_pipeline(&depth_prepass_shaders.into_inner().opaque_prepass_pipeline);
    }
}
impl RenderCommand<AlphaMaskDepth3d> for SetDepthPrepassPipeline {
    type Param = SRes<DepthPrepassShaders>;
    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: &AlphaMaskDepth3d,
        depth_prepass_shaders: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        pass.set_render_pipeline(
            &depth_prepass_shaders
                .into_inner()
                .alpha_mask_prepass_pipeline,
        );
    }
}

pub struct SetMeshViewBindGroup<const I: usize>;
impl<T: PhaseItem, const I: usize> RenderCommand<T> for SetMeshViewBindGroup<I> {
    type Param = SQuery<(Read<ViewUniformOffset>, Read<DepthPrepassViewBindGroup>)>;
    #[inline]
    fn render<'w>(
        view: Entity,
        _item: &T,
        view_query: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let (view_uniform, depth_prepass_view_bind_group) = view_query.get(view).unwrap();
        pass.set_bind_group(
            I,
            &depth_prepass_view_bind_group.value,
            &[view_uniform.offset],
        );
    }
}

pub struct SetTransformBindGroup<const I: usize>;
impl<T: EntityPhaseItem + PhaseItem, const I: usize> RenderCommand<T> for SetTransformBindGroup<I> {
    type Param = (
        SRes<TransformBindGroup>,
        SQuery<Read<DynamicUniformIndex<MeshUniform>>>,
    );
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: &T,
        (transform_bind_group, mesh_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let transform_index = mesh_query.get(item.entity()).unwrap();
        pass.set_bind_group(
            I,
            &transform_bind_group.into_inner().value,
            &[transform_index.index()],
        );
    }
}

pub struct SetDepthPrepassMaterialBindGroup<const I: usize>;
impl<T: EntityPhaseItem + PhaseItem, const I: usize> RenderCommand<T>
    for SetDepthPrepassMaterialBindGroup<I>
{
    type Param = (
        SRes<DepthPrepassMaterialMeta>,
        SQuery<Read<Handle<StandardMaterial>>>,
    );
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: &T,
        (material_bind_groups, handle_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let handle = handle_query.get(item.entity()).unwrap();
        let material_bind_groups = material_bind_groups.into_inner();
        let material_bind_group = material_bind_groups.get(handle).unwrap();
        pass.set_bind_group(I, material_bind_group, &[]);
    }
}

pub struct DrawMesh;
impl<T: EntityPhaseItem + PhaseItem> RenderCommand<T> for DrawMesh {
    type Param = (SRes<RenderAssets<Mesh>>, SQuery<Read<Handle<Mesh>>>);
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: &T,
        (meshes, mesh_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) {
        let mesh_handle = mesh_query.get(item.entity()).unwrap();
        let gpu_mesh = meshes.into_inner().get(mesh_handle).unwrap();
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        if let Some(index_info) = &gpu_mesh.index_info {
            pass.set_index_buffer(index_info.buffer.slice(..), 0, IndexFormat::Uint32);
            pass.draw_indexed(0..index_info.count, 0, 0..1);
        } else {
            panic!("non-indexed drawing not supported yet")
        }
    }
}
