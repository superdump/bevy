use bevy_ecs::{prelude::*, system::SystemState};
use bevy_render2::{
    camera::{ActiveCameras, CameraPlugin},
    mesh::Mesh,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_phase::{Draw, DrawFunctions, RenderPhase, TrackedRenderPass},
    render_resource::{BindGroup, BufferId, RenderPipeline},
    renderer::{RenderContext, RenderDevice},
    shader::Shader,
    view::{ExtractedView, ViewUniformOffset},
};
use bevy_utils::slab::{FrameSlabMap, FrameSlabMapKey};
use crevice::std140::AsStd140;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, BufferSize, CompareFunction, DepthBiasState, DepthStencilState, Face,
    FragmentState, FrontFace, IndexFormat, InputStepMode, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderModule,
    ShaderStage, StencilFaceState, StencilState, TextureFormat, TextureSampleType,
    TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
};

use crate::{ExtractedMeshes, MeshMeta, PbrShaders, StandardMaterialUniformData};

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
                        min_binding_size: BufferSize::new(
                            StandardMaterialUniformData::std140_size_static() as u64,
                        ),
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
                // AlphaMode
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        // TODO: change this to AlphaModeUniform::std140_size_static once crevice fixes this!
                        // Context: https://github.com/LPGhatguy/crevice/issues/29
                        min_binding_size: BufferSize::new(8),
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

#[derive(Default)]
pub struct DepthPrepassMeta {
    pub view_bind_group: Option<BindGroup>,
}

pub struct OpaqueDepthPhase;
pub struct AlphaMaskDepthPhase;

pub fn extract_depth_phases(mut commands: Commands, active_cameras: Res<ActiveCameras>) {
    if let Some(camera_3d) = active_cameras.get(CameraPlugin::CAMERA_3D) {
        if let Some(entity) = camera_3d.entity {
            commands.get_or_spawn(entity).insert_bundle((
                RenderPhase::<OpaqueDepthPhase>::default(),
                RenderPhase::<AlphaMaskDepthPhase>::default(),
            ));
        }
    }
}

pub struct DepthPrepassMeshDrawInfo {
    // TODO: compare cost of doing this vs cloning the BindGroup?
    pub material_bind_group_key: FrameSlabMapKey<BufferId, BindGroup>,
    pub alpha_mode_uniform_offset: u32,
}

#[derive(Default)]
pub struct DepthPrepassMeshMeta {
    pub material_bind_groups: FrameSlabMap<BufferId, BindGroup>,
    pub mesh_draw_info: Vec<DepthPrepassMeshDrawInfo>,
}

pub struct DepthPrepassNode {
    query: QueryState<
        (
            &'static RenderPhase<OpaqueDepthPhase>,
            &'static RenderPhase<AlphaMaskDepthPhase>,
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
        let draw_functions = world.get_resource::<DrawFunctions>().unwrap();

        let (opaque_depth_phase, alpha_mask_depth_phase) = self
            .query
            .get_manual(world, view_entity)
            .expect("view entity should exist");

        let mut draw_functions = draw_functions.write();

        {
            // Run the opaque pass, sorted front-to-back
            // NOTE: Scoped to drop the mutable borrow of render_context
            let opaque_depth_prepass_descriptor = RenderPassDescriptor {
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
            let opaque_depth_prepass = render_context
                .command_encoder
                .begin_render_pass(&opaque_depth_prepass_descriptor);
            let mut tracked_opaque_depth_prepass = TrackedRenderPass::new(opaque_depth_prepass);
            for drawable in opaque_depth_phase.drawn_things.iter() {
                let draw_function = draw_functions.get_mut(drawable.draw_function).unwrap();
                draw_function.draw(
                    world,
                    &mut tracked_opaque_depth_prepass,
                    view_entity,
                    drawable.draw_key,
                    drawable.sort_key,
                );
            }
        }

        {
            // Run the alpha_mask depth prepass, sorted front-to-back
            // NOTE: Scoped to drop the mutable borrow of render_context
            let alpha_mask_depth_prepass_descriptor = RenderPassDescriptor {
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
            let alpha_mask_depth_prepass = render_context
                .command_encoder
                .begin_render_pass(&alpha_mask_depth_prepass_descriptor);
            let mut tracked_alpha_mask_depth_prepass =
                TrackedRenderPass::new(alpha_mask_depth_prepass);
            for drawable in alpha_mask_depth_phase.drawn_things.iter() {
                let draw_function = draw_functions.get_mut(drawable.draw_function).unwrap();
                draw_function.draw(
                    world,
                    &mut tracked_alpha_mask_depth_prepass,
                    view_entity,
                    drawable.draw_key,
                    drawable.sort_key,
                );
            }
        }

        Ok(())
    }
}

type DrawDepthParams<'s, 'w> = (
    Res<'w, DepthPrepassShaders>,
    Res<'w, DepthPrepassMeta>,
    Res<'w, ExtractedMeshes>,
    Res<'w, DepthPrepassMeshMeta>,
    Res<'w, MeshMeta>,
    Res<'w, RenderAssets<Mesh>>,
    Query<'w, 's, &'w ViewUniformOffset>,
);
pub struct DrawOpaqueDepth {
    params: SystemState<DrawDepthParams<'static, 'static>>,
}

impl DrawOpaqueDepth {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

impl Draw for DrawOpaqueDepth {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        draw_key: usize,
        _sort_key: usize,
    ) {
        let (
            depth_prepass_shaders,
            depth_prepass_meta,
            extracted_meshes,
            depth_prepass_mesh_meta,
            mesh_meta,
            meshes,
            views,
        ) = self.params.get(world);

        let depth_prepass_shaders = depth_prepass_shaders.into_inner();
        pass.set_render_pipeline(&depth_prepass_shaders.opaque_prepass_pipeline);

        let view_uniforms = views.get(view).unwrap();
        let depth_prepass_meta = depth_prepass_meta.into_inner();
        pass.set_bind_group(
            0,
            depth_prepass_meta
                .view_bind_group
                .as_ref()
                .expect("No depth prepass view bind group"),
            &[view_uniforms.offset],
        );

        let depth_prepass_mesh_meta = depth_prepass_mesh_meta.into_inner();
        let depth_prepass_mesh_draw_info = &depth_prepass_mesh_meta.mesh_draw_info[draw_key];
        pass.set_bind_group(
            1,
            &depth_prepass_mesh_meta.material_bind_groups
                [depth_prepass_mesh_draw_info.material_bind_group_key],
            &[depth_prepass_mesh_draw_info.alpha_mode_uniform_offset],
        );

        let extracted_mesh = &extracted_meshes.meshes[draw_key];
        let mesh_meta = mesh_meta.into_inner();
        pass.set_bind_group(
            2,
            mesh_meta
                .mesh_transform_bind_group
                .get_value(mesh_meta.mesh_transform_bind_group_key.unwrap())
                .unwrap(),
            &[extracted_mesh.transform_binding_offset],
        );

        let gpu_mesh = meshes.into_inner().get(&extracted_mesh.mesh).unwrap();
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        if let Some(index_info) = &gpu_mesh.index_info {
            pass.set_index_buffer(index_info.buffer.slice(..), 0, IndexFormat::Uint32);
            pass.draw_indexed(0..index_info.count, 0, 0..1);
        } else {
            panic!("non-indexed drawing not supported yet")
        }
    }
}

pub struct DrawAlphaMaskDepth {
    params: SystemState<DrawDepthParams<'static, 'static>>,
}

impl DrawAlphaMaskDepth {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

impl Draw for DrawAlphaMaskDepth {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        draw_key: usize,
        _sort_key: usize,
    ) {
        let (
            depth_prepass_shaders,
            depth_prepass_meta,
            extracted_meshes,
            depth_prepass_mesh_meta,
            mesh_meta,
            meshes,
            views,
        ) = self.params.get(world);

        let depth_prepass_shaders = depth_prepass_shaders.into_inner();
        pass.set_render_pipeline(&depth_prepass_shaders.alpha_mask_prepass_pipeline);

        let view_uniforms = views.get(view).unwrap();
        let depth_prepass_meta = depth_prepass_meta.into_inner();
        pass.set_bind_group(
            0,
            depth_prepass_meta
                .view_bind_group
                .as_ref()
                .expect("No depth prepass view bind group"),
            &[view_uniforms.offset],
        );

        let depth_prepass_mesh_meta = depth_prepass_mesh_meta.into_inner();
        let depth_prepass_mesh_draw_info = &depth_prepass_mesh_meta.mesh_draw_info[draw_key];
        pass.set_bind_group(
            1,
            &depth_prepass_mesh_meta.material_bind_groups
                [depth_prepass_mesh_draw_info.material_bind_group_key],
            &[depth_prepass_mesh_draw_info.alpha_mode_uniform_offset],
        );

        let extracted_mesh = &extracted_meshes.meshes[draw_key];
        let mesh_meta = mesh_meta.into_inner();
        pass.set_bind_group(
            2,
            mesh_meta
                .mesh_transform_bind_group
                .get_value(mesh_meta.mesh_transform_bind_group_key.unwrap())
                .unwrap(),
            &[extracted_mesh.transform_binding_offset],
        );

        let gpu_mesh = meshes.into_inner().get(&extracted_mesh.mesh).unwrap();
        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        if let Some(index_info) = &gpu_mesh.index_info {
            pass.set_index_buffer(index_info.buffer.slice(..), 0, IndexFormat::Uint32);
            pass.draw_indexed(0..index_info.count, 0, 0..1);
        } else {
            panic!("non-indexed drawing not supported yet")
        }
    }
}
