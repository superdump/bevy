use std::sync::Mutex;

use bevy_ecs::prelude::{FromWorld, World};
use bevy_render2::{
    camera::{CameraPlugin, ExtractedCamera, ExtractedCameraNames},
    render_graph::{Node, RenderGraphContext, SlotInfo, SlotType},
    render_resource::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent, BlendFactor,
        BlendOperation, BlendState, ColorTargetState, ColorWrites, Face, FragmentState, LoadOp,
        Operations, PipelineLayoutDescriptor, PrimitiveState, RenderPassColorAttachment,
        RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler, ShaderModule,
        ShaderStages, TextureFormat, TextureSampleType, TextureViewDimension, TextureViewId,
        VertexState,
    },
    renderer::{RenderContext, RenderDevice},
    shader::Shader,
    texture::BevyDefault,
    view::ExtractedWindows,
};
use wgpu::Color;

pub struct ToneMappingShaders {
    pub pipeline: RenderPipeline,
    pub shader_module: ShaderModule,
    pub hdr_target_layout: BindGroupLayout,
    pub hdr_target_sampler: Sampler,
}

impl FromWorld for ToneMappingShaders {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let shader = Shader::from_wgsl(include_str!("tone_mapping.wgsl"));
        let shader_module = render_device.create_shader_module(&shader);

        let hdr_target_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("hdr_target_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        ty: BindingType::Sampler {
                            filtering: false,
                            comparison: false,
                        },
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        count: None,
                    },
                ],
            });

        let layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("tone_mapping_pipeline_layout"),
            bind_group_layouts: &[&hdr_target_layout],
            push_constant_ranges: &[],
        });

        let pipeline = render_device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("tone_mapping_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "main",
                targets: &[ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent::REPLACE,
                    }),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            multisample: Default::default(),
            depth_stencil: None,
        });

        let hdr_target_sampler = render_device.create_sampler(&Default::default());

        Self {
            pipeline,
            shader_module,
            hdr_target_layout,
            hdr_target_sampler,
        }
    }
}

#[derive(Default)]
pub struct ToneMappingNode {
    hdr_target_id: Mutex<Option<TextureViewId>>,
    hdr_target_bind_group: Mutex<Option<BindGroup>>,
}

impl ToneMappingNode {
    pub const HDR_TARGET: &'static str = "hdr_target";
}

impl Node for ToneMappingNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::HDR_TARGET, SlotType::TextureView)]
    }

    fn update(&mut self, world: &mut World) {
        if !world.contains_resource::<ToneMappingShaders>() {
            let tone_mapping_shaders = ToneMappingShaders::from_world(world);
            world.insert_resource(tone_mapping_shaders);
        }
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy_render2::render_graph::NodeRunError> {
        let tone_mapping_shaders = world.get_resource::<ToneMappingShaders>().unwrap();
        let extracted_cameras = world.get_resource::<ExtractedCameraNames>().unwrap();
        let extracted_windows = world.get_resource::<ExtractedWindows>().unwrap();

        if let Some(camera_3d) = extracted_cameras.entities.get(CameraPlugin::CAMERA_3D) {
            let extracted_camera = world.entity(*camera_3d).get::<ExtractedCamera>().unwrap();
            let extracted_window = extracted_windows.get(&extracted_camera.window_id).unwrap();
            let swap_chain_texture = extracted_window.swap_chain_texture.as_ref().unwrap();

            let hdr_target = graph.get_input_texture(Self::HDR_TARGET)?;

            let mut hdr_target_id = self.hdr_target_id.lock().unwrap();
            let mut hdr_target_bind_group = self.hdr_target_bind_group.lock().unwrap();

            if *hdr_target_id != Some(hdr_target.id()) {
                *hdr_target_id = Some(hdr_target.id());
                *hdr_target_bind_group = Some(render_context.render_device.create_bind_group(
                    &BindGroupDescriptor {
                        label: Some("tone_mapping_hdr_target"),
                        layout: &tone_mapping_shaders.hdr_target_layout,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::TextureView(hdr_target),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Sampler(
                                    &tone_mapping_shaders.hdr_target_sampler,
                                ),
                            },
                        ],
                    },
                ));
            }

            let load_op = if extracted_cameras
                .entities
                .contains_key(CameraPlugin::CAMERA_2D)
            {
                LoadOp::Load
            } else {
                LoadOp::Clear(Color::TRANSPARENT)
            };

            let mut render_pass =
                render_context
                    .command_encoder
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("tone_mapping_pass"),
                        color_attachments: &[RenderPassColorAttachment {
                            view: &swap_chain_texture,
                            resolve_target: None,
                            ops: Operations {
                                load: load_op,
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });

            render_pass.set_pipeline(&tone_mapping_shaders.pipeline);

            render_pass.set_bind_group(0, hdr_target_bind_group.as_ref().unwrap(), &[]);

            render_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
