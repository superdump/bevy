use crate::{
    camera::{ActiveCameras, Camera},
    render_graph::{CommandQueue, Node, ResourceSlots, SystemNode},
    renderer::{
        BufferId, BufferInfo, BufferMapMode, BufferUsage, RenderContext, RenderResourceBinding,
        RenderResourceContext,
    },
};
use bevy_core::bytes_of;
use bevy_ecs::{
    system::{BoxedSystem, IntoSystem, Local, Query, Res, ResMut},
    world::World,
};
use bevy_math::{Mat3, Mat4};
use bevy_transform::prelude::*;
use std::borrow::Cow;

#[derive(Debug)]
pub struct CameraNode {
    command_queue: CommandQueue,
    camera_name: Cow<'static, str>,
}

impl CameraNode {
    pub fn new<T>(camera_name: T) -> Self
    where
        T: Into<Cow<'static, str>>,
    {
        CameraNode {
            command_queue: Default::default(),
            camera_name: camera_name.into(),
        }
    }
}

impl Node for CameraNode {
    fn update(
        &mut self,
        _world: &World,
        render_context: &mut dyn RenderContext,
        _input: &ResourceSlots,
        _output: &mut ResourceSlots,
    ) {
        self.command_queue.execute(render_context);
    }
}

impl SystemNode for CameraNode {
    fn get_system(&self) -> BoxedSystem {
        let system = camera_node_system.system().config(|config| {
            config.0 = Some(CameraNodeState {
                camera_name: self.camera_name.clone(),
                command_queue: self.command_queue.clone(),
                staging_buffer: None,
            })
        });
        Box::new(system)
    }
}

const CAMERA_VIEW_PROJ: &str = "CameraViewProj";
const CAMERA_PROJ: &str = "CameraProj";
const CAMERA_PROJ_INV: &str = "CameraProjInv";
const CAMERA_VIEW: &str = "CameraView";
const CAMERA_VIEW_INV_3: &str = "CameraViewInv3";
const CAMERA_MATRICES: [&str; 5] = [
    CAMERA_VIEW_PROJ,
    CAMERA_PROJ,
    CAMERA_PROJ_INV,
    CAMERA_VIEW,
    CAMERA_VIEW_INV_3,
];
const CAMERA_POSITION: &str = "CameraPosition";

#[derive(Debug, Default)]
pub struct CameraNodeState {
    command_queue: CommandQueue,
    camera_name: Cow<'static, str>,
    staging_buffer: Option<BufferId>,
}

const MATRIX_SIZE: usize = std::mem::size_of::<[[f32; 4]; 4]>();
const VEC4_SIZE: usize = std::mem::size_of::<[f32; 4]>();

pub fn camera_node_system(
    mut state: Local<CameraNodeState>,
    mut active_cameras: ResMut<ActiveCameras>,
    render_resource_context: Res<Box<dyn RenderResourceContext>>,
    mut query: Query<(&Camera, &GlobalTransform)>,
) {
    let render_resource_context = &**render_resource_context;

    let ((camera, global_transform), bindings) =
        if let Some(active_camera) = active_cameras.get_mut(&state.camera_name) {
            if let Some(entity) = active_camera.entity {
                (query.get_mut(entity).unwrap(), &mut active_camera.bindings)
            } else {
                return;
            }
        } else {
            return;
        };

    let staging_buffer = if let Some(staging_buffer) = state.staging_buffer {
        render_resource_context.map_buffer(staging_buffer, BufferMapMode::Write);
        staging_buffer
    } else {
        let staging_buffer = render_resource_context.create_buffer(BufferInfo {
            size: MATRIX_SIZE * CAMERA_MATRICES.len() + VEC4_SIZE,
            buffer_usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            mapped_at_creation: true,
        });

        state.staging_buffer = Some(staging_buffer);
        staging_buffer
    };

    for matrix_name in CAMERA_MATRICES {
        if bindings.get(matrix_name).is_none() {
            let buffer = render_resource_context.create_buffer(BufferInfo {
                size: MATRIX_SIZE,
                buffer_usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
                ..Default::default()
            });
            bindings.set(
                matrix_name,
                RenderResourceBinding::Buffer {
                    buffer,
                    range: 0..MATRIX_SIZE as u64,
                    dynamic_index: None,
                },
            );
        }
    }

    let view = global_transform.compute_matrix();
    // NOTE: These MUST be in the same order as CAMERA_MATRICES
    let matrices = [
        // CAMERA_VIEW_PROJ
        camera.projection_matrix * view.inverse(),
        // CAMERA_PROJ
        camera.projection_matrix,
        // CAMERA_PROJ_INV
        camera.projection_matrix.inverse(),
        // CAMERA_VIEW
        view,
        // CAMERA_VIEW_INV_3
        {
            let v = view.to_cols_array();
            let view_inv_3 = Mat3::from_cols_array_2d(&[
                [v[0], v[1], v[2]],
                [v[4], v[5], v[6]],
                [v[8], v[9], v[10]],
            ])
            .inverse()
            .to_cols_array();
            Mat4::from_cols_array_2d(&[
                [view_inv_3[0], view_inv_3[1], view_inv_3[2], 0.0],
                [view_inv_3[3], view_inv_3[4], view_inv_3[5], 0.0],
                [view_inv_3[6], view_inv_3[7], view_inv_3[8], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        },
    ];
    let mut offset = 0;
    for (matrix_name, matrix) in CAMERA_MATRICES.iter().zip(matrices.iter()) {
        if let Some(RenderResourceBinding::Buffer { buffer, .. }) = bindings.get(*matrix_name) {
            render_resource_context.write_mapped_buffer(
                staging_buffer,
                offset..(offset + MATRIX_SIZE as u64),
                &mut |data, _renderer| {
                    data[0..MATRIX_SIZE].copy_from_slice(bytes_of(matrix));
                },
            );
            state.command_queue.copy_buffer_to_buffer(
                staging_buffer,
                offset,
                *buffer,
                0,
                MATRIX_SIZE as u64,
            );
            offset += MATRIX_SIZE as u64;
        }
    }

    if let Some(RenderResourceBinding::Buffer { buffer, .. }) = bindings.get(CAMERA_POSITION) {
        let position: [f32; 3] = global_transform.translation.into();
        let position: [f32; 4] = [position[0], position[1], position[2], 0.0];
        render_resource_context.write_mapped_buffer(
            staging_buffer,
            offset..(offset + VEC4_SIZE as u64),
            &mut |data, _renderer| {
                data[0..VEC4_SIZE].copy_from_slice(bytes_of(&position));
            },
        );
        state.command_queue.copy_buffer_to_buffer(
            staging_buffer,
            offset,
            *buffer,
            0,
            VEC4_SIZE as u64,
        );
    }

    render_resource_context.unmap_buffer(staging_buffer);
}
