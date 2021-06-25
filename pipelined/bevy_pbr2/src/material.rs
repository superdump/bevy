use bevy_app::{App, CoreStage, EventReader, Plugin};
use bevy_asset::{AddAsset, AssetEvent, Assets, Handle};
use bevy_ecs::prelude::*;
use bevy_math::Vec4;
use bevy_reflect::TypeUuid;
use bevy_render2::{
    color::Color,
    render_resource::{Buffer, BufferInitDescriptor, BufferUsage},
    renderer::{RenderDevice, RenderQueue},
    texture::Image,
};
use bevy_utils::HashSet;
use crevice::std140::{AsStd140, Std140};

// TODO: this shouldn't live in the StandardMaterial type
#[derive(Debug, Clone)]
pub struct StandardMaterialGpuData {
    pub buffer: Buffer,
}

// NOTE: These must match the bit flags in bevy_pbr2/src/render/pbr.frag!
bitflags::bitflags! {
    #[repr(transparent)]
    struct StandardMaterialFlags: u32 {
        const BASE_COLOR_TEXTURE         = (1 << 0);
        const EMISSIVE_TEXTURE           = (1 << 1);
        const METALLIC_ROUGHNESS_TEXTURE = (1 << 2);
        const OCCLUSION_TEXTURE          = (1 << 3);
        const DOUBLE_SIDED               = (1 << 4);
        const UNLIT                      = (1 << 5);
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

/// A material with "standard" properties used in PBR lighting
/// Standard property values with pictures here https://google.github.io/filament/Material%20Properties.pdf
#[derive(Debug, Clone, TypeUuid)]
#[uuid = "7494888b-c082-457b-aacf-517228cc0c22"]
pub struct StandardMaterial {
    /// Doubles as diffuse albedo for non-metallic, specular for metallic and a mix for everything
    /// in between If used together with a base_color_texture, this is factored into the final
    /// base color as `base_color * base_color_texture_value`
    pub base_color: Color,
    pub base_color_texture: Option<Handle<Image>>,
    // Use a color for user friendliness even though we technically don't use the alpha channel
    // Might be used in the future for exposure correction in HDR
    pub emissive: Color,
    pub emissive_texture: Option<Handle<Image>>,
    /// Linear perceptual roughness, clamped to [0.089, 1.0] in the shader
    /// Defaults to minimum of 0.089
    /// If used together with a roughness/metallic texture, this is factored into the final base
    /// color as `roughness * roughness_texture_value`
    pub perceptual_roughness: f32,
    /// From [0.0, 1.0], dielectric to pure metallic
    /// If used together with a roughness/metallic texture, this is factored into the final base
    /// color as `metallic * metallic_texture_value`
    pub metallic: f32,
    pub metallic_roughness_texture: Option<Handle<Image>>,
    /// Specular intensity for non-metals on a linear scale of [0.0, 1.0]
    /// defaults to 0.5 which is mapped to 4% reflectance in the shader
    pub reflectance: f32,
    pub occlusion_texture: Option<Handle<Image>>,
    pub double_sided: bool,
    pub unlit: bool,
    pub gpu_data: Option<StandardMaterialGpuData>,
}

impl StandardMaterial {
    pub fn gpu_data(&self) -> Option<&StandardMaterialGpuData> {
        self.gpu_data.as_ref()
    }
}

impl Default for StandardMaterial {
    fn default() -> Self {
        StandardMaterial {
            base_color: Color::rgb(1.0, 1.0, 1.0),
            base_color_texture: None,
            emissive: Color::BLACK,
            emissive_texture: None,
            // This is the minimum the roughness is clamped to in shader code
            // See https://google.github.io/filament/Filament.html#materialsystem/parameterization/
            // It's the minimum floating point value that won't be rounded down to 0 in the
            // calculations used. Although technically for 32-bit floats, 0.045 could be
            // used.
            perceptual_roughness: 0.089,
            // Few materials are purely dielectric or metallic
            // This is just a default for mostly-dielectric
            metallic: 0.01,
            metallic_roughness_texture: None,
            // Minimum real-world reflectance is 2%, most materials between 2-5%
            // Expressed in a linear scale and equivalent to 4% reflectance see https://google.github.io/filament/Material%20Properties.pdf
            reflectance: 0.5,
            occlusion_texture: None,
            double_sided: false,
            unlit: false,
            gpu_data: None,
        }
    }
}

impl From<Color> for StandardMaterial {
    fn from(color: Color) -> Self {
        StandardMaterial {
            base_color: color,
            ..Default::default()
        }
    }
}

impl From<Handle<Image>> for StandardMaterial {
    fn from(texture: Handle<Image>) -> Self {
        StandardMaterial {
            base_color_texture: Some(texture),
            ..Default::default()
        }
    }
}

#[derive(Clone, AsStd140)]
pub struct StandardMaterialUniformData {
    /// Doubles as diffuse albedo for non-metallic, specular for metallic and a mix for everything
    /// in between.
    pub base_color: Vec4,
    // Use a color for user friendliness even though we technically don't use the alpha channel
    // Might be used in the future for exposure correction in HDR
    pub emissive: Vec4,
    /// Linear perceptual roughness, clamped to [0.089, 1.0] in the shader
    /// Defaults to minimum of 0.089
    pub roughness: f32,
    /// From [0.0, 1.0], dielectric to pure metallic
    pub metallic: f32,
    /// Specular intensity for non-metals on a linear scale of [0.0, 1.0]
    /// defaults to 0.5 which is mapped to 4% reflectance in the shader
    pub reflectance: f32,
    pub flags: u32,
}

pub struct StandardMaterialPlugin;

impl Plugin for StandardMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<StandardMaterial>().add_system_to_stage(
            CoreStage::PostUpdate,
            standard_material_resource_system.system(),
        );
    }
}

pub fn standard_material_resource_system(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut material_events: EventReader<AssetEvent<StandardMaterial>>,
) {
    let mut changed_materials = HashSet::default();
    for event in material_events.iter() {
        match event {
            AssetEvent::Created { ref handle } => {
                changed_materials.insert(handle.clone_weak());
            }
            AssetEvent::Modified { ref handle } => {
                changed_materials.insert(handle.clone_weak());
                // TODO: uncomment this to support mutated materials
                // remove_current_material_resources(render_resource_context, handle, &mut materials);
            }
            AssetEvent::Removed { ref handle } => {
                // if material was modified and removed in the same update, ignore the modification
                // events are ordered so future modification events are ok
                changed_materials.remove(handle);
            }
        }
    }

    // update changed material data
    for changed_material_handle in changed_materials.iter() {
        if let Some(material) = materials.get_mut(changed_material_handle) {
            // TODO: this avoids creating new materials each frame because storing gpu data in the material flags it as
            // modified. this prevents hot reloading and therefore can't be used in an actual impl.
            if material.gpu_data.is_some() {
                continue;
            }

            let mut flags = StandardMaterialFlags::NONE;
            if material.base_color_texture.is_some() {
                flags |= StandardMaterialFlags::BASE_COLOR_TEXTURE;
            }
            if material.emissive_texture.is_some() {
                flags |= StandardMaterialFlags::EMISSIVE_TEXTURE;
            }
            if material.metallic_roughness_texture.is_some() {
                flags |= StandardMaterialFlags::METALLIC_ROUGHNESS_TEXTURE;
            }
            if material.occlusion_texture.is_some() {
                flags |= StandardMaterialFlags::OCCLUSION_TEXTURE;
            }
            if material.double_sided {
                flags |= StandardMaterialFlags::DOUBLE_SIDED;
            }
            if material.unlit {
                flags |= StandardMaterialFlags::UNLIT;
            }
            let value = StandardMaterialUniformData {
                base_color: material.base_color.as_rgba_linear().into(),
                emissive: material.emissive.into(),
                roughness: material.perceptual_roughness,
                metallic: material.metallic,
                reflectance: material.reflectance,
                flags: flags.bits,
            };
            let value_std140 = value.as_std140();

            let size = StandardMaterialUniformData::std140_size_static();

            let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: None,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                contents: value_std140.as_bytes(),
            });

            material.gpu_data = Some(StandardMaterialGpuData { buffer });
        }
    }
}
