mod hdr_texture_node;

pub use hdr_texture_node::*;

use bevy_render2::render_resource::TextureFormat;

pub const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;