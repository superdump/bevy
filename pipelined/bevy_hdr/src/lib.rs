mod bloom_node;
mod hdr_texture_node;
mod tone_mapping_node;

pub use bloom_node::*;
pub use hdr_texture_node::*;
pub use tone_mapping_node::*;

use bevy_render2::render_resource::TextureFormat;

pub const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
