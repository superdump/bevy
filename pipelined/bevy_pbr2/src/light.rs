use bevy_math::{Vec2, Vec3};
use bevy_render2::{camera::OrthographicProjection, color::Color};

pub const DEFAULT_SHADOW_BIAS_MIN: f32 = 0.00005;
pub const DEFAULT_SHADOW_BIAS_MAX: f32 = 0.002;

/// An omnidirectional light
#[derive(Debug, Clone, Copy)]
pub struct OmniLight {
    pub color: Color,
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    pub shadow_bias_min_max: Vec2,
}

impl Default for OmniLight {
    fn default() -> Self {
        OmniLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            intensity: 200.0,
            range: 20.0,
            radius: 0.0,
            shadow_bias_min_max: Vec2::new(DEFAULT_SHADOW_BIAS_MIN, DEFAULT_SHADOW_BIAS_MAX),
        }
    }
}

/// A Directional light.
///
/// Directional lights don't exist in reality but they are a good
/// approximation for light sources VERY far away, like the sun or
/// the moon.
///
/// Valid values for `illuminance` are:
///
/// | Illuminance (lux) | Surfaces illuminated by                        |
/// |-------------------|------------------------------------------------|
/// | 0.0001            | Moonless, overcast night sky (starlight)       |
/// | 0.002             | Moonless clear night sky with airglow          |
/// | 0.05–0.3          | Full moon on a clear night                     |
/// | 3.4               | Dark limit of civil twilight under a clear sky |
/// | 20–50             | Public areas with dark surroundings            |
/// | 50                | Family living room lights                      |
/// | 80                | Office building hallway/toilet lighting        |
/// | 100               | Very dark overcast day                         |
/// | 150               | Train station platforms                        |
/// | 320–500           | Office lighting                                |
/// | 400               | Sunrise or sunset on a clear day.              |
/// | 1000              | Overcast day; typical TV studio lighting       |
/// | 10,000–25,000     | Full daylight (not direct sun)                 |
/// | 32,000–100,000    | Direct sunlight                                |
///
/// Source: [Wikipedia](https://en.wikipedia.org/wiki/Lux)
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    pub color: Color,
    pub illuminance: f32,
    direction: Vec3,
    pub shadow_projection: OrthographicProjection,
    pub shadow_bias_min_max: Vec2,
}

impl DirectionalLight {
    /// Create a new directional light component.
    pub fn new(
        color: Color,
        illuminance: f32,
        direction: Vec3,
        shadow_projection: OrthographicProjection,
        shadow_bias_min_max: Vec2,
    ) -> Self {
        DirectionalLight {
            color,
            illuminance,
            direction: direction.normalize(),
            shadow_projection,
            shadow_bias_min_max,
        }
    }

    /// Set direction of light.
    pub fn set_direction(&mut self, direction: Vec3) {
        self.direction = direction.normalize();
    }

    pub fn get_direction(&self) -> Vec3 {
        self.direction
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        let size = 100.0;
        DirectionalLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            illuminance: 100000.0,
            direction: Vec3::new(0.0, -1.0, 0.0),
            shadow_projection: OrthographicProjection {
                left: -size,
                right: size,
                bottom: -size,
                top: size,
                near: -size,
                far: size,
                ..Default::default()
            },
            shadow_bias_min_max: Vec2::new(DEFAULT_SHADOW_BIAS_MIN, DEFAULT_SHADOW_BIAS_MAX),
        }
    }
}

// Ambient light color.
#[derive(Debug)]
pub struct AmbientLight {
    pub color: Color,
    /// Color is premultiplied by brightness before being passed to the shader
    pub brightness: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: Color::rgb(1.0, 1.0, 1.0),
            brightness: 0.05,
        }
    }
}
