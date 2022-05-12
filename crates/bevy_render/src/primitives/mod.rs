use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_math::{Mat4, Vec3, Vec3A, Vec4, Vec4Swizzles};
use bevy_reflect::Reflect;
use wgpu::PrimitiveTopology;

use crate::mesh::{Indices, Mesh};

pub trait BoundingVolume {
    fn new_debug_mesh(&self) -> Mesh;
}

/// An Axis-Aligned Bounding Box
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct Aabb {
    pub center: Vec3A,
    pub half_extents: Vec3A,
}

impl Aabb {
    #[inline]
    pub fn from_min_max(minimum: Vec3, maximum: Vec3) -> Self {
        let minimum = Vec3A::from(minimum);
        let maximum = Vec3A::from(maximum);
        let center = 0.5 * (maximum + minimum);
        let half_extents = 0.5 * (maximum - minimum);
        Self {
            center,
            half_extents,
        }
    }

    /// Calculate the relative radius of the AABB with respect to a plane
    #[inline]
    pub fn relative_radius(&self, p_normal: &Vec3A, axes: &[Vec3A]) -> f32 {
        // NOTE: dot products on Vec3A use SIMD and even with the overhead of conversion are net faster than Vec3
        let half_extents = self.half_extents;
        Vec3A::new(
            p_normal.dot(axes[0]),
            p_normal.dot(axes[1]),
            p_normal.dot(axes[2]),
        )
        .abs()
        .dot(half_extents)
    }

    #[inline]
    pub fn min(&self) -> Vec3A {
        self.center - self.half_extents
    }

    #[inline]
    pub fn max(&self) -> Vec3A {
        self.center + self.half_extents
    }

    #[inline]
    pub fn vertices_mesh_space(&self) -> [Vec3; 8] {
        /*
              (2)-----(3)               Y
               | \     | \              |
               |  (1)-----(0) MAX       o---X
               |   |   |   |             \
          MIN (6)--|--(7)  |              Z
                 \ |     \ |
                  (5)-----(4)
        */
        let min = self.min();
        let max = self.max();
        [
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
        ]
    }
}

impl From<Sphere> for Aabb {
    #[inline]
    fn from(sphere: Sphere) -> Self {
        Self {
            center: sphere.center,
            half_extents: Vec3A::splat(sphere.radius),
        }
    }
}

impl From<&Aabb> for Mesh {
    fn from(aabb: &Aabb) -> Self {
        /*
              (2)-----(3)               Y
               | \     | \              |
               |  (1)-----(0) MAX       o---X
               |   |   |   |             \
          MIN (6)--|--(7)  |              Z
                 \ |     \ |
                  (5)-----(4)
        */
        let vertices: Vec<[f32; 3]> = aabb
            .vertices_mesh_space()
            .iter()
            .map(|vert| [vert.x, vert.y, vert.z])
            .collect();
        let uvs = vec![[0.0f32; 2]; 8];

        let indices = Indices::U32(vec![
            0, 1, 1, 2, 2, 3, 3, 0, // Top ring
            4, 5, 5, 6, 6, 7, 7, 4, // Bottom ring
            0, 4, 1, 5, 2, 6, 3, 7, // Verticals
        ]);

        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertices);
        mesh.set_indices(Some(indices));
        mesh
    }
}

impl BoundingVolume for Aabb {
    fn new_debug_mesh(&self) -> Mesh {
        Mesh::from(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Sphere {
    pub center: Vec3A,
    pub radius: f32,
}

impl Sphere {
    #[inline]
    pub fn intersects_obb(&self, aabb: &Aabb, local_to_world: &Mat4) -> bool {
        let aabb_center_world = *local_to_world * aabb.center.extend(1.0);
        let axes = [
            Vec3A::from(local_to_world.x_axis),
            Vec3A::from(local_to_world.y_axis),
            Vec3A::from(local_to_world.z_axis),
        ];
        let v = Vec3A::from(aabb_center_world) - self.center;
        let d = v.length();
        let relative_radius = aabb.relative_radius(&(v / d), &axes);
        d < self.radius + relative_radius
    }
}

impl From<&Sphere> for Mesh {
    fn from(sphere: &Sphere) -> Self {
        let radius = sphere.radius;
        let origin = sphere.center;
        let n_points = 24;
        let vertices_x0: Vec<[f32; 3]> = (0..n_points)
            .map(|i| {
                let angle = i as f32 * 2.0 * std::f32::consts::PI / (n_points as f32);
                [
                    0.0,
                    angle.sin() * radius + origin.y,
                    angle.cos() * radius + origin.z,
                ]
            })
            .collect();
        let vertices_y0: Vec<[f32; 3]> = (0..n_points)
            .map(|i| {
                let angle = i as f32 * 2.0 * std::f32::consts::PI / (n_points as f32);
                [
                    angle.cos() * radius + origin.x,
                    0.0,
                    angle.sin() * radius + origin.z,
                ]
            })
            .collect();
        let vertices_z0: Vec<[f32; 3]> = (0..n_points)
            .map(|i| {
                let angle = i as f32 * 2.0 * std::f32::consts::PI / (n_points as f32);
                [
                    angle.cos() * radius + origin.x,
                    angle.sin() * radius + origin.y,
                    0.0,
                ]
            })
            .collect();
        let vertices = [vertices_x0, vertices_y0, vertices_z0].concat();
        let indices_single: Vec<u32> = (0..n_points * 2)
            .map(|i| {
                let result = (i as u32 + 1) / 2;
                if result == n_points as u32 {
                    0
                } else {
                    result
                }
            })
            .collect();
        let indices = Indices::U32(
            [
                indices_single
                    .iter()
                    .map(|&index| index + n_points as u32)
                    .collect(),
                indices_single
                    .iter()
                    .map(|&index| index + 2 * n_points as u32)
                    .collect(),
                indices_single,
            ]
            .concat(),
        );
        let uvs = vec![[0.0f32; 2]; n_points * 3];
        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertices);
        mesh.set_indices(Some(indices));
        mesh
    }
}

impl BoundingVolume for Sphere {
    fn new_debug_mesh(&self) -> Mesh {
        Mesh::from(self)
    }
}

/// A plane defined by a unit normal and distance from the origin along the normal
/// Any point p is in the plane if n.p + d = 0
/// For planes defining half-spaces such as for frusta, if n.p + d > 0 then p is on
/// the positive side (inside) of the plane.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plane {
    normal_d: Vec4,
}

impl Plane {
    /// Constructs a `Plane` from a 4D vector whose first 3 components
    /// are the normal and whose last component is the distance along the normal
    /// from the origin.
    /// This constructor ensures that the normal is normalized and the distance is
    /// scaled accordingly so it represents the signed distance from the origin.
    #[inline]
    pub fn new(normal_d: Vec4) -> Self {
        Self {
            normal_d: normal_d * normal_d.xyz().length_recip(),
        }
    }

    /// `Plane` unit normal
    #[inline]
    pub fn normal(&self) -> Vec3A {
        Vec3A::from(self.normal_d)
    }

    /// Signed distance from the origin along the unit normal such that n.p + d = 0 for point p in
    /// the `Plane`
    #[inline]
    pub fn d(&self) -> f32 {
        self.normal_d.w
    }

    /// `Plane` unit normal and signed distance from the origin such that n.p + d = 0 for point p
    /// in the `Plane`
    #[inline]
    pub fn normal_d(&self) -> Vec4 {
        self.normal_d
    }

    pub fn triplanar_intersection(&self, p1: &Plane, p2: &Plane) -> Option<Vec3A> {
        let p0 = self;

        let m1 = Vec3A::new(p0.normal_d.x, p1.normal_d.x, p2.normal_d.x);
        let m2 = Vec3A::new(p0.normal_d.y, p1.normal_d.y, p2.normal_d.y);
        let m3 = Vec3A::new(p0.normal_d.z, p1.normal_d.z, p2.normal_d.z);
        let d = Vec3A::new(p0.normal_d.w, p1.normal_d.w, p2.normal_d.w);

        let u = m2.cross(m3);
        let v = m1.cross(d);

        let denominator = m1.dot(u);
        if denominator.abs() < f32::EPSILON {
            // Planes don't actually intersect in a point
            return None;
        }

        Some(Vec3A::new(
            d.dot(u) / denominator,
            m3.dot(v) / denominator,
            -m2.dot(v) / denominator,
        ))
    }
}

#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct Frustum {
    #[reflect(ignore)]
    pub planes: [Plane; 6],
}

impl Frustum {
    // NOTE: This approach of extracting the frustum planes from the view
    // projection matrix is from Foundations of Game Engine Development 2
    // Rendering by Lengyel. Slight modification has been made for when
    // the far plane is infinite but we still want to cull to a far plane.
    #[inline]
    pub fn from_view_projection(
        view_projection: &Mat4,
        view_translation: &Vec3,
        view_backward: &Vec3,
        far: f32,
    ) -> Self {
        let row3 = view_projection.row(3);
        let mut planes = [Plane::default(); 6];
        for (i, plane) in planes.iter_mut().enumerate().take(5) {
            let row = view_projection.row(i / 2);
            *plane = Plane::new(if (i & 1) == 0 && i != 4 {
                row3 + row
            } else {
                row3 - row
            });
        }
        let far_center = *view_translation - far * *view_backward;
        planes[5] = Plane::new(view_backward.extend(-view_backward.dot(far_center)));
        Self { planes }
    }

    #[inline]
    pub fn intersects_sphere(&self, sphere: &Sphere, intersect_far: bool) -> bool {
        let sphere_center = sphere.center.extend(1.0);
        let max = if intersect_far { 6 } else { 5 };
        for plane in &self.planes[..max] {
            if plane.normal_d().dot(sphere_center) + sphere.radius <= 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn intersects_obb(&self, aabb: &Aabb, model_to_world: &Mat4, intersect_far: bool) -> bool {
        let aabb_center_world = model_to_world.transform_point3a(aabb.center).extend(1.0);
        let axes = [
            Vec3A::from(model_to_world.x_axis),
            Vec3A::from(model_to_world.y_axis),
            Vec3A::from(model_to_world.z_axis),
        ];

        let max = if intersect_far { 6 } else { 5 };
        for plane in &self.planes[..max] {
            let p_normal = Vec3A::from(plane.normal_d());
            let relative_radius = aabb.relative_radius(&p_normal, &axes);
            if plane.normal_d().dot(aabb_center_world) + relative_radius <= 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn vertices_mesh_space(&self) -> [Vec3; 8] {
        /*
              (2)-----(3)               Y
               | \     | \              |
               |  (1)-----(0) MAX       o---X
               |   |   |   |             \
          MIN (6)--|--(7)  |              Z
                 \ |     \ |
                  (5)-----(4)
        */
        let n = self.planes.len();
        let mut corners = Vec::with_capacity(8);
        for i in 0..n {
            for j in (i - (i & 1) + 2)..n {
                for k in (j - (j & 1) + 2)..n {
                    corners.push(Vec3::from(
                        self.planes[i]
                            .triplanar_intersection(&self.planes[j], &self.planes[k])
                            .unwrap(),
                    ));
                }
            }
        }
        let corners = [
            corners[0], // right  top    near
            corners[4], // left   top    near
            corners[5], // left   top    far
            corners[1], // right  top    far
            corners[2], // right  bottom near
            corners[6], // left   bottom near
            corners[7], // left   bottom far
            corners[3], // right  bottom far
        ];
        // dbg!(&self.planes);
        // dbg!(&corners);
        corners
    }
}

impl From<&Frustum> for Mesh {
    fn from(frustum: &Frustum) -> Self {
        /*
              (2)-----(3)               Y
               | \     | \              |
               |  (1)-----(0) MAX       o---X
               |   |   |   |             \
          MIN (6)--|--(7)  |              Z
                 \ |     \ |
                  (5)-----(4)
        */
        let vertices: Vec<[f32; 3]> = frustum
            .vertices_mesh_space()
            .iter()
            .map(|vert| [vert.x, vert.y, vert.z])
            .collect();
        let uvs = vec![[0.0f32; 2]; 8];

        let indices = Indices::U32(vec![
            0, 1, 1, 2, 2, 3, 3, 0, // Top ring
            4, 5, 5, 6, 6, 7, 7, 4, // Bottom ring
            0, 4, 1, 5, 2, 6, 3, 7, // Verticals
        ]);

        let mut mesh = Mesh::new(PrimitiveTopology::LineList);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertices);
        mesh.set_indices(Some(indices));
        mesh
    }
}

impl BoundingVolume for Frustum {
    fn new_debug_mesh(&self) -> Mesh {
        Mesh::from(self)
    }
}

#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct CubemapFrusta {
    #[reflect(ignore)]
    pub frusta: [Frustum; 6],
}

impl CubemapFrusta {
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Frustum> {
        self.frusta.iter()
    }
    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Frustum> {
        self.frusta.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A big, offset frustum
    fn big_frustum() -> Frustum {
        Frustum {
            planes: [
                Plane::new(Vec4::new(-0.9701, -0.2425, -0.0000, 7.7611)),
                Plane::new(Vec4::new(-0.0000, 1.0000, -0.0000, 4.0000)),
                Plane::new(Vec4::new(-0.0000, -0.2425, -0.9701, 2.9104)),
                Plane::new(Vec4::new(-0.0000, -1.0000, -0.0000, 4.0000)),
                Plane::new(Vec4::new(-0.0000, -0.2425, 0.9701, 2.9104)),
                Plane::new(Vec4::new(0.9701, -0.2425, -0.0000, -1.9403)),
            ],
        }
    }

    #[test]
    fn intersects_sphere_big_frustum_outside() {
        // Sphere outside frustum
        let frustum = big_frustum();
        let sphere = Sphere {
            center: Vec3A::new(0.9167, 0.0000, 0.0000),
            radius: 0.7500,
        };
        assert!(!frustum.intersects_sphere(&sphere, true));
    }

    #[test]
    fn intersects_sphere_big_frustum_intersect() {
        // Sphere intersects frustum boundary
        let frustum = big_frustum();
        let sphere = Sphere {
            center: Vec3A::new(7.9288, 0.0000, 2.9728),
            radius: 2.0000,
        };
        assert!(frustum.intersects_sphere(&sphere, true));
    }

    // A frustum
    fn frustum() -> Frustum {
        Frustum {
            planes: [
                Plane::new(Vec4::new(-0.9701, -0.2425, -0.0000, 0.7276)),
                Plane::new(Vec4::new(-0.0000, 1.0000, -0.0000, 1.0000)),
                Plane::new(Vec4::new(-0.0000, -0.2425, -0.9701, 0.7276)),
                Plane::new(Vec4::new(-0.0000, -1.0000, -0.0000, 1.0000)),
                Plane::new(Vec4::new(-0.0000, -0.2425, 0.9701, 0.7276)),
                Plane::new(Vec4::new(0.9701, -0.2425, -0.0000, 0.7276)),
            ],
        }
    }

    #[test]
    fn intersects_sphere_frustum_surrounding() {
        // Sphere surrounds frustum
        let frustum = frustum();
        let sphere = Sphere {
            center: Vec3A::new(0.0000, 0.0000, 0.0000),
            radius: 3.0000,
        };
        assert!(frustum.intersects_sphere(&sphere, true));
    }

    #[test]
    fn intersects_sphere_frustum_contained() {
        // Sphere is contained in frustum
        let frustum = frustum();
        let sphere = Sphere {
            center: Vec3A::new(0.0000, 0.0000, 0.0000),
            radius: 0.7000,
        };
        assert!(frustum.intersects_sphere(&sphere, true));
    }

    #[test]
    fn intersects_sphere_frustum_intersects_plane() {
        // Sphere intersects a plane
        let frustum = frustum();
        let sphere = Sphere {
            center: Vec3A::new(0.0000, 0.0000, 0.9695),
            radius: 0.7000,
        };
        assert!(frustum.intersects_sphere(&sphere, true));
    }

    #[test]
    fn intersects_sphere_frustum_intersects_2_planes() {
        // Sphere intersects 2 planes
        let frustum = frustum();
        let sphere = Sphere {
            center: Vec3A::new(1.2037, 0.0000, 0.9695),
            radius: 0.7000,
        };
        assert!(frustum.intersects_sphere(&sphere, true));
    }

    #[test]
    fn intersects_sphere_frustum_intersects_3_planes() {
        // Sphere intersects 3 planes
        let frustum = frustum();
        let sphere = Sphere {
            center: Vec3A::new(1.2037, -1.0988, 0.9695),
            radius: 0.7000,
        };
        assert!(frustum.intersects_sphere(&sphere, true));
    }

    #[test]
    fn intersects_sphere_frustum_dodges_1_plane() {
        // Sphere avoids intersecting the frustum by 1 plane
        let frustum = frustum();
        let sphere = Sphere {
            center: Vec3A::new(-1.7020, 0.0000, 0.0000),
            radius: 0.7000,
        };
        assert!(!frustum.intersects_sphere(&sphere, true));
    }

    // A long frustum.
    fn long_frustum() -> Frustum {
        Frustum {
            planes: [
                Plane::new(Vec4::new(-0.9998, -0.0222, -0.0000, -1.9543)),
                Plane::new(Vec4::new(-0.0000, 1.0000, -0.0000, 45.1249)),
                Plane::new(Vec4::new(-0.0000, -0.0168, -0.9999, 2.2718)),
                Plane::new(Vec4::new(-0.0000, -1.0000, -0.0000, 45.1249)),
                Plane::new(Vec4::new(-0.0000, -0.0168, 0.9999, 2.2718)),
                Plane::new(Vec4::new(0.9998, -0.0222, -0.0000, 7.9528)),
            ],
        }
    }

    #[test]
    fn intersects_sphere_long_frustum_outside() {
        // Sphere outside frustum
        let frustum = long_frustum();
        let sphere = Sphere {
            center: Vec3A::new(-4.4889, 46.9021, 0.0000),
            radius: 0.7500,
        };
        assert!(!frustum.intersects_sphere(&sphere, true));
    }

    #[test]
    fn intersects_sphere_long_frustum_intersect() {
        // Sphere intersects frustum boundary
        let frustum = long_frustum();
        let sphere = Sphere {
            center: Vec3A::new(-4.9957, 0.0000, -0.7396),
            radius: 4.4094,
        };
        assert!(frustum.intersects_sphere(&sphere, true));
    }
}
