use bevy_math::{Mat4, Vec3, Vec4, Vec4Swizzles};

// NOTE: This Aabb implementation mostly just copied from aevyrie's bevy_mod_bounding
#[derive(Clone, Debug)]
pub struct Aabb {
    pub minimum: Vec3,
    pub maximum: Vec3,
}

impl Aabb {
    pub fn vertices(&self) -> [Vec3; 8] {
        /*
              (2)-----(3)               Y
               | \     | \              |
               |  (1)-----(0) MAX       o---X
               |   |   |   |             \
          MIN (6)--|--(7)  |              Z
                 \ |     \ |
                  (5)-----(4)
        */
        [
            Vec3::new(self.maximum.x, self.maximum.y, self.maximum.z),
            Vec3::new(self.minimum.x, self.maximum.y, self.maximum.z),
            Vec3::new(self.minimum.x, self.maximum.y, self.minimum.z),
            Vec3::new(self.maximum.x, self.maximum.y, self.minimum.z),
            Vec3::new(self.maximum.x, self.minimum.y, self.maximum.z),
            Vec3::new(self.minimum.x, self.minimum.y, self.maximum.z),
            Vec3::new(self.minimum.x, self.minimum.y, self.minimum.z),
            Vec3::new(self.maximum.x, self.minimum.y, self.minimum.z),
        ]
    }

    pub fn translated(&self, translation: Vec3) -> Aabb {
        Aabb {
            minimum: translation + self.minimum,
            maximum: translation + self.maximum,
        }
    }
}

/// A plane defined by a normal and distance value along the normal
/// Any point p is in the plane if n.p = d
/// For planes defining half-spaces such as for frusta, if n.p > d then p is on the positive side of the plane.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plane {
    pub normal_d: Vec4,
}

pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    // NOTE: This approach of extracting the frustum planes from the view projection matrix is taken from:
    //       https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
    pub fn from_view_projection(
        view_projection: &Mat4,
        view_translation: &Vec3,
        view_backward: &Vec3,
        far: f32,
    ) -> Self {
        let row3 = view_projection.row(3);
        let mut planes = [Plane::default(); 6];
        for i in 0..=4 {
            let row = view_projection.row(i / 2);
            let v = if i & 1 == 0 && i != 4 {
                row3 + row
            } else {
                row3 - row
            };
            planes[i].normal_d = v.xyz().normalize().extend(v.w);
        }
        let far_center = *view_translation - far * *view_backward;
        planes[5].normal_d = view_backward.extend(-view_backward.dot(far_center));

        Self { planes }
    }

    // NOTE: Copied from rafx!
    fn calculate_bitmask(&self, point: Vec3) -> i32 {
        let px = point.x;
        let py = point.y;
        let pz = point.z;

        let f1 = self.planes[0].normal_d;
        let f2 = self.planes[1].normal_d;
        let f3 = self.planes[2].normal_d;
        let f4 = self.planes[3].normal_d;
        let f5 = self.planes[4].normal_d;
        let f6 = self.planes[5].normal_d;

        let mut bitmask = 0;
        bitmask |= ((f1.w + f1.x * px + f1.y * py + f1.z * pz <= 0.) as i32) << 0;
        bitmask |= ((f2.w + f2.x * px + f2.y * py + f2.z * pz <= 0.) as i32) << 1;
        bitmask |= ((f3.w + f3.x * px + f3.y * py + f3.z * pz <= 0.) as i32) << 2;
        bitmask |= ((f4.w + f4.x * px + f4.y * py + f4.z * pz <= 0.) as i32) << 3;
        bitmask |= ((f5.w + f5.x * px + f5.y * py + f5.z * pz <= 0.) as i32) << 4;
        bitmask |= ((f6.w + f6.x * px + f6.y * py + f6.z * pz <= 0.) as i32) << 5;

        bitmask
    }

    pub fn contains(&self, point: Vec3) -> bool {
        self.calculate_bitmask(point) <= 0
    }

    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        for corner in aabb.vertices() {
            if self.contains(corner) {
                return true;
            }
        }
        false
    }
}
