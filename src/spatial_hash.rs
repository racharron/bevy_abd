use std::collections::HashMap;
use nalgebra::{Point3, RealField};
use crate::AffineTransform;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SpatialBin {
    level: i16,
    x: i16,
    y: i16,
    z: i16,
}

/// Holds spatially hashed references to shapes.
#[derive(Debug)]
pub struct SpatialHash<'a, Scalar: RealField> {
    max_level: i16,
    points: HashMap<SpatialBin, Vec<&'a Point3<Scalar>>>,
    edges: HashMap<SpatialBin, Vec<&'a [Point3<Scalar>; 2]>>,
    triangles: HashMap<SpatialBin, Vec<&'a [Point3<Scalar>; 3]>>,
}

/// Represents that an object on some trajectory will stay within these bounds for some specified time interval.
pub trait Bounded<Scalar: RealField + TryInto<i16>> {
    fn bounds(&self, q: AffineTransform<Scalar>, dq: AffineTransform<Scalar>) -> (Point3<Scalar>, Point3<Scalar>);
    fn add_to_spatial_hash<'a>(&'a self, sh: &mut SpatialHash<'a, Scalar>, bin: SpatialBin);
    fn add_bins<'a>(&'a self, sh: &mut SpatialHash<'a, Scalar>, q: AffineTransform<Scalar>, dq: AffineTransform<Scalar>) {
        let (min, max) = self.bounds(q, dq);
        let dx = max.x.clone() - min.x.clone();
        let dy = max.y.clone() - min.y.clone();
        let dz = max.z.clone() - min.z.clone();
        let scale = dx.max(dy.max(dz));
        let level = scale.log2().ceil().try_into().ok().unwrap();
        let min_x = min.x.clone();
        let max_x = max.x.clone();
        let min_y = min.y.clone();
        let max_y = max.y.clone();
        let min_z = min.z.clone();
        let max_z = max.z.clone();
        let xu = dimbin(max_x, level);
        let xl = dimbin(min_x, level);
        let yu = dimbin(max_y, level);
        let yl = dimbin(min_y, level);
        let zu = dimbin(max_z, level);
        let zl = dimbin(min_z, level);
        self.add_to_spatial_hash(sh, SpatialBin {
            level,
            x: xl,
            y: yl,
            z: zl,
        });
        if yl != yu {
            self.add_to_spatial_hash(sh, SpatialBin {
                level,
                x: xl,
                y: yu,
                z: zl,
            })
        }
        if xl != xu {
            self.add_to_spatial_hash(sh, SpatialBin {
                level,
                x: xu,
                y: yl,
                z: zl,
            });
        }
        if xl != xu && yl != yu  {
            self.add_to_spatial_hash(sh, SpatialBin {
                level,
                x: xu,
                y: yu,
                z: zl,
            });
        }
        if zl != zu {
            self.add_to_spatial_hash(sh, SpatialBin {
                level,
                x: xl,
                y: yl,
                z: zu
            })
        }
        if xl != xu && zl != zu {
            self.add_to_spatial_hash(sh, SpatialBin {
                level,
                x: xu,
                y: yl,
                z: zu,
            })
        }
        if yl != yu && zl != zu {
            self.add_to_spatial_hash(sh, SpatialBin {
                level,
                x: xl,
                y: yu,
                z: zu,
            })
        }
        if xl != xu && yl != yu && zl != zu {
            self.add_to_spatial_hash(sh, SpatialBin {
                level,
                x: xu,
                y: yu,
                z: zu,
            })
        }
    }
}

impl SpatialBin {
    pub fn enclosing(&self) -> Self {
        SpatialBin {
            level: self.level + 1,
            x: self.x >> 1,
            y: self.y >> 1,
            z: self.z >> 1,
        }
    }
}

fn dimbin<Scalar: RealField + TryInto<i16>>(x: Scalar, level: i16) -> i16 {
    let factor = Scalar::from_u8(2).unwrap().powi(-(level as i32));
    let scaled = x * factor;
    let is_negative = scaled.is_negative();
    let binned = scaled % Scalar::from_i16(i16::MIN).unwrap();
    if binned.is_zero() && is_negative {
        i16::MIN
    } else {
        binned.try_into().ok().unwrap()
    }
}

impl<'a, Scalar: RealField + TryInto<i16>> Bounded<Scalar> for &'a Point3<Scalar> {
    fn bounds(&self, q: AffineTransform<Scalar>, dq: AffineTransform<Scalar>) -> (Point3<Scalar>, Point3<Scalar>) {
        let [[min_x, max_x], [min_y, max_y], [min_z, max_z]] = point_bounds(self, &q, &dq);
        (Point3::new(min_x, min_y, min_z), Point3::new(max_x, max_y, max_z))
    }

    fn add_to_spatial_hash<'b>(&'b self, sh: &mut SpatialHash<'b, Scalar>, bin: SpatialBin) {
        sh.points.entry(bin).or_default().push(self)
    }
}

impl<'a, Scalar: RealField + TryInto<i16>> Bounded<Scalar> for &'a [Point3<Scalar>; 2] {
    fn bounds(&self, q: AffineTransform<Scalar>, dq: AffineTransform<Scalar>) -> (Point3<Scalar>, Point3<Scalar>) {
        let [[min_x_0, max_x_0], [min_y_0, max_y_0], [min_z_0, max_z_0]] = point_bounds(&self[0], &q, &dq);
        let [[min_x_1, max_x_1], [min_y_1, max_y_1], [min_z_1, max_z_1]] = point_bounds(&self[1], &q, &dq);
        let min_x = min_x_0.min(min_x_1);
        let max_x = max_x_0.max(max_x_1);
        let min_y = min_y_0.min(min_y_1);
        let max_y = max_y_0.max(max_y_1);
        let min_z = min_z_0.min(min_z_1);
        let max_z = max_z_0.max(max_z_1);
        (Point3::new(min_x, min_y, min_z), Point3::new(max_x, max_y, max_z))
    }

    fn add_to_spatial_hash<'b>(&'b self, sh: &mut SpatialHash<'b, Scalar>, bin: SpatialBin) {
        sh.edges.entry(bin).or_default().push(self);
    }
}

impl<'a, Scalar: RealField + TryInto<i16>> Bounded<Scalar> for &'a [Point3<Scalar>; 3] {
    fn bounds(&self, q: AffineTransform<Scalar>, dq: AffineTransform<Scalar>) -> (Point3<Scalar>, Point3<Scalar>) {
        let [[min_x_0, max_x_0], [min_y_0, max_y_0], [min_z_0, max_z_0]] = point_bounds(&self[0], &q, &dq);
        let [[min_x_1, max_x_1], [min_y_1, max_y_1], [min_z_1, max_z_1]] = point_bounds(&self[1], &q, &dq);
        let [[min_x_2, max_x_2], [min_y_2, max_y_2], [min_z_2, max_z_2]] = point_bounds(&self[2], &q, &dq);
        let min_x = min_x_0.min(min_x_1).min(min_x_2);
        let max_x = max_x_0.max(max_x_1).max(max_x_2);
        let min_y = min_y_0.min(min_y_1).min(min_y_2);
        let max_y = max_y_0.max(max_y_1).max(max_y_2);
        let min_z = min_z_0.min(min_z_1).min(min_z_2);
        let max_z = max_z_0.max(max_z_1).max(max_z_2);
        (Point3::new(min_x, min_y, min_z), Point3::new(max_x, max_y, max_z))
    }

    fn add_to_spatial_hash<'b>(&'b self, sh: &mut SpatialHash<'b, Scalar>, bin: SpatialBin) {
        sh.triangles.entry(bin).or_default().push(self)
    }
}

fn point_bounds<'a, Scalar: RealField + TryInto<i16>>(point: &Point3<Scalar>, q: &AffineTransform<Scalar>, dq: &AffineTransform<Scalar>) -> [[Scalar; 2]; 3] {
    let endpoint = dq.linear_transform() * (q.linear_transform() * point) + q.translation() + dq.translation();
    let min_x = point.x.clone().min(endpoint.x.clone());
    let max_x = point.x.clone().max(endpoint.x.clone());
    let min_y = point.y.clone().min(endpoint.y.clone());
    let max_y = point.y.clone().max(endpoint.y.clone());
    let min_z = point.z.clone().min(endpoint.z.clone());
    let max_z = point.z.clone().max(endpoint.z.clone());
    [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
}

