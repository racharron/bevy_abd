use nalgebra::{RealField, Vector3};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use bevy_asset::Handle;
use bevy_ecs::event::Event;
use bevy_ecs::system::Resource;
use bevy_render::mesh::{Indices, Mesh};
use crate::dynamics::Moments;

#[derive(Resource)]
pub struct MeshCache<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> {
    pub(crate) mapping: HashMap<Handle<Mesh>, MeshShape<Scalar, Index>>,
    pub(crate) queued: HashSet<Handle<Mesh>>,
}

/// There are no degenerate triangles in the triangle strip.
pub struct MeshShape<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> {
    vertices: [Vec<Scalar>; 3],
    /// If empty, assume that the there is a single triangle strip consisting of
    /// the vertices in-order.
    indices: Vec<Index>,
    /// Indicates the boundaries between the triangle strips in the indexes
    strip_boundaries: Vec<usize>,
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> MeshShape<Scalar, Index> {
    pub fn moments(&self) -> Moments<Scalar> {
        let m0 =
            self.accumulate_tet_property(|a, b, c| a.dot(&b.cross(&c)))
                / Scalar::from_u32(6).unwrap();
        let four = Scalar::from_u32(4).unwrap();
        let two = Scalar::from_u32(2).unwrap();
        let ten = Scalar::from_u32(10).unwrap();
        let twenty = Scalar::from_u32(20).unwrap();
        let mx =
            self.accumulate_tet_property(|a, b, c|
                a.x.clone() + b.x.clone() + c.x.clone()
            ) * m0.clone() / four.clone();
        let my =
            self.accumulate_tet_property(|a, b, c|
                a.y.clone() + b.y.clone() + c.y.clone()
            ) * m0.clone() / four.clone();
        let mz =
            self.accumulate_tet_property(|a, b, c|
                a.z.clone() + b.z.clone() + c.z.clone()
            ) * m0.clone() / four;
        let mxy =
            self.accumulate_tet_property(|a, b, c|
                a.x.clone() * (two.clone() * a.y.clone() + b.y.clone() + c.y.clone())
                    + b.x.clone() * (a.y.clone() + two.clone() * b.y.clone() + c.y.clone())
                    + c.x.clone() * (a.y.clone() + b.y.clone() + two.clone() * c.y.clone())
            ) * m0.clone() / twenty.clone();
        let mxz =
            self.accumulate_tet_property(|a, b, c|
                a.x.clone() * (two.clone() * a.z.clone() + b.z.clone() + c.z.clone())
                    + b.x.clone() * (a.z.clone() + two.clone() * b.z.clone() + c.z.clone())
                    + c.x.clone() * (a.z.clone() + b.z.clone() + two.clone() * c.z.clone())
            ) * m0.clone() / twenty.clone();
        let myz =
            self.accumulate_tet_property(|a, b, c|
                a.z.clone() * (two.clone() * a.y.clone() + b.y.clone() + c.y.clone())
                    + b.z.clone() * (a.y.clone() + two.clone() * b.y.clone() + c.y.clone())
                    + c.z.clone() * (a.y.clone() + b.y.clone() + two.clone() * c.y.clone())
            ) * m0.clone() / twenty;
        let mx2 = self.accumulate_tet_property(|a, b, c|
            a.x.clone().powi(2) + b.x.clone().powi(2) + c.x.clone().powi(2)
                + a.x.clone() * b.x.clone() + a.x.clone() * c.x.clone() + b.x.clone() * c.x.clone()
        ) * m0.clone() / ten.clone();
        let my2 = self.accumulate_tet_property(|a, b, c|
            a.y.clone().powi(2) + b.y.clone().powi(2) + c.y.clone().powi(2)
                + a.y.clone() * b.y.clone() + a.y.clone() * c.y.clone() + b.y.clone() * c.y.clone()
        ) * m0.clone() / ten.clone();
        let mz2 = self.accumulate_tet_property(|a, b, c|
            a.z.clone().powi(2) + b.z.clone().powi(2) + c.z.clone().powi(2)
                + a.z.clone() * b.z.clone() + a.z.clone() * c.z.clone() + b.z.clone() * c.z.clone()
        ) * m0.clone() / ten.clone();
        Moments {
            v: m0,
            x: mx,
            y: my,
            z: mz,
            xx: mx2,
            xy: mxy,
            xz: mxz,
            yy: my2,
            yz: myz,
            zz: mz2,
        }
    }
    fn accumulate_tet_property(&self, f: impl Clone + Fn(Vector3<Scalar>, Vector3<Scalar>, Vector3<Scalar>) -> Scalar) -> Scalar {
        let mut tri_vols =
            Vec::with_capacity(self.indices.len() - 2 * (1 + self.strip_boundaries.len()));
        let mut indices = &*self.indices;
        for next_start in self.strip_boundaries.iter().cloned() {
            let (current, next) = indices.split_at(next_start.into());
            self.acc_strip(f.clone(), &mut tri_vols, current);
            indices = next;
        }
        tri_vols.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sum_vec = Vec::with_capacity(tri_vols.capacity() / 2 + 1);
        while tri_vols.len() > 1 {
            for pair in tri_vols.chunks(2) {
                match pair {
                    [a, b] => sum_vec.push(a.clone() + b.clone()),
                    [a] => sum_vec.push(a.clone()),
                    _ => unreachable!()
                }
            }
            tri_vols.clear();
            std::mem::swap(&mut tri_vols, &mut sum_vec)
        }
        tri_vols.into_iter().next().unwrap()
    }

    fn acc_strip(
        &self,
        f: impl Fn(Vector3<Scalar>, Vector3<Scalar>, Vector3<Scalar>) -> Scalar,
        tri_vols: &mut Vec<Scalar>,
        strip: &[Index],
    ) {
        for tri in strip.windows(3) {
            let mut points = tri.into_iter()
                .map(|&i| Vector3::new(
                    self.vertices[0][i.into()].clone(),
                    self.vertices[1][i.into()].clone(),
                    self.vertices[2][i.into()].clone(),
                ));
            let a = points.next().unwrap();
            let b = points.next().unwrap();
            let c = points.next().unwrap();
            tri_vols.push(f(a, b, c));
        }
    }
}

#[derive(Event, Debug)]
pub enum MeshColliderConversionError {
    U16ToIndex(usize),
    U32ToIndex(usize),
    NoPositions,
    NoFloatVertices,
    /// Indicates an a-b-a pattern in the mesh.  This is invalid, and not handled (at the moment).
    ///
    /// This variant carries the position of the index (or point for meshes without index lists) where
    /// the degenerate triangle began
    InvalidDegeneracy(usize),
}

impl Display for MeshColliderConversionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshColliderConversionError::U16ToIndex(position) => {
                write!(f, "Could not convert u16 to index type at position {}", position)
            }
            MeshColliderConversionError::U32ToIndex(position) => {
                write!(f, "Could not convert u16 to index type at position {}", position)
            }
            MeshColliderConversionError::NoPositions => {
                write!(f, "Mesh has no vertex position attribute")
            }
            MeshColliderConversionError::NoFloatVertices => {
                write!(f, "Could not convert vertex positions to floats")
            }
            MeshColliderConversionError::InvalidDegeneracy(position) => {
                write!(f, "Invalid ABA degeneracy at position {}", position)
            }
        }
    }
}

impl Error for MeshColliderConversionError {}

impl<Scalar, Index> TryFrom<&Mesh> for MeshShape<Scalar, Index>
where Scalar: RealField + From<f32>,
      Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32>
{
    type Error = MeshColliderConversionError;
    fn try_from(value: &Mesh) -> Result<Self, Self::Error> {
        let vertices = value.attribute(Mesh::ATTRIBUTE_POSITION).ok_or(MeshColliderConversionError::NoPositions)?
            .as_float3().ok_or(MeshColliderConversionError::NoFloatVertices)?;
        let mut xs = Vec::with_capacity(vertices.len());
        let mut ys = Vec::with_capacity(vertices.len());
        let mut zs = Vec::with_capacity(vertices.len());
        let mut is = Vec::with_capacity(vertices.len());
        let mut boundaries = Vec::new();

        match value.indices() {
            Some(Indices::U16(indices))  => {
                let mut prev = None;
                let mut prev_prev = None;
                for (i, &index) in indices.iter().enumerate() {
                    if Some(index) == prev_prev {
                        return Err(MeshColliderConversionError::InvalidDegeneracy(i - 2))
                    } else if Some(index) != prev {
                        if prev == prev_prev {
                            boundaries.push(i);
                        } else {
                            let [x, y, z] = vertices[index as usize].clone();
                            let index = Index::try_from(index).map_err(|_| MeshColliderConversionError::U16ToIndex(i))?;
                            xs.push(x.into());
                            ys.push(y.into());
                            zs.push(z.into());
                            is.push(index);
                        }
                    }
                    prev_prev = prev;
                    prev = Some(index);
                }
                Ok(MeshShape {
                    vertices: [xs, ys, zs],
                    indices: is,
                    strip_boundaries: boundaries,
                })
            }
            Some(Indices::U32(indices))   =>  {
                let mut prev = None;
                let mut prev_prev = None;
                for (i, &index) in indices.iter().enumerate() {
                    if Some(index) == prev_prev {
                        return Err(MeshColliderConversionError::InvalidDegeneracy(i - 2))
                    } else if Some(index) != prev {
                        if prev == prev_prev {
                            boundaries.push(i);
                        } else {
                            let [x, y, z] = vertices[index as usize].clone();
                            let index = Index::try_from(index).map_err(|_| MeshColliderConversionError::U32ToIndex(i))?;
                            xs.push(x.into());
                            ys.push(y.into());
                            zs.push(z.into());
                            is.push(index);
                        }
                    }
                    prev_prev = prev;
                    prev = Some(index);
                }
                Ok(MeshShape {
                    vertices: [xs, ys, zs],
                    indices: is,
                    strip_boundaries: boundaries,
                })
            }
            None    => {
                let mut prev = None;
                let mut prev_prev = None;
                for (i, &p @ [x, y, z]) in vertices.iter().enumerate() {
                    if Some(p) == prev_prev {
                        return Err(MeshColliderConversionError::InvalidDegeneracy(i - 2))
                    } else if Some(p) != prev {
                        if prev == prev_prev {
                            boundaries.push(i);
                        } else {
                            xs.push(x.into());
                            ys.push(y.into());
                            zs.push(z.into());
                        }
                    }
                    prev_prev = prev;
                    prev = Some(p);
                }
                Ok(MeshShape {
                    vertices: [xs, ys, zs],
                    indices: is,
                    strip_boundaries: boundaries,
                })
            }
        }
    }
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> MeshCache<Scalar, Index> {
    pub fn new() -> Self {
        MeshCache {
            mapping: Default::default(),
            queued: Default::default(),
        }
    }
}
