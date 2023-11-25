use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::HashSet;
use nalgebra::{Point3, RealField, Vector3};
use std::hash::Hash;
use bevy_asset::Asset;
use bevy_reflect::TypePath;
use itertools::Itertools;
use thiserror::Error;
use crate::dynamics::Moments;

#[derive(TypePath, Debug)]
pub struct MeshGeomtry<Scalar: RealField + PartialOrd, Index: Copy + Hash + TryInto<usize>> {
    vertices: Vec<Point3<Scalar>>,
    pub(super) indices: Vec<Index>,
    edges: Vec<[Index; 2]>,
}

/// There are no degenerate triangles in the triangle strip.
#[derive(TypePath, Debug)]
pub struct ColliderMesh<Scalar: RealField + PartialOrd, Index: Copy + Hash + TryInto<usize>> {
    pub(super) mesh: MeshGeomtry<Scalar, Index>,
    /// The moments of inertia (including volume)
    moments: Moments<Scalar>,
}

#[derive(Debug, Error)]
#[error("Vertex index in mesh too large ({index})")]
pub struct MeshCreationError {
    index: usize,
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Eq + Hash + TryInto<usize>> MeshGeomtry<Scalar, Index> {
    /// Returns the (unscaled) moments of inertia of the strip.
    ///
    /// Unscaled means that the density of the object is assumed to be 1, and thus this will
    /// need to be scaled by the actual density of the object.

    pub fn moments(&self) -> Moments<Scalar> {
        let m0 = ikb_comp_sum(self.oriented_triangles()
            .map(|[a, b, c]|
                signed_tri_volume(a.coords, b.coords, c.coords)));
        let mx = self.accumulate_centroid(|p| p.x.clone());
        let my = self.accumulate_centroid(|p| p.y.clone());
        let mz = self.accumulate_centroid(|p| p.z.clone());
        let mxy = self.accumulate_quadratic(|p| p.x.clone() * p.y.clone());
        let mxz = self.accumulate_quadratic(|p| p.x.clone() * p.z.clone());
        let myz = self.accumulate_quadratic(|p| p.y.clone() * p.z.clone());
        let mx2 = self.accumulate_quadratic(|p| p.x.clone().powi(2));
        let my2 = self.accumulate_quadratic(|p| p.y.clone().powi(2));
        let mz2 = self.accumulate_quadratic(|p| p.z.clone().powi(2));
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
    /// Returns the triangles properly oriented.
    pub fn oriented_triangles<'a>(&'a self) -> impl 'a + Iterator<Item=[Point3<Scalar>; 3]> {
        // let mut indices = &self.indices[..];
        // let mut strips = &self.strips[..];
        let mut flip = false;
        self.indices.windows(3).filter_map(move |tri| {
            let [a, b, c] = <&[Index; 3]>::try_from(tri).unwrap().clone();
            let [a, b, c] = if a == b || b == c || c == a {
                return None
            } else if flip {
                [b, a, c]
            } else {
                [a, b, c]
            };
            flip = !flip;
            Some([self.vertices[a.try_into().ok().unwrap()].clone(), self.vertices[b.try_into().ok().unwrap()].clone(), self.vertices[c.try_into().ok().unwrap()].clone()])
        })
        /*
        std::iter::from_fn(move || {
            let (current, mut flip) = if let Some((&StripData { length, flip }, rest)) = strips.split_first() {
                strips = rest;
                let (current, rest) = indices.split_at(length as usize);
                indices = rest;
                (current, flip)
            } else if indices.is_empty() {
                return None
            } else {
                let current = indices;
                indices = &indices[0..0];
                (current, self.flip_last)
            };
            Some(current.windows(3).map(move |tri| {
                let tri = if flip {
                    [
                        self.vertices[tri[1].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[0].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[2].clone().try_into().ok().unwrap()].clone()
                    ]
                } else {
                    [
                        self.vertices[tri[0].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[1].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[2].clone().try_into().ok().unwrap()].clone()
                    ]
                };
                flip = !flip;
                tri
            }))
        }).flatten()
        */
    }

    fn accumulate_centroid(&self, f: fn(Vector3<Scalar>) -> Scalar) -> Scalar {
        let mut terms = Vec::new();
        for [a, b, c] in self.oriented_triangles() {
            let scale = signed_tri_volume(a.coords.clone(), b.coords.clone(), c.coords.clone()) / Scalar::from_u8(4).unwrap();
            terms.extend([
                scale.clone() * f(a.coords.clone_owned()),
                scale.clone() * f(b.coords.clone_owned()),
                scale.clone() * f(c.coords.clone_owned()),
            ]);
        }
        ikb_comp_sum(terms)
    }

    fn accumulate_quadratic(
        &self,
        f: fn(Vector3<Scalar>) -> Scalar,
    ) -> Scalar {
        let mut terms = Vec::new();
        for [a, b, c] in self.oriented_triangles() {
            let scale = signed_tri_volume(a.coords.clone(), b.coords.clone(), c.coords.clone()) / Scalar::from_u8(20).unwrap();
            terms.extend([
                scale.clone() * f(a.coords.clone_owned()),
                scale.clone() * f(b.coords.clone_owned()),
                scale.clone() * f(c.coords.clone_owned()),
                scale * f(a.coords + b.coords + c.coords)
            ]);
        }
        ikb_comp_sum(terms)
    }
}

fn signed_tri_volume<Scalar: RealField>(a: Vector3<Scalar>, b: Vector3<Scalar>, c: Vector3<Scalar>) -> Scalar {
    a.dot(&b.cross(&c)) / Scalar::from_u8(6).unwrap()
}

/// Improved Kahan–Babuška compensated summation algorithm.
fn ikb_comp_sum<Scalar: RealField>(terms: impl IntoIterator<Item=Scalar>) -> Scalar {
    let mut sum = Scalar::zero();
    let mut c = Scalar::zero();
    for term in terms {
        let t = sum.clone() + term.clone();
        if sum.clone().abs() >= term.clone().abs() {
            c += (sum.clone() - t.clone()) + term;
        } else {
            c += (term - t.clone()) + sum.clone();
        }
        sum = t;
    }
    sum + c
}

impl<Scalar, Index> Asset for ColliderMesh<Scalar, Index>
    where Scalar: RealField + PartialOrd + TypePath, Index: Copy + Hash + TryInto<usize> + TypePath + Send + Sync {}

impl<Scalar: RealField + PartialOrd, Index: Copy + Hash + TryInto<usize>> bevy_asset::VisitAssetDependencies for ColliderMesh<Scalar, Index> {
    fn visit_dependencies(&self, _: &mut impl FnMut(bevy_asset::UntypedAssetId)) {}
}

fn sorted_edge<Index: Ord>(a: Index, b: Index) -> [Index; 2] {
    if a < b {
        [a, b]
    } else {
        [b, a]
    }
}

#[derive(PartialOrd, PartialEq)]
struct OrdWrapper<T: PartialOrd>(T);
impl<T: PartialOrd> Eq for OrdWrapper<T> {}
impl<T: PartialOrd> Ord for OrdWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl<Scalar: RealField + PartialEq + Clone, Index: Copy + Hash + Ord + TryInto<usize> + TryFrom<usize>> ColliderMesh<Scalar, Index> {
    pub fn from_triangle_strip<S: Clone + Into<Scalar>>(vertices: impl IntoIterator<Item=impl Borrow<[S; 3]>>) -> Result<Self, MeshCreationError> {
        let mut deduplicated = Vec::new();
        let mut patched_indices = Vec::new();
        let mut edges = HashSet::new();
        let mut prev = None;
        for vertex in vertices {
            let vertex = Point3::from(vertex.borrow().clone().map(Into::into));
            let index = if let Some((i, _)) = deduplicated.iter().find_position(|p| *p == &vertex) {
                Index::try_from(i).ok().unwrap()
            } else {
                let index = Index::try_from(deduplicated.len()).ok().unwrap();
                deduplicated.push(vertex);
                index
            };
            patched_indices.push(index);
            if let Some(prev) = prev {
                if prev != index {
                    edges.insert(sorted_edge(prev, index));
                }
            }
            prev = Some(index);
        }
        let mesh = MeshGeomtry {
            vertices: deduplicated,
            indices: patched_indices,
            edges: edges.into_iter().collect(),
        };
        let moments = mesh.moments();
        Ok(ColliderMesh {
            mesh,
            moments,
        })
    }

    pub fn from_indexed_triangle_strip<S: Clone + Into<Scalar>, I: TryInto<usize>>(
        vertices: impl IntoIterator<Item = impl Borrow<[S; 3]>>,
        indices: impl IntoIterator<Item = I>
    ) -> Result<Self, MeshCreationError> {
        let mut deduplicated = Vec::new();
        let mut patched_indices = Vec::new();
        for vertex in vertices {
            let vertex = Point3::from(vertex.borrow().clone().map(Into::into));
            let index = if let Some((i, _)) = deduplicated.iter().find_position(|p| *p == &vertex) {
                Index::try_from(i).ok().unwrap()
            } else {
                let index = Index::try_from(deduplicated.len()).ok().unwrap();
                deduplicated.push(vertex);
                index
            };
            patched_indices.push(index);
        }
        let mut prev = None;
        let mut vertex_indices = Vec::new();
        let mut edges = HashSet::new();
        for index in indices {
            let index = Index::try_from(index.try_into().ok().unwrap()).ok().unwrap();
            vertex_indices.push(patched_indices[index.try_into().ok().unwrap()]);
            if let Some(prev) = prev {
                if prev != index {
                    edges.insert(sorted_edge(prev, index));
                }
            }
            prev = Some(index);
        }
        let mesh = MeshGeomtry {
            vertices: deduplicated,
            indices: vertex_indices,
            edges: edges.into_iter().collect(),
        };
        let moments = mesh.moments();
        Ok(ColliderMesh {
            mesh,
            moments,
        })
    }
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Ord + Hash + TryInto<usize> + TryFrom<usize>> ColliderMesh<Scalar, Index> {
    pub fn moments(&self) -> &Moments<Scalar> {
        &self.moments
    }
    pub fn oriented_triangles<'a>(&'a self) -> impl 'a + Iterator<Item=[Point3<Scalar>; 3]> {
        self.mesh.oriented_triangles()
    }
    /// Creates a new cuboid mesh.
    pub fn cuboid(x_length: Scalar, y_length: Scalar, z_length: Scalar) -> Self {
        const CUBE_INDICES: [u16; 14] = [0, 1, 2, 3, 7, 1, 5, 4, 7, 6, 2, 4, 0, 1];

        let x = x_length / Scalar::from_u8(2).unwrap();
        let y = y_length / Scalar::from_u8(2).unwrap();
        let z = z_length / Scalar::from_u8(2).unwrap();
        ColliderMesh::from_indexed_triangle_strip(
            [
                [-x.clone(), -y.clone(), z.clone()],
                [x.clone(), -y.clone(), z.clone()],
                [-x.clone(), y.clone(), z.clone()],
                [x.clone(), y.clone(), z.clone()],
                [-x.clone(), -y.clone(), -z.clone()],
                [x.clone(), -y.clone(), -z.clone()],
                [-x.clone(), y.clone(), -z.clone()],
                [x, y, -z],
            ],
            CUBE_INDICES
        ).unwrap()
    }
}
