use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use nalgebra::{Point3, RealField, Vector3};
use std::error::Error;
use std::hash::Hash;
use std::marker::PhantomData;
use bevy_asset::{Asset, AssetLoader, AssetPath, BoxedFuture, LoadContext};
use bevy_asset::io::Reader;
use bevy_reflect::TypePath;
use bevy_render::mesh::{Indices, Mesh};
use thiserror::Error;
use crate::dynamics::Moments;

#[derive(TypePath, Debug)]
pub struct TriangleStrips<Scalar: RealField + PartialOrd, Index: Copy + Hash + TryInto<usize>> {
    vertices: Vec<Point3<Scalar>>,
    /// Does not include the last strip.
    pub(super) strips: Vec<StripData>,
    /// Indicates if the last triangle strip needs to have its orientation flipped.
    flip_last: bool,
    pub(super) indices: Vec<Index>,
    edges: Vec<[Index; 2]>,
}

/// There are no degenerate triangles in the triangle strip.
#[derive(TypePath, Debug)]
pub struct MeshShape<Scalar: RealField + PartialOrd, Index: Copy + Hash + TryInto<usize>> {
    pub(super) triangle_strips: TriangleStrips<Scalar, Index>,
    /// The moments of inertia (including volume)
    moments: Moments<Scalar>,
}

#[derive(Debug, Error)]
pub enum MeshShapeConversionError {
    #[error("Could not convert u16 to index type at position {0}")]
    U16ToIndex(usize),
    #[error("Could not convert u16 to index type at position {0}")]
    U32ToIndex(usize),
    #[error("Could not convert usize to index type at position {0}")]
    IndexTooBig(usize),
    #[error("Mesh has no vertex position attribute")]
    NoPositions,
    #[error("Could not convert vertex positions to floats")]
    NoFloatVertices,
}

#[derive(Debug, Error)]
#[error("Vertex index in mesh too large ({index})")]
pub struct MeshCreationError {
    index: usize,
}

#[derive(Debug, Error)]
pub enum MeshShapeLoadingError<LE: Error> {
    #[error("Could not convert mesh from file `{path}`: {conversion_error}")]
    Conversion {
        path: AssetPath<'static>,
        #[source]
        conversion_error: MeshShapeConversionError,
    },
    #[error("Could not load mesh file from `{path}`: {load_error}")]
    Loading {
        path: AssetPath<'static>,
        #[source]
        load_error: LE,
    },
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub(super) struct StripData {
    length: u32,
    /// Indicates if we need to flip the orientation of this strip.
    flip: bool,
}

impl<LE: Error> MeshShapeLoadingError<LE> {
    pub fn path(&self) -> &AssetPath<'static> {
        match self {
            MeshShapeLoadingError::Conversion { path, .. } => path,
            MeshShapeLoadingError::Loading { path, .. } => path,
        }
    }
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Hash + TryInto<usize>> TriangleStrips<Scalar, Index> {
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
        let mut indices = &self.indices[..];
        let mut strips = &self.strips[..];
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
                /*let tri = if flip {
                    [
                        self.vertices[tri[1].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[0].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[2].clone().try_into().ok().unwrap()].clone()
                    ]
                } else {*/
                    [
                        self.vertices[tri[0].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[1].clone().try_into().ok().unwrap()].clone(),
                        self.vertices[tri[2].clone().try_into().ok().unwrap()].clone()
                    ]
                /*};
                flip = !flip;
                tri*/
            }))
        }).flatten()
    }

    fn accumulate_centroid(&self, f: fn(Vector3<Scalar>) -> Scalar) -> Scalar {
        let mut terms = Vec::with_capacity(4 * self.tri_count());
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
        let mut terms = Vec::with_capacity(4 * self.tri_count());
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

    fn tri_count(&self) -> usize {
        self.indices.len() - 2 * (1 + self.strips.len())
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

impl<Scalar, Index> Asset for MeshShape<Scalar, Index>
    where Scalar: RealField + PartialOrd + TypePath, Index: Copy + Hash + TryInto<usize> + TypePath + Send + Sync {}

impl<Scalar: RealField + PartialOrd, Index: Copy + Hash + TryInto<usize>> bevy_asset::VisitAssetDependencies for MeshShape<Scalar, Index> {
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

impl<Scalar: RealField + PartialOrd + Clone, Index: Copy + Hash + Eq + Ord + TryInto<usize> + TryFrom<usize>> MeshShape<Scalar, Index> {
    pub fn from_triangle_strip<S: Clone + Into<Scalar>>(vertices: impl IntoIterator<Item=impl Borrow<[S; 3]>>) -> Result<Self, MeshCreationError> {
        let mut strips = Vec::new();
        let mut vertex_map: BTreeMap<OrdWrapper<[Scalar; 3]>, _> = BTreeMap::new();
        let mut indices = Vec::new();
        let mut edges = HashSet::new();
        let mut global_flip = false;
        let mut current_flip = global_flip;
        let mut stored_prev = false;
        let mut length = 0;
        let mut prev = None;
        let mut prev_prev = None;
        let mut tri_vertices = Vec::new();
        for p in vertices.into_iter().map(|v| v.borrow().clone().map(Into::into)) {
            use std::collections::btree_map::Entry;
            let i = vertex_map.len();
            let index = match vertex_map.entry(OrdWrapper(p.clone())) {
                Entry::Occupied(occ) => *occ.get(),
                Entry::Vacant(vac) => {
                    tri_vertices.push(Point3::from(p.clone().map(Scalar::from)));
                    *vac.insert(i)
                }
            };
            let index = Index::try_from(index).map_err(|_| MeshCreationError { index })?;
            let stored;
            if Some(index) == prev || Some(index) == prev_prev || prev == prev_prev {
                if stored_prev {
                    strips.push(StripData { length, flip: current_flip });
                }
                length = 0;
                stored = false;
            } else if let [Some(prev), Some(prev_prev)] = [prev, prev_prev] {
                if !stored_prev {
                    indices.push(prev_prev);
                    indices.push(prev);
                    edges.insert(sorted_edge(prev, prev_prev));
                    edges.insert(sorted_edge(prev_prev, index));
                    length += 2;
                    current_flip = global_flip;
                }
                indices.push(index);
                edges.insert(sorted_edge(prev, index));
                stored = true;
                length += 1;
            } else {
                stored = false;
            }
            prev_prev = prev;
            prev = Some(index);
            stored_prev = stored;
            global_flip = !global_flip;
        }
        let strips = TriangleStrips {
            vertices: tri_vertices,
            strips,
            flip_last: current_flip,
            indices,
            edges: edges.into_iter().collect(),
        };
        let moments = strips.moments();
        Ok(MeshShape {
            triangle_strips: strips,
            moments,
        })
    }

    pub fn from_indexed_triangle_strip<S: Clone + Into<Scalar>, I: TryInto<usize>>(
        vertices: impl IntoIterator<Item = impl Borrow<[S; 3]>>,
        indices: impl IntoIterator<Item = I>
    ) -> Result<Self, MeshCreationError> {
        let mut strips = Vec::new();
        let mut vertex_map = BTreeMap::new();
        let mut index_map = Vec::new();
        let mut tri_indices = Vec::new();
        let mut edges = HashSet::new();
        let mut global_flip = false;
        let mut current_flip = global_flip;
        let mut stored_prev = false;
        let mut length = 0;
        let mut prev = None;
        let mut prev_prev = None;
        let mut tri_vertices = Vec::new();
        for p in vertices.into_iter().map(|v| v.borrow().clone().map(|s| s.try_into().unwrap())) {
            use std::collections::btree_map::Entry;
            let i = vertex_map.len();
            match vertex_map.entry(OrdWrapper(p.clone())) {
                Entry::Occupied(occ) => {
                    index_map.push(*occ.get());
                },
                Entry::Vacant(vac)  =>  {
                    tri_vertices.push(Point3::from(p.clone().map(Scalar::from)));
                    index_map.push(*vac.insert(i));
                }
            };
        }
        for index in indices.into_iter().
            map(|i| {
                let i = i.try_into().ok().unwrap();
                Index::try_from(i.clone()).map_err(|_| MeshCreationError { index: i })
            })
        {
            let index = index_map[index?.try_into().ok().unwrap()];
            let index = Index::try_from(index).map_err(|_| MeshCreationError { index })?;
            let stored;
            if Some(index) == prev || Some(index) == prev_prev || prev == prev_prev {
                if stored_prev {
                    strips.push(StripData { length, flip: current_flip });
                }
                length = 0;
                stored = false;
            } else if let [Some(prev), Some(prev_prev)] = [prev, prev_prev] {
                if !stored_prev {
                    tri_indices.push(prev_prev);
                    tri_indices.push(prev);
                    edges.insert(sorted_edge(prev, prev_prev));
                    edges.insert(sorted_edge(prev_prev, index));
                    length += 2;
                    current_flip = global_flip;
                }
                tri_indices.push(index);
                edges.insert(sorted_edge(prev, index));
                stored = true;
                length += 1;
            } else {
                stored = false;
            }
            prev_prev = prev;
            prev = Some(index);
            stored_prev = stored;
            global_flip = !global_flip;
        }
        let strips = TriangleStrips {
            vertices: tri_vertices,
            strips,
            flip_last: current_flip,
            indices: tri_indices,
            edges: edges.into_iter().collect(),
        };
        let moments = strips.moments();
        Ok(MeshShape {
            triangle_strips: strips,
            moments,
        })
    }
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Hash + Into<usize>> MeshShape<Scalar, Index> {
    pub fn moments(&self) -> &Moments<Scalar> {
        &self.moments
    }
    pub fn oriented_triangles<'a>(&'a self) -> impl 'a + Iterator<Item=[Point3<Scalar>; 3]> {
        self.triangle_strips.oriented_triangles()
    }
}

impl<Scalar, Index> TryFrom<&Mesh> for MeshShape<Scalar, Index>
    where Scalar: RealField + From<f32>,
          Index: Copy + Hash + Eq + Ord + TryInto<usize> + TryFrom<usize>
{
    type Error = MeshShapeConversionError;
    fn try_from(value: &Mesh) -> Result<Self, Self::Error> {
        let vertices = value.attribute(Mesh::ATTRIBUTE_POSITION).ok_or(MeshShapeConversionError::NoPositions)?
            .as_float3().ok_or(MeshShapeConversionError::NoFloatVertices)?;

        match value.indices() {
            Some(Indices::U16(raw_indices)) => {
                MeshShape::from_indexed_triangle_strip::<f32, u16>(vertices, raw_indices.into_iter().cloned())
                    .map_err(|err| MeshShapeConversionError::U16ToIndex(err.index))
            }
            Some(Indices::U32(raw_indices)) => {
                MeshShape::from_indexed_triangle_strip::<f32, u32>(vertices, raw_indices.into_iter().cloned())
                    .map_err(|err| MeshShapeConversionError::U32ToIndex(err.index))
            }
            None => {
                MeshShape::from_triangle_strip(vertices)
                    .map_err(|err| MeshShapeConversionError::IndexTooBig(err.index))
            }
        }
    }
}

/// This loader requires a loader that loads a mesh, and calls t
pub struct MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh>,
          Scalar: RealField + From<f32>,
          Index: Copy + TryInto<usize> + TryFrom<usize>
{
    loader: Loader,
    extensions: Option<&'static [&'static str]>,
    _phantom: PhantomData<fn() -> (Scalar, Index)>,
}

impl<Loader, Scalar, Index> MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh>,
          Scalar: RealField + From<f32>,
          Index: Copy + TryInto<usize> + TryFrom<usize>
{
    pub fn with_loader_and_extensions(loader: Loader, extensions: &'static [&'static str]) -> Self {
        MeshShapeLoader {
            loader,
            extensions: Some(extensions),
            _phantom: PhantomData,
        }
    }
    pub fn with_loader(loader: Loader) -> Self {
        MeshShapeLoader {
            loader,
            extensions: None,
            _phantom: PhantomData,
        }
    }
}

impl<Loader, Scalar, Index> Default for MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh> + Default,
          Scalar: RealField + From<f32>,
          Index: Copy + TryInto<usize> + TryFrom<usize>
{
    fn default() -> Self {
        Self::with_loader(Loader::default())
    }
}

impl<Loader, Scalar, Index> MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh> + Default,
          Scalar: RealField + From<f32>,
          Index: Copy + TryInto<usize> + TryFrom<usize>
{
    pub fn with_extensions(extensions: &'static [&'static str]) -> Self {
        Self::with_loader_and_extensions(Default::default(), extensions)
    }
}

impl<Loader, Scalar, Index> AssetLoader for MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh>,
          Scalar: RealField + PartialOrd + From<f32> + TypePath,
          Index: Copy + Hash + Ord + TryInto<usize> + TryFrom<usize> + TypePath + Send + Sync + 'static
{
    type Asset = MeshShape<Scalar, Index>;
    type Settings = Loader::Settings;
    type Error = MeshShapeLoadingError<Loader::Error>;

    fn load<'a>(&'a self, reader: &'a mut Reader, settings: &'a Self::Settings, load_context: &'a mut LoadContext) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mesh = self.loader.load(reader, settings, load_context).await.map_err(|err| MeshShapeLoadingError::Loading {
                path: load_context.asset_path().clone(),
                load_error: err,
            })?;
            MeshShape::try_from(&mesh).map_err(|err| MeshShapeLoadingError::Conversion {
                path: load_context.asset_path().clone(),
                conversion_error: err,
            })
        })
    }

    fn extensions(&self) -> &[&str] {
        self.extensions.unwrap_or_else(|| self.loader.extensions())
    }
}
