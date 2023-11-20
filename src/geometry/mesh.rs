use nalgebra::{Point3, RealField, Vector3};
use std::error::Error;
use std::marker::PhantomData;
use bevy_asset::{Asset, AssetLoader, AssetPath, BoxedFuture, LoadContext};
use bevy_asset::io::Reader;
use bevy_reflect::TypePath;
use bevy_render::mesh::{Indices, Mesh};
use thiserror::Error;
use crate::dynamics::Moments;

#[derive(TypePath, Debug)]
pub struct TriangleStrips<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> {
    vertices: Vec<Point3<Scalar>>,
    /// Does not include the last strip.
    pub(super) strips: Vec<StripData>,
    /// Indicates if the last triangle strip needs to have its orientation flipped.
    flip_last: bool,
    pub(super) indices: Option<Vec<Index>>
}

/// There are no degenerate triangles in the triangle strip.
#[derive(TypePath, Debug)]
pub struct MeshShape<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> {
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
    #[error("Mesh has no vertex position attribute")]
    NoPositions,
    #[error("Could not convert vertex positions to floats")]
    NoFloatVertices,
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

impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> TriangleStrips<Scalar, Index> {
    /// Returns the (unscaled) moments of inertia of the strip.
    ///
    /// Unscaled means that the density of the object is assumed to be 1, and thus this will
    /// need to be scaled by the actual density of the object.

    pub fn moments(&self) -> Moments<Scalar> {
        let m0 =
            self.accumulate_tet_property(|a, b, c| a.dot(&b.cross(&c)))
                / Scalar::from_u32(6).unwrap();
        let four = Scalar::from_u32(4).unwrap();
        let two = Scalar::from_u32(2).unwrap();
        let ten = Scalar::from_u32(10).unwrap();
        let twenty = Scalar::from_u32(20).unwrap();
        let mx =
            self.accumulate_tet_property(
                |a, b, c| a.x.clone() + b.x.clone() + c.x.clone(),
            ) * m0.clone() / four.clone();
        let my =
            self.accumulate_tet_property(
                |a, b, c| a.y.clone() + b.y.clone() + c.y.clone(),
            ) * m0.clone() / four.clone();
        let mz =
            self.accumulate_tet_property(
                |a, b, c| a.z.clone() + b.z.clone() + c.z.clone(),
            ) * m0.clone() / four;
        let mxy =
            self.accumulate_tet_property(
                |a, b, c| a.x.clone() * (two.clone() * a.y.clone() + b.y.clone() + c.y.clone())
                    + b.x.clone() * (a.y.clone() + two.clone() * b.y.clone() + c.y.clone())
                    + c.x.clone() * (a.y.clone() + b.y.clone() + two.clone() * c.y.clone()),
            ) * m0.clone() / twenty.clone();
        let mxz =
            self.accumulate_tet_property(
                |a, b, c| a.x.clone() * (two.clone() * a.z.clone() + b.z.clone() + c.z.clone())
                    + b.x.clone() * (a.z.clone() + two.clone() * b.z.clone() + c.z.clone())
                    + c.x.clone() * (a.z.clone() + b.z.clone() + two.clone() * c.z.clone()),
            ) * m0.clone() / twenty.clone();
        let myz =
            self.accumulate_tet_property(
                |a, b, c| a.z.clone() * (two.clone() * a.y.clone() + b.y.clone() + c.y.clone())
                    + b.z.clone() * (a.y.clone() + two.clone() * b.y.clone() + c.y.clone())
                    + c.z.clone() * (a.y.clone() + b.y.clone() + two.clone() * c.y.clone()),
            ) * m0.clone() / twenty;
        let mx2 = self.accumulate_tet_property(
            |a, b, c| a.x.clone().powi(2) + b.x.clone().powi(2) + c.x.clone().powi(2)
                + a.x.clone() * b.x.clone() + a.x.clone() * c.x.clone() + b.x.clone() * c.x.clone(),
        ) * m0.clone() / ten.clone();
        let my2 = self.accumulate_tet_property(
            |a, b, c| a.y.clone().powi(2) + b.y.clone().powi(2) + c.y.clone().powi(2)
                + a.y.clone() * b.y.clone() + a.y.clone() * c.y.clone() + b.y.clone() * c.y.clone(),
        ) * m0.clone() / ten.clone();
        let mz2 = self.accumulate_tet_property(
            |a, b, c| a.z.clone().powi(2) + b.z.clone().powi(2) + c.z.clone().powi(2)
                + a.z.clone() * b.z.clone() + a.z.clone() * c.z.clone() + b.z.clone() * c.z.clone(),
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
    /// Returns the triangles properly oriented.
    pub fn oriented_triangles<'a>(&'a self) -> impl 'a + Iterator<Item=[Point3<Scalar>; 3]> {
        match &self.indices {
            Some(indices) => {
                let mut indices = &indices[..];
                let mut strips = &self.strips[..];
                either::Right(std::iter::from_fn(move || {
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
                            [self.vertices[tri[1].clone().into()].clone(), self.vertices[tri[0].clone().into()].clone(), self.vertices[tri[2].clone().into()].clone()]
                        } else {
                            [self.vertices[tri[0].clone().into()].clone(), self.vertices[tri[1].clone().into()].clone(), self.vertices[tri[2].clone().into()].clone()]
                        };
                        flip = !flip;
                        tri
                    }))
                }).flatten())
            }
            None => {
                let mut vertices = &self.vertices[..];
                let mut strips = &self.strips[..];
                either::Left(std::iter::from_fn(move || {
                    let (current, mut flip) = if let Some((&StripData { length, flip }, rest)) = strips.split_first() {
                        strips = rest;
                        let (current, rest) = vertices.split_at(length as usize);
                        vertices = rest;
                        (current, flip)
                    } else if vertices.is_empty() {
                        return None
                    } else {
                        let current = vertices;
                        vertices = &vertices[0..0];
                        (current, self.flip_last)
                    };
                    Some(current.windows(3).map(move |tri| {
                        let tri = if flip {
                            [tri[1].clone(), tri[0].clone(), tri[2].clone()]
                        } else {
                            [tri[0].clone(), tri[1].clone(), tri[2].clone()]
                        };
                        flip = !flip;
                        tri
                    }))
                }).flatten())
            },
        }
    }

    fn accumulate_tet_property(
        &self,
        f: impl Clone + Fn(Vector3<Scalar>, Vector3<Scalar>, Vector3<Scalar>) -> Scalar,
    ) -> Scalar {
        let tri_count = if let Some(indices) = &self.indices {
            indices.len() - 2 * (1 + self.strips.len())
        } else {
            self.vertices.len() - 2
        };
        let mut terms = Vec::with_capacity(tri_count);
        terms.extend(self.oriented_triangles().map(|[a, b, c]| f(a.coords.clone(), b.coords.clone(), c.coords.clone())));
        //  Improved Kahan–Babuška compensated summation algorithm
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
}

impl<Scalar, Index> Asset for MeshShape<Scalar, Index>
    where Scalar: RealField + PartialOrd + TypePath, Index: Copy + Into<usize> + TypePath + Send + Sync {}

impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> bevy_asset::VisitAssetDependencies for MeshShape<Scalar, Index> {
    fn visit_dependencies(&self, _: &mut impl FnMut(bevy_asset::UntypedAssetId)) {}
}

impl<LE: Error> MeshShapeLoadingError<LE> {
    pub fn path(&self) -> &AssetPath<'static> {
        match self {
            MeshShapeLoadingError::Conversion { path, .. } => path,
            MeshShapeLoadingError::Loading { path, .. } => path,
        }
    }
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> MeshShape<Scalar, Index> {
    pub fn moments(&self) -> &Moments<Scalar> {
        &self.moments
    }
    pub fn oriented_triangles<'a>(&'a self) -> impl 'a + Iterator<Item=[Point3<Scalar>; 3]> {
       self.triangle_strips.oriented_triangles()
    }
}

impl<Scalar, Index> TryFrom<&Mesh> for MeshShape<Scalar, Index>
    where Scalar: RealField + From<f32>,
          Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32>
{
    type Error = MeshShapeConversionError;
    fn try_from(value: &Mesh) -> Result<Self, Self::Error> {
        let vertices = value.attribute(Mesh::ATTRIBUTE_POSITION).ok_or(MeshShapeConversionError::NoPositions)?
            .as_float3().ok_or(MeshShapeConversionError::NoFloatVertices)?;
        let mut strips = Vec::new();
        let mut global_flip = false;
        let mut current_flip = global_flip;
        let mut stored_prev = false;
        let mut length = 0;

        match value.indices() {
            Some(Indices::U16(raw_indices)) => {
                let mut indices = Vec::with_capacity(raw_indices.len());
                let mut prev = None;
                let mut stored_prev = false;
                let mut prev_prev = None;
                for (i, &index) in raw_indices.iter().enumerate() {
                    let stored;
                    if Some(index) == prev || Some(index) == prev_prev || prev == prev_prev {
                        stored = false;
                    } else if let [Some(prev), Some(prev_prev)] = [prev, prev_prev] {
                        let index = Index::try_from(index).map_err(|_| MeshShapeConversionError::U16ToIndex(i))?;
                        if !stored_prev {
                            if length != 0 {
                                strips.push(StripData { length, flip: current_flip });
                                length = 0;
                            }
                            indices.push(Index::try_from(prev_prev).map_err(|_| MeshShapeConversionError::U16ToIndex(i))?);
                            indices.push(Index::try_from(prev).map_err(|_| MeshShapeConversionError::U16ToIndex(i))?);
                            length += 2;
                            current_flip = global_flip;
                        }
                        indices.push(index);
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
                let vertices: Vec<_> = vertices.into_iter().map(|&p| Point3::from(p.map(Scalar::from))).collect();
                let strips = TriangleStrips {
                    vertices,
                    strips,
                    flip_last: current_flip,
                    indices: Some(indices),
                };
                let moments = strips.moments();
                Ok(MeshShape {
                    triangle_strips: strips,
                    moments,
                })
            }
            Some(Indices::U32(raw_indices)) => {
                let mut indices = Vec::with_capacity(raw_indices.len());
                let mut prev = None;
                let mut prev_prev = None;
                for (i, &index) in raw_indices.iter().enumerate() {
                    let stored;
                    if Some(index) == prev || Some(index) == prev_prev || prev == prev_prev {
                        stored = false;
                    } else if let [Some(prev), Some(prev_prev)] = [prev, prev_prev] {
                        let index = Index::try_from(index).map_err(|_| MeshShapeConversionError::U32ToIndex(i))?;
                        if !stored_prev {
                            if length != 0 {
                                strips.push(StripData { length, flip: current_flip });
                                length = 0;
                            }
                            indices.push(Index::try_from(prev_prev).map_err(|_| MeshShapeConversionError::U32ToIndex(i))?);
                            indices.push(Index::try_from(prev).map_err(|_| MeshShapeConversionError::U32ToIndex(i))?);
                            length += 2;
                            current_flip = global_flip;
                        }
                        indices.push(index);
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
                let vertices: Vec<_> = vertices.into_iter().map(|&p| Point3::from(p.map(Scalar::from))).collect();
                let strips = TriangleStrips {
                    vertices,
                    strips,
                    flip_last: current_flip,
                    indices: Some(indices),
                };
                let moments = strips.moments();
                Ok(MeshShape {
                    triangle_strips: strips,
                    moments,
                })
            }
            None => {
                let mut prev = None;
                let mut prev_prev = None;
                let mut tri_vertices = Vec::with_capacity(vertices.len());
                for &p in vertices.iter() {
                    let stored;
                    if Some(p) == prev || Some(p) == prev_prev || prev == prev_prev {
                        length = 0;
                        stored = false;
                    } else if let [Some(prev), Some(prev_prev)] = [prev, prev_prev] {
                        if !stored_prev {
                            if length != 0 {
                                strips.push(StripData { length, flip: current_flip });
                                length = 0;
                            }
                            tri_vertices.push(Point3::from(prev_prev.map(Scalar::from)));
                            tri_vertices.push(Point3::from(prev.map(Scalar::from)));
                            length += 2;
                            current_flip = global_flip;
                        }
                        tri_vertices.push(Point3::from(p.map(Scalar::from)));
                        stored = true;
                        length += 1;
                    } else {
                        stored = false;
                    }
                    prev_prev = prev;
                    prev = Some(p);
                    stored_prev = stored;
                    global_flip = !global_flip;
                }
                let strips = TriangleStrips {
                    vertices: tri_vertices,
                    strips,
                    flip_last: current_flip,
                    indices: None,
                };
                let moments = strips.moments();
                Ok(MeshShape {
                    triangle_strips: strips,
                    moments,
                })
            }
        }
    }
}

/// This loader requires a loader that loads a mesh, and calls t
pub struct MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh>,
          Scalar: RealField + From<f32>,
          Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32>
{
    loader: Loader,
    extensions: Option<&'static [&'static str]>,
    _phantom: PhantomData<fn() -> (Scalar, Index)>,
}

impl<Loader, Scalar, Index> MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh>,
          Scalar: RealField + From<f32>,
          Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32>
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
          Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32>
{
    fn default() -> Self {
        Self::with_loader(Loader::default())
    }
}

impl<Loader, Scalar, Index> MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh> + Default,
          Scalar: RealField + From<f32>,
          Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32>
{
    pub fn with_extensions(extensions: &'static [&'static str]) -> Self {
        Self::with_loader_and_extensions(Default::default(), extensions)
    }
}

impl<Loader, Scalar, Index> AssetLoader for MeshShapeLoader<Loader, Scalar, Index>
    where Loader: AssetLoader<Asset=Mesh>,
          Scalar: RealField + PartialOrd + From<f32> + TypePath,
          Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32> + TypePath + Send + Sync + 'static
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
