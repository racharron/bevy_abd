use nalgebra::{Point3, RealField, Vector3};
use std::error::Error;
use std::marker::PhantomData;
use argmin_math::ArgminDot;
use bevy_asset::{Asset, AssetLoader, AssetPath, BoxedFuture, LoadContext};
use bevy_asset::io::Reader;
use bevy_reflect::TypePath;
use bevy_render::mesh::{Indices, Mesh};
use thiserror::Error;
use crate::dynamics::Moments;

/// There are no degenerate triangles in the triangle strip.
#[derive(TypePath, Debug)]
pub struct MeshShape<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> {
    vertices: Vec<Point3<Scalar>>,
    /// If empty, assume that the there is a single triangle strip consisting of
    /// the vertices in-order.
    indices: Vec<Index>,
    /// Indicates the boundaries between the triangle strips in the indexes
    strip_boundaries: Vec<u32>,
    /// The moments of inertia (including volume)
    moments: Moments<Scalar>,
}

impl<Scalar, Index> Asset for MeshShape<Scalar, Index>
where Scalar: RealField + PartialOrd + TypePath, Index: Copy + Into<usize> + TypePath + Send + Sync {}

impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> bevy_asset::VisitAssetDependencies for MeshShape<Scalar, Index> { fn visit_dependencies(&self, _visit: &mut impl FnMut(bevy_asset::UntypedAssetId)) {} }
/*
impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> bevy_reflect::TypePath for MeshShape<Scalar, Index>
    where ( Scalar ): bevy_reflect::TypePath, ( Index ): bevy_reflect::TypePath,
{
    fn type_path() -> &'static str {
        static CELL: bevy_reflect::utility::GenericTypePathCell = bevy_reflect::utility::GenericTypePathCell::new();
        CELL.get_or_insert::<Self, _>(||
            concat!(concat!(concat!(module_path!(), "::" ), "MeshShape" ), "<" ).to_string() + Scalar::type_path() + ", " + Index::type_path() + ">"
        )
    }
    fn short_type_path() -> &'static str {
        static CELL: bevy_reflect::utility::GenericTypePathCell = bevy_reflect::utility::GenericTypePathCell::new();
        CELL.get_or_insert::<Self, _>(|| "MeshShape<".to_string() + &(TypePath::short_type_path()).to_string() + ", " + Index::short_type_path() + ">")
    }
    fn type_ident() -> Option<&'static str> { Some("MeshShape") }
    fn crate_name() -> Option<&'static str> {
        Some(module_path!().split(':').next().unwrap())
    }
    fn module_path() -> Option<&'static str> { Some(module_path!()) }
}
*/
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
    /// Indicates an a-b-a pattern in the mesh.  This is invalid, and not handled (at the moment).
    ///
    /// This variant carries the position of the index (or point for meshes without index lists) where
    /// the degenerate triangle began
    #[error("Invalid ABA degeneracy at position {0}")]
    InvalidDegeneracy(usize),
}

#[derive(Debug, Error)]
pub enum MeshShapeLoadingError<LE: Error> {
    #[error("Could not convert mesh from file `{path}`: {0}")]
    Conversion {
        path: AssetPath<'static>,
        #[source]
        conversion_error: MeshShapeConversionError
    },
    #[error("Could not load mesh file from `{path}`: {load_error}")]
    Loading {
        path: AssetPath<'static>,
        #[source]
        load_error: LE,
    },
}

impl<LE: Error> MeshShapeLoadingError<LE> {
    pub fn path(&self) -> &AssetPath<'static> {
        match self {
            MeshShapeLoadingError::Conversion { path, .. } => path,
            MeshShapeLoadingError::Loading { path, .. } => path,
        }
    }
}

pub fn moments<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>>(
    vertices: &[Point3<Scalar>],
    indices: &[Index],
    strip_boundaries: &[u32],
) -> Moments<Scalar> {
    let m0 =
        accumulate_tet_property::<_, Index>(vertices, indices, strip_boundaries, |a, b, c| a.dot(&b.cross(&c)))
            / Scalar::from_u32(6).unwrap();
    let four = Scalar::from_u32(4).unwrap();
    let two = Scalar::from_u32(2).unwrap();
    let ten = Scalar::from_u32(10).unwrap();
    let twenty = Scalar::from_u32(20).unwrap();
    let mx =
        accumulate_tet_property::<_, Index>(
            vertices,
            indices,
            strip_boundaries,
            |a, b, c| a.x.clone() + b.x.clone() + c.x.clone(),
        ) * m0.clone() / four.clone();
    let my =
        accumulate_tet_property::<_, Index>(
            vertices,
            indices,
            strip_boundaries,
            |a, b, c| a.y.clone() + b.y.clone() + c.y.clone(),
        ) * m0.clone() / four.clone();
    let mz =
        accumulate_tet_property::<_, Index>(
            vertices,
            indices,
            strip_boundaries,
            |a, b, c| a.z.clone() + b.z.clone() + c.z.clone(),
        ) * m0.clone() / four;
    let mxy =
        accumulate_tet_property::<_, Index>(
            vertices,
            indices,
            strip_boundaries,
            |a, b, c| a.x.clone() * (two.clone() * a.y.clone() + b.y.clone() + c.y.clone())
                + b.x.clone() * (a.y.clone() + two.clone() * b.y.clone() + c.y.clone())
                + c.x.clone() * (a.y.clone() + b.y.clone() + two.clone() * c.y.clone()),
        ) * m0.clone() / twenty.clone();
    let mxz =
        accumulate_tet_property::<_, Index>(
            vertices,
            indices,
            strip_boundaries,
            |a, b, c| a.x.clone() * (two.clone() * a.z.clone() + b.z.clone() + c.z.clone())
                + b.x.clone() * (a.z.clone() + two.clone() * b.z.clone() + c.z.clone())
                + c.x.clone() * (a.z.clone() + b.z.clone() + two.clone() * c.z.clone()),
        ) * m0.clone() / twenty.clone();
    let myz =
        accumulate_tet_property::<_, Index>(
            vertices,
            indices,
            strip_boundaries,
            |a, b, c| a.z.clone() * (two.clone() * a.y.clone() + b.y.clone() + c.y.clone())
                + b.z.clone() * (a.y.clone() + two.clone() * b.y.clone() + c.y.clone())
                + c.z.clone() * (a.y.clone() + b.y.clone() + two.clone() * c.y.clone()),
        ) * m0.clone() / twenty;
    let mx2 = accumulate_tet_property::<_, Index>(
        vertices,
        indices,
        strip_boundaries,
        |a, b, c| a.x.clone().powi(2) + b.x.clone().powi(2) + c.x.clone().powi(2)
            + a.x.clone() * b.x.clone() + a.x.clone() * c.x.clone() + b.x.clone() * c.x.clone(),
    ) * m0.clone() / ten.clone();
    let my2 = accumulate_tet_property::<_, Index>(
        vertices,
        indices,
        strip_boundaries,
        |a, b, c| a.y.clone().powi(2) + b.y.clone().powi(2) + c.y.clone().powi(2)
            + a.y.clone() * b.y.clone() + a.y.clone() * c.y.clone() + b.y.clone() * c.y.clone(),
    ) * m0.clone() / ten.clone();
    let mz2 = accumulate_tet_property::<_, Index>(
        vertices,
        indices,
        strip_boundaries,
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

fn accumulate_tet_property<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>>(
    vertices: &[Point3<Scalar>],
    indices: &[Index],
    strip_boundaries: &[u32],
    f: impl Clone + Fn(Vector3<Scalar>, Vector3<Scalar>, Vector3<Scalar>) -> Scalar,
) -> Scalar {
    let mut terms =
        Vec::with_capacity(indices.len() - 2 * (1 + strip_boundaries.len()));
    let mut indices = indices;
    for next_start in strip_boundaries.iter().cloned() {
        let (current, next) = indices.split_at(next_start as usize);
        acc_strip(vertices, current, &mut terms, f.clone());
        indices = next;
    }
    terms.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sum_vec = Vec::with_capacity(terms.capacity() / 2 + 1);
    while terms.len() > 1 {
        for pair in terms.chunks(2) {
            match pair {
                [a, b] => sum_vec.push(a.clone() + b.clone()),
                [a] => sum_vec.push(a.clone()),
                _ => unreachable!()
            }
        }
        terms.clear();
        std::mem::swap(&mut terms, &mut sum_vec)
    }
    terms.into_iter().next().unwrap()
}

fn acc_strip<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>>(
    vertices: &[Point3<Scalar>],
    strip: &[Index],
    terms: &mut Vec<Scalar>,
    f: impl Fn(Vector3<Scalar>, Vector3<Scalar>, Vector3<Scalar>) -> Scalar,
) {
    for tri in strip.windows(3) {
        let a = vertices[tri[0].into()].coords.clone();
        let b = vertices[tri[1].into()].coords.clone();
        let c = vertices[tri[2].into()].coords.clone();
        terms.push(f(a, b, c));
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
        let mut points = Vec::with_capacity(vertices.len());
        let mut indices = Vec::with_capacity(vertices.len());
        let mut boundaries = Vec::new();

        match value.indices() {
            Some(Indices::U16(raw_indices)) => {
                let mut prev = None;
                let mut prev_prev = None;
                for (i, &index) in raw_indices.iter().enumerate() {
                    if Some(index) == prev_prev {
                        return Err(MeshShapeConversionError::InvalidDegeneracy(i - 2));
                    } else if Some(index) != prev {
                        if prev == prev_prev {
                            boundaries.push(i as u32);
                        } else {
                            let p = Point3::from(vertices[index as usize].clone().map(Scalar::from));
                            let index = Index::try_from(index).map_err(|_| MeshShapeConversionError::U16ToIndex(i))?;
                            points.push(p);
                            indices.push(index);
                        }
                    }
                    prev_prev = prev;
                    prev = Some(index);
                }
                let moments = moments(&points[..], &indices[..], &boundaries[..]);
                Ok(MeshShape {
                    vertices: points,
                    indices,
                    strip_boundaries: boundaries,
                    moments,
                })
            }
            Some(Indices::U32(raw_indices)) => {
                let mut prev = None;
                let mut prev_prev = None;
                for (i, &index) in raw_indices.iter().enumerate() {
                    if Some(index) == prev_prev {
                        return Err(MeshShapeConversionError::InvalidDegeneracy(i - 2));
                    } else if Some(index) != prev {
                        if prev == prev_prev {
                            boundaries.push(i as u32);
                        } else {
                            let p = Point3::from(vertices[index as usize].clone().map(Scalar::from));
                            let index = Index::try_from(index).map_err(|_| MeshShapeConversionError::U32ToIndex(i))?;
                            points.push(p.into());
                            indices.push(index);
                        }
                    }
                    prev_prev = prev;
                    prev = Some(index);
                }
                let moments = moments(&points, &indices, &boundaries);
                Ok(MeshShape {
                    vertices: points,
                    indices,
                    strip_boundaries: boundaries,
                    moments,
                })
            }
            None => {
                let mut prev = None;
                let mut prev_prev = None;
                for (i, &p) in vertices.iter().enumerate() {
                    if Some(p) == prev_prev {
                        return Err(MeshShapeConversionError::InvalidDegeneracy(i - 2));
                    } else if Some(p) != prev {
                        if prev == prev_prev {
                            boundaries.push(i as u32);
                        } else {
                            points.push(Point3::from(p.map(Scalar::from)));
                        }
                    }
                    prev_prev = prev;
                    prev = Some(p);
                }
                let moments = moments(&points, &indices, &boundaries);
                Ok(MeshShape {
                    vertices: points,
                    indices,
                    strip_boundaries: boundaries,
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
