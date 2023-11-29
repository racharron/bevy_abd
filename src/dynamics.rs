use std::hash::Hash;
use bevy_ecs::component::Component;
use nalgebra::{Matrix3, Matrix4, MatrixView3, Point3, RealField, Vector3, VectorView3};
use bevy_asset::Handle;

#[cfg(test)]
mod tests;
mod implicit_euler;
mod moments;

pub use implicit_euler::State;

pub(crate) use moments::Moments;
use bevy_reflect::TypePath;
use crate::geometry::mesh::ColliderMesh;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct AffineTransform<Scalar: RealField> {
    linear_transform: Matrix3<Scalar>,
    translation: Vector3<Scalar>,
}

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AffineBody<Scalar: RealField> {
    /// The current translation and affine rotation of the body.
    q: AffineTransform<Scalar>,
    /// The change in `q` since the last update.
    dq: AffineTransform<Scalar>,
    /// The inverse mass matrix.
    m_inv: Matrix4<Scalar>,
}

impl<Scalar: RealField> AffineTransform<Scalar> {
    pub fn linear_transform(&self) -> MatrixView3<Scalar> {
        self.linear_transform.as_view()
    }
    pub fn translation(&self) -> VectorView3<Scalar> {
        self.translation.as_view()
    }
    pub fn apply(&self, p: Point3<Scalar>) -> Point3<Scalar> {
        self.linear_transform() * p + self.translation()
    }
}

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColliderGeometryTransform<Scalar: RealField>(pub AffineTransform<Scalar>);

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColliderGeometry<Scalar: RealField + PartialOrd + TypePath, Index: Copy + Hash + TryInto<usize> + TypePath + Send + Sync> {
    pub(crate) geometry: Handle<ColliderMesh<Scalar, Index>>,
}

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColliderInternal<Scalar: RealField> {
    /// The displacement of the furthest point after the collider geometry transform (if there is one) is applied.
    pub base_radius: Scalar,
}
