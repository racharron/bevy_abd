//! A physics library tightly coupled with Bevy, based of off affine body dynamics.

use std::sync::Arc;
use bevy_asset::{Asset, Handle};
use nalgebra::{Matrix4, Matrix4x3, RealField};
use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;

mod geometry;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct AffineTransform<Scalar: RealField>(Matrix4x3<Scalar>);

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AffineBody<Scalar: RealField> {
    /// The current translation and affine rotation of the body.
    q: AffineTransform<Scalar>,
    /// The change in `q` since the last update.
    dq: AffineTransform<Scalar>,
    /// The inverse mass matrix.
    m_inv: Matrix4<Scalar>,
}

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColliderGeometryTransform<Scalar: RealField>(pub AffineTransform<Scalar>);

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Collider<Scalar: RealField + TypePath, Index: Into<usize> + TypePath + Send + Sync = u16> {
    geometry: Handle<ColliderGeometry<Scalar, Index>>,
    /// The displacement of the furthest point after the collider geometry transform is applied.
    base_radius: Scalar,
}

#[derive(Asset, TypePath, Debug, Hash, PartialEq, Eq)]
pub struct ColliderGeometry<Scalar: RealField + TypePath, Index: Into<usize> + TypePath + Send + Sync = u16> {
    vertices: [Arc<[Scalar]>; 3],
    indexes: Arc<[Index]>,
    /// Indicates the boundaries between the triangle strips in the indexes
    strip_boundaries: Option<Arc<[Index]>>,
}
