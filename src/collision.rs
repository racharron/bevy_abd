use std::marker::PhantomData;
use bevy_ecs::component::Component;
use nalgebra::RealField;
use bevy_asset::Handle;
use bevy_render::mesh::Mesh;
use crate::dynamics::AffineTransform;

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColliderGeometryTransform<Scalar: RealField>(pub AffineTransform<Scalar>);

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColliderGeometry<Scalar: RealField> {
    pub(crate) geometry: Handle<Mesh>,
    _phantom: PhantomData<Scalar>,
}

#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColliderInternal<Scalar: RealField> {
    /// The displacement of the furthest point after the collider geometry transform (if there is one) is applied.
    pub base_radius: Scalar,
}

