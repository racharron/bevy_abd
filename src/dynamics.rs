use bevy_ecs::component::Component;
use nalgebra::{Matrix3, Matrix4, MatrixView3, RealField, Vector3, VectorView3};

#[cfg(test)]
mod tests;
mod implicit_euler;
mod moments;

pub use implicit_euler::State;

pub(crate) use moments::Moments;

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

}
