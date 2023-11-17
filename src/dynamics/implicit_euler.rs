use bevy_ecs::component::Component;
use nalgebra::RealField;
use crate::dynamics::AffineTransform;

#[derive(Component, Debug)]
pub struct State<Scalar: RealField> {
    current: AffineTransform<Scalar>,
    derivative: AffineTransform<Scalar>,
}


