//! A physics library tightly coupled with Bevy, based of off affine body dynamics.


use std::marker::PhantomData;
use bevy_app::{App, Plugin};
use nalgebra::RealField;

mod geometry;
mod broad_phase;
mod dynamics;


pub use dynamics::*;

pub struct PhysicsPlugin<Scalar: RealField + PartialOrd, Index: Copy + Into<usize>> {
    _phantom: PhantomData<fn() -> (Scalar, Index)>
}

impl<Scalar: RealField + PartialOrd, Index: Copy + Into<usize> + Send + Sync + 'static> Plugin for PhysicsPlugin<Scalar, Index> {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(geometry::MeshCache::<Scalar, Index>::new());
    }
}
