use std::collections::HashMap;
use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use nalgebra::RealField;
use crate::dynamics::ColliderInternal;
use crate::dynamics::AffineBody;


pub struct Islands {
    quantity: usize,
    map: HashMap<Entity, usize>
}

pub fn quadratic<Scalar: RealField + TypePath>(
    colliders: Query<(Entity, &AffineBody<Scalar>, &ColliderInternal<Scalar>)>
) {
    for (o0, b0, c0) in &colliders {
        for (o1, b1, c1) in colliders.iter().filter(|(o1, _, _)| &o0 != o1) {
            todo!()
        }
    }
}


