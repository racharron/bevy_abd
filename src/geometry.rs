//! Assorted geometric primitives.

mod line_line;
mod seg_seg;
mod tri_point;
#[cfg(test)]
mod tests;

pub mod mesh;


pub use seg_seg::seg_seg_squared_distance;
pub use tri_point::tri_pt_squared_distance;
pub use mesh::MeshShapeConversionError;

/// The distance squared queries are guaranteed to be within this of the actual value.
/// This is fairly conservative, so adding a factor of 10 should be good enough for things
/// like the contact potential.
pub const TOLERANCE: f32 = 1e-5;
/*
pub fn update_meshes<Scalar, Index>(
    meshes: Res<Assets<Mesh>>,
    mut mesh_changes: EventReader<AssetEvent<Mesh>>,
    mut cache: ResMut<MeshCache<Scalar, Index>>,
    changed: Query<&ColliderGeometry<Scalar>, Or<(Changed<ColliderGeometry<Scalar>>, Added<ColliderGeometry<Scalar>>)>>,
    mut errors: EventWriter<MeshShapeConversionError>,
)
    where Scalar: RealField + From<f32>,
          Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32> + Send + Sync + 'static
{
    let mut updated = HashSet::new();
    for collider in changed.iter() {
        if updated.insert(collider.geometry.id()) {
            if let Some(mesh) = meshes.get(&collider.geometry) {
                update_mesh(&mut cache, &mut errors, collider.geometry.clone_weak(), mesh);
            } else {
                cache.queued.insert(collider.geometry.clone_weak());
            }
        }
    }
    for event in mesh_changes.read() {
        match event {
            AssetEvent::Added { id }
            | AssetEvent::LoadedWithDependencies { id }
            | AssetEvent::Modified { id }
            | AssetEvent::Removed { id }
            => if let Some(mesh) = meshes.get(id.clone()) {
                if !updated.contains(id) {
                    updated.insert(id.clone());
                    update_mesh(&mut cache, &mut errors, Handle::Weak(id.clone()), mesh)
                }
            },
        }
    }
}

fn update_mesh<Scalar, Index>(
    cache: &mut ResMut<MeshCache<Scalar, Index>>,
    errors: &mut EventWriter<MeshShapeConversionError>,
    handle: Handle<Mesh>,
    mesh: &Mesh,
)
    where Scalar: RealField + From<f32>, Index: Copy + Into<usize> + TryFrom<u16> + TryFrom<u32> + Send + Sync + 'static
{
    match MeshShape::try_from(mesh) {
        Ok(shape) => {
            cache.mapping.insert(handle, shape);
        }
        Err(err) => {
            errors.send(err);
        }
    }
}
*/