use super::*;
use itertools::Itertools;
use nalgebra::{Point3, Rotation3, Vector3};
use crate::geometry::mesh::ColliderMesh;

const LENGTHS: [f32; 3] = [0.5, 1.0, 2.0];
const OFFSETS: [f32; 7] = [0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0];

const ANGLES: [f32; 6] = [0.0, 0.1, 0.5, 0.9, 1.4, 1.5];


#[test]
fn tmp() {
    for n in -16..0 {
        println!("{n}: {}, {}", n >> 1, n / 2);
    }
    assert!(false)
}


#[test]
fn simple_cuboid_moments() {
    const LENGTHS: [f32; 4] = [0.5, 1.0, 1.5, 2.0];
    for ((x_length, y_length), z_length) in LENGTHS.into_iter().cartesian_product(LENGTHS).cartesian_product(LENGTHS) {
        let cube = ColliderMesh::<f32, u16>::cuboid(x_length, y_length, z_length);
        println!("{:#?}", cube.mesh);
        println!();
        assert_eq!(cube.moments().v, x_length * y_length * z_length);
        assert_eq!(cube.moments().x, 0.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().y, 0.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().z, 0.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().xy, 0.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().yz, 0.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().xz, 0.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().xx, x_length.powi(3) * y_length * z_length / 12.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().yy, x_length * y_length.powi(3) * z_length / 12.0, "({x_length}, {y_length}, {z_length})");
        assert_eq!(cube.moments().zz, x_length * y_length * z_length.powi(3) / 12.0, "({x_length}, {y_length}, {z_length})");
    }
}

#[test]
fn tri_moments() {
    const LENGTHS: [f32; 8] = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0];
    for ((xlen, ylen), zlen) in LENGTHS.into_iter().cartesian_product(LENGTHS).cartesian_product(LENGTHS) {
        let mut vertices = [[xlen, 0.0, 0.0], [0.0, ylen, 0.0], [0.0, 0.0, zlen]];
        let tri = ColliderMesh::<f32, u16>::from_triangle_strip(&vertices).unwrap();
        vertices.swap(0, 1);
        let neg_tri = ColliderMesh::<f32, u16>::from_triangle_strip(&vertices).unwrap();
        assert_eq!(tri.moments().v, -neg_tri.moments().v);
        assert_eq!(tri.moments().v, xlen * ylen * zlen / 6.0);
        fn m100(xlen: f32, ylen: f32, zlen: f32) -> f32 {
            xlen.powi(2) * ylen * zlen / 24.0
        }
        assert_eq!(tri.moments().x, -neg_tri.moments().x);
        assert_eq!(tri.moments().x, m100(xlen, ylen, zlen));
        assert_eq!(tri.moments().y, -neg_tri.moments().y);
        assert_eq!(tri.moments().y, m100(ylen, xlen, zlen));
        assert_eq!(tri.moments().z, -neg_tri.moments().z);
        assert_eq!(tri.moments().z, m100(zlen, ylen, xlen));
        fn m110(xlen: f32, ylen: f32, zlen: f32) -> f32 {
            xlen.powi(2) * ylen.powi(2) * zlen / 120.0
        }
        assert_eq!(tri.moments().xy, -neg_tri.moments().xy);
        assert!((1.0 - tri.moments().xy / m110(xlen, ylen, zlen)).abs() < 1e-6, "{} != {}", tri.moments().xy, m110(xlen, ylen, zlen));
        assert_eq!(tri.moments().xz, -neg_tri.moments().xz);
        assert!((1.0 - tri.moments().xz / m110(xlen, zlen, ylen)).abs() < 1e-6, "{} != {}", tri.moments().xz, m110(xlen, zlen, ylen));
        assert_eq!(tri.moments().yz, -neg_tri.moments().yz);
        assert!((1.0 - tri.moments().yz / m110(ylen, zlen, xlen)).abs() < 1e-6, "{} != {}", tri.moments().yz, m110(ylen, zlen, xlen));
        fn m200(xlen: f32, ylen: f32, zlen: f32) -> f32 {
            xlen.powi(3) * ylen * zlen / 60.0
        }
        assert_eq!(tri.moments().xx, -neg_tri.moments().xx);
        assert!((1.0 - tri.moments().xx / m200(xlen, ylen, zlen)).abs() < 1e-6, "{} != {}", tri.moments().xx, m200(xlen, ylen, zlen));
        assert_eq!(tri.moments().yy, -neg_tri.moments().yy);
        assert!((1.0 - tri.moments().yy / m200(ylen, xlen, zlen)).abs() < 1e-6, "{} != {}", tri.moments().yy, m200(ylen, xlen, zlen));
        assert_eq!(tri.moments().zz, -neg_tri.moments().zz);
        assert!((1.0 - tri.moments().zz / m200(zlen, ylen, xlen)).abs() < 1e-6, "{} != {}", tri.moments().zz, m200(zlen, ylen, xlen));
    }
}

#[test]
fn flat_at_origin() {
    const COORDS: [f32; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let planar = COORDS.into_iter().cartesian_product(COORDS);
    for (a, b, c) in planar.clone().cartesian_product(planar.clone()).cartesian_product(planar)
        .filter_map(|((a, b), c)| {
            if a == b || a == c || b == c {
                None
            } else {
                Some((a, b, c))
            }
        })
    {
        for (a, b, c) in [
            ([a.0, a.1, 0.0], [b.0, b.1, 0.0], [c.0, c.1, 0.0]),
            ([a.0, 0.0, a.1], [b.0, 0.0, b.1], [c.0, 0.0, c.1]),
            ([0.0, a.0, a.1], [0.0, b.0, b.1], [0.0, c.0, c.1]),
        ] {
            for tri in [
                ColliderMesh::<f32, u16>::from_triangle_strip([a, b, c]).unwrap(),
                ColliderMesh::from_triangle_strip([b, a, c]).unwrap(),
                ColliderMesh::from_indexed_triangle_strip([a, b, c], [0, 1, 2]).unwrap(),
                ColliderMesh::from_indexed_triangle_strip([a, b, c], [1, 0, 2]).unwrap(),
            ] {

                let moments = tri.moments();
                assert_eq!(moments.v, 0.0);
                assert_eq!(moments.x, 0.0);
                assert_eq!(moments.y, 0.0);
                assert_eq!(moments.z, 0.0);
                assert_eq!(moments.xx, 0.0);
                assert_eq!(moments.yy, 0.0);
                assert_eq!(moments.zz, 0.0);
                assert_eq!(moments.xy, 0.0);
                assert_eq!(moments.xy, 0.0);
                assert_eq!(moments.yz, 0.0);
            }
        }
    }
}

#[test]
fn points() {
    for (a0, b0) in offsets_3d()
        .map(Point3::from)
        .cartesian_product(offsets_3d().map(Point3::from))
    {
        fn filter_invalid(dir: Vector3<f32>) -> impl Clone + Fn([f32; 3]) -> Option<Vector3<f32>> {
            move |v| {
                if v == [0.0; 3] {
                    None
                } else {
                    let v = Vector3::from(v);
                    if v.dot(&dir) < 0.0 {
                        None
                    } else {
                        Some(v)
                    }
                }
            }
        }
        for (va, vb) in offsets_3d()
            .filter_map(filter_invalid(a0.clone() - b0.clone()))
            .cartesian_product(offsets_3d().filter_map(filter_invalid(b0.clone() - a0.clone())))
        {
            assert_eq!(
                (a0 - b0).magnitude_squared(),
                seg_seg_squared_distance(a0, a0 + va, b0, b0 + vb),
                "{:?}->{:?}, {:?}->{:?}",
                a0,
                a0 + va,
                b0,
                b0 + vb
            );
            assert_eq!(
                (a0 - b0).magnitude_squared(),
                seg_seg_squared_distance(a0 + va, a0, b0, b0 + vb),
                "{:?}->{:?}, {:?}->{:?}",
                a0 + va,
                a0,
                b0,
                b0 + vb
            );
            assert_eq!(
                (a0 - b0).magnitude_squared(),
                seg_seg_squared_distance(a0, a0 + va, b0 + vb, b0),
                "{:?}->{:?}, {:?}->{:?}",
                a0,
                a0 + va,
                b0 + vb,
                b0
            );
            assert_eq!(
                (a0 - b0).magnitude_squared(),
                seg_seg_squared_distance(a0 + va, a0, b0 + vb, b0),
                "{:?}->{:?}, {:?}->{:?}",
                a0 + va,
                a0,
                b0 + vb,
                b0
            );
        }
    }
}

//  Takes 22 min to run at the moment.
#[test]
fn edges() {
    for (offset, dist) in offsets_3d().map(Point3::from).cartesian_product(LENGTHS) {
        for (a0_dist, a1_dist) in LENGTHS.into_iter().cartesian_product(LENGTHS) {
            for (b0_dist, b1_dist) in LENGTHS.into_iter().cartesian_product(LENGTHS) {
                for basis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    .map(Vector3::from)
                    .into_iter()
                    .permutations(3)
                {
                    let [ab, u, v] = <[_; 3] as TryFrom<_>>::try_from(basis).unwrap();
                    for ((ab, u), v) in [ab, -ab]
                        .into_iter()
                        .cartesian_product([u, -u])
                        .cartesian_product([v, -v])
                    {
                        for (a_angle, b_angle) in ANGLES.into_iter().cartesian_product(ANGLES) {
                            let a_dir =
                                Rotation3::from_scaled_axis(ab.scale(a_angle)).transform_vector(&u);
                            let b_dir =
                                Rotation3::from_scaled_axis(ab.scale(b_angle)).transform_vector(&v);
                            let a0 = offset - a_dir.scale(a0_dist);
                            let a1 = offset + a_dir.scale(a1_dist);
                            let b0 = offset + ab.scale(dist) - b_dir.scale(b0_dist);
                            let b1 = offset + ab.scale(dist) + b_dir.scale(b1_dist);
                            let expected = dist.powi(2);
                            let actual = seg_seg_squared_distance(a0, a1, b0, b1);
                            let diff = (expected - actual).abs();
                            if diff > TOLERANCE {
                                assert_eq!(
                                    expected, actual,
                                    "segs: {:?}->{:?}, {:?}->{:?}, diff: {}",
                                    a0, a1, b0, b1, diff
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn intersect() {
    for offset in offsets_3d().map(Point3::from) {
        for (da, db) in offsets_3d()
            .filter_map(|a| {
                if a == [0.0; 3] || a[0] < 0.0 {
                    None
                } else {
                    Some(Vector3::from(a))
                }
            })
            .cartesian_product(offsets_3d().filter_map(|a| {
                if a == [0.0; 3] || a[0] < 0.0 {
                    None
                } else {
                    Some(Vector3::from(a))
                }
            }))
        {
            let a0 = offset - da;
            let a1 = offset + da;
            let b0 = offset - db;
            let b1 = offset + db;
            let actual = seg_seg_squared_distance(a0, a1, b0, b1);
            let diff = actual.abs();
            if diff > TOLERANCE {
                assert_eq!(
                    0.0, actual,
                    "segs: {:?}->{:?}, {:?}->{:?}, diff: {}",
                    a0, a1, b0, b1, diff
                )
            }
        }
    }
}

#[test]
fn parallel() {
    for (((a0, a_len), b_len), dist) in offsets_3d()
        .map(Point3::from)
        .cartesian_product(LENGTHS)
        .cartesian_product(LENGTHS)
        .cartesian_product(LENGTHS)
    {
        for [v_dir, ab_dir] in extended_points(1.0)
            .filter(|[v, ab]| v.coords.cross(&ab.coords) != Vector3::new(0.0, 0.0, 0.0))
        {
            let v_dir = v_dir.coords;
            let b0 = a0 + ab_dir.coords.scale(dist);
            let a1 = a0 + v_dir.scale(a_len);
            let b1 = b0 + v_dir.scale(b_len);
            assert_eq!(
                dist.powi(2),
                seg_seg_squared_distance(a0, a1, b0, b1),
                "{:?}->{:?}, {:?}->{:?}",
                a0,
                a1,
                b0,
                b1
            )
        }
    }
}

fn offsets_3d() -> impl Iterator<Item=[f32; 3]> + Clone {
    OFFSETS
        .into_iter()
        .cartesian_product(OFFSETS)
        .cartesian_product(OFFSETS)
        .map(|((x, y), z)| [x, y, z])
}

#[test]
fn off_corners() {
    for [len, x, y, z] in parameters() {
        for [t1, t2] in extended_points(len) {
            for [a, b, c] in corners(t1, t2) {
                for p_proj in [
                    -t1,
                    -t2,
                    2.0 * t1,
                    2.0 * t2,
                    Point3::from(t1 - t2),
                    Point3::from(t2 - t1),
                ] {
                    for offset in OFFSETS {
                        let p = p_proj + offset * t1.coords.cross(&t2.coords).normalize();
                        let [a, b, c, p] =
                            [a, b, c, p].map(|p| p + Vector3::new(x, y, z));
                        assert_eq!(
                            len * len + offset * offset,
                            tri_pt_squared_distance(a, b, c, p),
                            "a = {:?}, b = {:?}, c = {:?}, p = {:?}",
                            a,
                            b,
                            c,
                            p
                        )
                    }
                }
            }
        }
    }
}

#[test]
fn off_edges() {
    for [len, x, y, z] in parameters() {
        for [t1, t2] in extended_points(len) {
            let e1 = t1.coords.normalize();
            let e2 = t2.coords.normalize();
            for [a, b, c] in corners(t1, t2) {
                for fraction in 1..4 {
                    let fraction = (fraction as f32) / 4.0;
                    for p_proj in [fraction * t1 - len * e2, fraction * t2 - len * e1] {
                        for offset in OFFSETS {
                            let p = p_proj + offset * e1.cross(&e2);
                            let [a, b, c, p] =
                                [a, b, c, p.into()].map(|p| p + Vector3::new(x, y, z));
                            assert_eq!(
                                len * len + offset * offset,
                                tri_pt_squared_distance(a, b, c, p),
                                "a = {:?}, b = {:?}, c = {:?}, p = {:?}",
                                a,
                                b,
                                c,
                                p
                            )
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn in_tri_prism() {
    for [len, x, y, z] in parameters() {
        for [t1, t2] in extended_points(len) {
            for [a, b, c] in corners(t1, t2) {
                for p_proj in [a, b, c].into_iter().chain([
                    0.5 * t1,
                    0.5 * t2,
                    0.25 * Point3::from(t1.coords + t2.coords),
                ]) {
                    for offset in OFFSETS {
                        let p = p_proj + offset * t1.coords.cross(&t2.coords).normalize();
                        let [a, b, c, p] =
                            [a, b, c, p].map(|p| p + Vector3::new(x, y, z));
                        assert_eq!(
                            offset * offset,
                            tri_pt_squared_distance(a, b, c, p),
                            "a = {:?}, b = {:?}, c = {:?}, p = {:?}",
                            a,
                            b,
                            c,
                            p
                        )
                    }
                }
            }
        }
    }
}

fn corners(t1: Point3<f32>, t2: Point3<f32>) -> impl Iterator<Item=[Point3<f32>; 3]> {
    [[0.0; 3].into(), t1, t2]
        .into_iter()
        .permutations(3)
        .map(|vertices| <[_; 3] as TryFrom<_>>::try_from(vertices).unwrap())
}

fn extended_points(len: f32) -> impl Iterator<Item=[Point3<f32>; 2]> {
    [[len, 0.0, 0.0], [0.0, len, 0.0], [0.0, 0.0, len]]
        .into_iter()
        .flat_map(|p| {
            let p = Point3::from(p);
            [p, -p]
        })
        .permutations(2)
        .filter_map(|acute_corners| {
            let t1 = acute_corners[0];
            let t2 = acute_corners[1];
            if t1.coords.cross(&t2.coords) == Vector3::new(0.0, 0.0, 0.0) {
                None
            } else {
                Some([t1, t2])
            }
        })
}

fn parameters() -> impl Iterator<Item=[f32; 4]> {
    [
        LENGTHS.iter(),
        OFFSETS.iter(),
        OFFSETS.iter(),
        OFFSETS.iter(),
    ]
        .into_iter()
        .map(Iterator::cloned)
        .multi_cartesian_product()
        .map(|parameters| <[_; 4] as TryFrom<_>>::try_from(parameters).unwrap())
}
