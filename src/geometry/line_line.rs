use nalgebra::{Point3, RealField};

/// Computes the weighing factor between two non-degenerate, non-parallel
///
/// Assumes that the segments are non-degenerate.
///
/// Based off this stack overflow answer: [https://math.stackexchange.com/a/4347294](https://math.stackexchange.com/a/4347294).
pub fn line_line_weights<Scalar: RealField + PartialOrd>(a0: Point3<Scalar>, a1: Point3<Scalar>, b0: Point3<Scalar>, b1: Point3<Scalar>) -> [Scalar; 2] {
    // let [x1,y1,z1] = a0.into();
    // let [x2,y2,z2] = a1.into();
    // let [x3,y3,z3] = b0.into();
    // let [x4,y4,z4] = b1.into();
    // let [x21, y21, z21] = a1.clone() - a0.clone();
    let va = a1.clone() - a0.clone();
    // let [x43, y43, z43] = b1.clone() - b0.clone();
    let vb = b1.clone() - b0.clone();
    // let [x31, y31, z31] = b0.clone() - a0.clone();
    let d = b0.clone() - a0.clone();
    let r1 = va.clone().magnitude_squared();
    let r2 = vb.clone().magnitude_squared();
    let d4321 = va.dot(&vb);
    let d3121 = va.dot(&d);
    let d4331 = vb.dot(&d);
    let den = d4321.clone().powi(2) - r1.clone()* r2.clone();
    //  R1*s - d4321*t - d3121 = 0
    //  d4321*s - R2*t - d4331 = 0
    let s = (d4321.clone()*d4331.clone() - r2.clone()*d3121.clone()) / den.clone();
    let t = (r1 *d4331 - d4321*d3121) / den;
    // ps = l1.p1 + s*l1.direction
    // pt = l2.p1 + t*l2.direction
    // return s, t, ps, pt, ps.distance(pt)
    [s, t]
}