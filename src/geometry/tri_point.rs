use nalgebra::{Point3, RealField};

/// Computes the squared difference between a triangle and a point.
///
/// Taken from
/// (https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf)[https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf].
///
/// Assumes that the triangle is non-degenerate.
pub fn tri_pt_squared_distance<Scalar: RealField + PartialOrd>(
    a: Point3<Scalar>,
    b: Point3<Scalar>,
    c: Point3<Scalar>,
    p: Point3<Scalar>,
) -> Scalar {
    let base = a.clone();
    let edge0 = &b - &a;
    let edge1 = &c - &a;
    let dist = &a - p.clone();
    let a = edge0.magnitude_squared();
    let b = edge0.dot(&edge1);
    let c = edge1.magnitude_squared();
    let d = edge0.dot(&dist);
    let e = edge1.dot(&dist);
    let two = Scalar::one() + Scalar::one();
    let mut s = b.clone() * e.clone() - c.clone() * d.clone();
    let mut t = b.clone() * d.clone() - a.clone() * e.clone();
    let det = a.clone() * c.clone() - b.clone().powi(2);
    if s.clone() + t.clone() <= det {
        if s < Scalar::zero() {
            if t.is_negative() {
                // Region 4
                if d.is_negative() {
                    t = Scalar::zero();
                    if -d.clone() >= a {
                        s = Scalar::one();
                    } else {
                        s = -d / a;
                    }
                } else {
                    s = Scalar::zero();
                    if e >= Scalar::zero() {
                        t = Scalar::zero();
                    } else if -e.clone() >= c {
                        t = Scalar::one();
                    } else {
                        t = -e / c;
                    }
                }
            } else {
                //  Region 3
                s = Scalar::zero();
                if e >= Scalar::zero() {
                    t = Scalar::zero();
                } else if -e.clone() >= c {
                    t = Scalar::one();
                } else {
                    t = -e / c;
                }
            }
        } else if t.is_negative() {
            //  Region 5
            t = Scalar::zero();
            if d >= Scalar::zero() {
                s = Scalar::zero();
            } else if -d.clone() >= a {
                s = Scalar::one();
            } else {
                s = -d / a;
            }
        } else {
            //  Region 0
            s /= det.clone();
            t /= det;
        }
    } else {
        if s.is_negative() {
            //  Region 2
            let tmp0 = b.clone() + d.clone();
            let tmp1 = c.clone() + e.clone();
            if tmp1 > tmp0 {
                let numerator = tmp1 - tmp0;
                let denominator = a.clone() - two * b + c;
                if numerator >= denominator {
                    s = Scalar::one();
                    t = Scalar::zero();
                } else {
                    s = numerator / denominator;
                    t = Scalar::one() - s.clone();
                }
            } else {
                s = Scalar::zero();
                if tmp1 <= Scalar::zero() {
                    t = Scalar::one();
                } else if e.clone() >= Scalar::zero() {
                    t = Scalar::zero();
                } else {
                    t = -e / c;
                }
            }
        } else if t.is_negative() {
            //  Region 6
            let tmp0 = b.clone() + e.clone();
            let tmp1 = a.clone() + d.clone();
            if tmp1 > tmp0 {
                let numerator = tmp1 - tmp0;
                let denominator = a - two * b + c;
                if numerator >= denominator {
                    t = Scalar::one();
                } else {
                    t = numerator / denominator;
                }
                s = Scalar::one() - t.clone();
            } else {
                t = Scalar::zero();
                if tmp1 <= Scalar::zero() {
                    s = Scalar::one();
                } else if d >= Scalar::zero() {
                    s = Scalar::zero();
                } else {
                    s = -d / a;
                }
            }
        } else {
            //  Region 1
            let numerator = (c.clone() + e) - (b.clone() + d);
            if numerator <= Scalar::zero() {
                s = Scalar::zero();
            } else {
                let denominator = a - two * b + c;
                if numerator >= denominator {
                    s = Scalar::one();
                } else {
                    s = numerator / denominator;
                }
            }
            t = Scalar::one() - s.clone();
        }
    }
    (base + edge0.scale(s) + edge1.scale(t) - p).magnitude_squared()
}
