use nalgebra::{Point3, RealField};

/// Computes the squared distance between two line segments.
///
/// Assumes that the segments are non-degenerate.
///
/// Translated from [https://www.geometrictools.com/GTE/Mathematics/DistSegmentSegment.h](https://www.geometrictools.com/GTE/Mathematics/DistSegmentSegment.h).
#[allow(non_snake_case)]
pub fn seg_seg_squared_distance<Scalar: RealField + PartialOrd>(
    a0: Point3<Scalar>,
    a1: Point3<Scalar>,
    b0: Point3<Scalar>,
    b1: Point3<Scalar>,
) -> Scalar {
    // The code allows degenerate line segments; that is, P0 and P1
    // can be the same point or Q0 and Q1 can be the same point.  The
    // quadratic function for squared distance between the segment is
    //   R(s,t) = a*s^2 - 2*b*s*t + c*t^2 + 2*d*s - 2*e*t + f
    // for (s,t) in [0,1]^2 where
    //   a = Dot(P1-P0,P1-P0), b = Dot(P1-P0,Q1-Q0), c = Dot(Q1-Q0,Q1-Q0),
    //   d = Dot(P1-P0,P0-Q0), e = Dot(Q1-Q0,P0-Q0), f = Dot(P0-Q0,P0-Q0)
    let P1mP0 = a1.clone() - a0.clone();
    let Q1mQ0 = b1.clone() - b0.clone();
    let P0mQ0 = a0.clone() - b0.clone();
    let a = P1mP0.dot(&P1mP0);
    let b = P1mP0.dot(&Q1mQ0);
    let c = Q1mQ0.dot(&Q1mQ0);
    let d = P1mP0.dot(&P0mQ0);
    let e = Q1mQ0.dot(&P0mQ0);

    // The derivatives dR/ds(i,j) at the four corners of the domain.
    let f00 = d.clone();
    let f10 = f00.clone() + a.clone();
    let f01 = f00.clone() - b.clone();
    let f11 = f10.clone() - b.clone();

    // The derivatives dR/dt(i,j) at the four corners of the domain.
    let g00 = -e.clone();
    let g10 = g00.clone() - b.clone();
    let g01 = g00.clone() + c.clone();
    let g11 = g10.clone() + c.clone();

    let parameter;

    if a > Scalar::zero() && c > Scalar::zero() {
        // Compute the solutions to dR/ds(s0,0) = 0 and
        // dR/ds(s1,1) = 0.  The location of sI on the s-axis is
        // stored in classifyI (I = 0 or 1).  If sI <= 0, classifyI
        // is -1.  If sI >= 1, classifyI is 1.  If 0 < sI < 1,
        // classifyI is 0.  This information helps determine where to
        // search for the minimum point (s,t).  The fij values are
        // dR/ds(i,j) for i and j in {0,1}.

        let sValue = [
            GetClampedRoot(a.clone(), f00.clone(), f10.clone()),
            GetClampedRoot(a.clone(), f01.clone(), f11.clone()),
        ];

        let classify = sValue.clone().map(|value| {
            if value <= Scalar::zero() {
                -1i32
            } else if value >= Scalar::one() {
                1i32
            } else {
                0i32
            }
        });

        parameter = if classify[0] == -1 && classify[1] == -1 {
            // The minimum must occur on s = 0 for 0 <= t <= 1.
            [Scalar::zero(), GetClampedRoot(c, g00, g01)]
        } else if classify[0] == 1 && classify[1] == 1 {
            // The minimum must occur on s = 1 for 0 <= t <= 1.
            [Scalar::one(), GetClampedRoot(c, g10, g11)]
        } else {
            // The line dR/ds = 0 intersects the domain [0,1]^2 in a
            // nondegenerate segment. Compute the endpoints of that
            // segment, end[0] and end[1]. The edge[i] flag tells you
            // on which domain edge end[i] lives: 0 (s=0), 1 (s=1),
            // 2 (t=0), 3 (t=1).
            let (edge, end) = ComputeIntersection(sValue, classify, b.clone(), f00, f10);

            // The directional derivative of R along the segment of
            // intersection is
            //   H(z) = (end[1][1]-end[1][0]) *
            //          dR/dt((1-z)*end[0] + z*end[1])
            // for z in [0,1]. The formula uses the fact that
            // dR/ds = 0 on the segment. Compute the minimum of
            // H on [0,1].
            ComputeMinimumParameters(edge, end, b, c, e, g00, g10, g01, g11)
        };
    } else {
        if a > Scalar::zero() {
            // The Q-segment is degenerate (Q0 and Q1 are the same
            // point) and the quadratic is R(s,0) = a*s^2 + 2*d*s + f
            // and has (half) first derivative F(t) = a*s + d.  The
            // closest P-point is interior to the P-segment when
            // F(0) < 0 and F(1) > 0.
            parameter = [GetClampedRoot(a, f00, f10), Scalar::zero()]
        } else if c > Scalar::zero() {
            // The P-segment is degenerate (P0 and P1 are the same
            // point) and the quadratic is R(0,t) = c*t^2 - 2*e*t + f
            // and has (half) first derivative G(t) = c*t - e.  The
            // closest Q-point is interior to the Q-segment when
            // G(0) < 0 and G(1) > 0.
            parameter = [Scalar::zero(), GetClampedRoot(c, g00, g01)]
        } else {
            // P-segment and Q-segment are degenerate.
            parameter = [Scalar::zero(), Scalar::zero()]
        }
    }

    let a = a0
        .clone()
        .coords
        .scale(Scalar::one() - parameter[0].clone())
        + a1.coords.scale(parameter[0].clone());
    let b = b0
        .clone()
        .coords
        .scale(Scalar::one() - parameter[1].clone())
        + b1.coords.scale(parameter[1].clone());
    (a - b).magnitude_squared()
}

/// Compute the intersection of the line dR/ds = 0 with the domain
/// [0,1]^2. The direction of the line dR/ds is conjugate to (1,0),
/// so the algorithm for minimization is effectively the conjugate
/// gradient algorithm for a quadratic function.
#[allow(non_snake_case)]
fn ComputeIntersection<Scalar: RealField + PartialOrd>(
    sValue: [Scalar; 2],
    classify: [i32; 2],
    b: Scalar,
    f00: Scalar,
    f10: Scalar,
) -> ([i32; 2], [[Scalar; 2]; 2]) {
    // The divisions are theoretically numbers in [0,1]. Numerical
    // rounding errors might cause the result to be outside the
    // interval. When this happens, it must be that both numerator
    // and denominator are nearly zero. The denominator is nearly
    // zero when the segments are nearly perpendicular. The
    // numerator is nearly zero when the P-segment is nearly
    // degenerate (f00 = a is small). The choice of 0.5 should not
    // cause significant accuracy problems.
    //
    // NOTE: You can use bisection to recompute the root or even use
    // bisection to compute the root and skip the division. This is
    // generally slower, which might be a problem for high-performance
    // applications.

    let mut edge = [0; 2];
    let mut end = [
        [Scalar::zero(), Scalar::zero()],
        [Scalar::zero(), Scalar::zero()],
    ];

    if classify[0] < 0 {
        edge[0] = 0;
        end[0][0] = Scalar::zero();
        end[0][1] = f00.clone() / b.clone();
        if end[0][1] < Scalar::zero() || end[0][1] > Scalar::one() {
            end[0][1] = Scalar::from_f32(0.5).unwrap();
        }

        if classify[1] == 0 {
            edge[1] = 3;
            end[1][0] = sValue[1].clone();
            end[1][1] = Scalar::one();
        } else {
            // classify[1] > 0
            edge[1] = 1;
            end[1][0] = Scalar::one();
            end[1][1] = f10.clone() / b.clone();
            if end[1][1] < Scalar::zero() || end[1][1] > Scalar::one() {
                end[1][1] = Scalar::from_f32(0.5).unwrap();
            }
        }
    } else if classify[0] == 0 {
        edge[0] = 2;
        end[0][0] = sValue[0].clone();
        end[0][1] = Scalar::zero();

        if classify[1] < 0 {
            edge[1] = 0;
            end[1][0] = Scalar::zero();
            end[1][1] = f00.clone() / b.clone();
            if end[1][1] < Scalar::zero() || end[1][1] > Scalar::one() {
                end[1][1] = Scalar::from_f32(0.5).unwrap();
            }
        } else if classify[1] == 0 {
            edge[1] = 3;
            end[1][0] = sValue[1].clone();
            end[1][1] = Scalar::one();
        } else {
            edge[1] = 1;
            end[1][0] = Scalar::one();
            end[1][1] = f10.clone() / b.clone();
            if end[1][1] < Scalar::zero() || end[1][1] > Scalar::one() {
                end[1][1] = Scalar::from_f32(0.5).unwrap();
            }
        }
    } else {
        // classify[0] > 0
        edge[0] = 1;
        end[0][0] = Scalar::one();
        end[0][1] = f10.clone() / b.clone();
        if end[0][1] < Scalar::zero() || end[0][1] > Scalar::one() {
            end[0][1] = Scalar::from_f32(0.5).unwrap();
        }

        if classify[1] == 0 {
            edge[1] = 3;
            end[1][0] = sValue[1].clone();
            end[1][1] = Scalar::one();
        } else {
            edge[1] = 0;
            end[1][0] = Scalar::zero();
            end[1][1] = f00.clone() / b.clone();
            if end[1][1] < Scalar::zero() || end[1][1] > Scalar::one() {
                end[1][1] = Scalar::from_f32(0.5).unwrap();
            }
        }
    }
    (edge, end)
}

// Compute the location of the minimum of R on the segment of
// intersection for the line dR/ds = 0 and the domain [0,1]^2.
#[allow(non_snake_case)]
fn ComputeMinimumParameters<Scalar: RealField + PartialOrd>(
    edge: [i32; 2],
    end: [[Scalar; 2]; 2],
    b: Scalar,
    c: Scalar,
    e: Scalar,
    g00: Scalar,
    g10: Scalar,
    g01: Scalar,
    g11: Scalar,
) -> [Scalar; 2] {
    let delta = end[1][1].clone() - end[0][1].clone();
    let h0 = delta.clone()
        * (-b.clone() * end[0][0].clone() + c.clone() * end[0][1].clone() - e.clone());
    if h0 >= Scalar::zero() {
        if edge[0] == 0 {
            [Scalar::zero(), GetClampedRoot(c, g00, g01)]
        } else if edge[0] == 1 {
            [Scalar::one(), GetClampedRoot(c, g10, g11)]
        } else {
            [end[0][0].clone(), end[0][1].clone()]
        }
    } else {
        let h1 =
            delta * (-b.clone() * end[1][0].clone() + c.clone() * end[1][1].clone() - e.clone());
        if h1 <= Scalar::zero() {
            if edge[1] == 0 {
                [Scalar::zero(), GetClampedRoot(c, g00, g01)]
            } else if edge[1] == 1 {
                [Scalar::one(), GetClampedRoot(c, g10, g11)]
            } else {
                [end[1][0].clone(), end[1][1].clone()]
            }
        } else
        // h0 < 0 and h1 > 0
        {
            let z = Scalar::min(
                Scalar::max(h0.clone() / (h0 - h1), Scalar::zero()),
                Scalar::one(),
            );
            let omz = Scalar::one() - z.clone();
            [
                omz.clone() * end[0][0].clone() + z.clone() * end[1][0].clone(),
                omz.clone() * end[0][1].clone() + z.clone() * end[1][1].clone(),
            ]
        }
    }
}

#[allow(non_snake_case)]
fn GetClampedRoot<Scalar: RealField + PartialOrd>(slope: Scalar, h0: Scalar, h1: Scalar) -> Scalar {
    if h0 < Scalar::zero() {
        if h1 > Scalar::zero() {
            let r = -h0 / slope;
            if r > Scalar::one() {
                Scalar::from_f32(0.5).unwrap()
            } else {
                r
            }
            // The slope is positive and -h0 is positive, so there is
            // no need to test for a negative value and clamp it.
        } else {
            Scalar::one()
        }
    } else {
        Scalar::zero()
    }
}
