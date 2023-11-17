use itertools::Itertools;
use nalgebra::{Const, Matrix3, Matrix4, OMatrix, RealField};

/// The mass matrix takes the form of
/// ```py
/// Matrix([
///     [1, 0, 0, x, 0, 0, y, 0, 0, z, 0, 0],
///     [0, 1, 0, 0, x, 0, 0, y, 0, 0, z, 0],
///     [0, 0, 1, 0, 0, x, 0, 0, y, 0, 0, z],
///     [x, 0, 0, x**2, 0, 0, x*y, 0, 0, x*z, 0, 0],
///     [0, x, 0, 0, x**2, 0, 0, x*y, 0, 0, x*z, 0],
///     [0, 0, x, 0, 0, x**2, 0, 0, x*y, 0, 0, x*z],
///     [y, 0, 0, x*y, 0, 0, y**2, 0, 0, y*z, 0, 0],
///     [0, y, 0, 0, x*y, 0, 0, y**2, 0, 0, y*z, 0],
///     [0, 0, y, 0, 0, x*y, 0, 0, y**2, 0, 0, y*z],
///     [z, 0, 0, x*z, 0, 0, y*z, 0, 0, z**2, 0, 0],
///     [0, z, 0, 0, x*z, 0, 0, y*z, 0, 0, z**2, 0],
///     [0, 0, z, 0, 0, x*z, 0, 0, y*z, 0, 0, z**2]
/// ])
/// ```
/// So we only need to know the volume integrals of 1 (the volume)
/// 1, x, y, z, xx, yy, zz, xy, xz, yz
/// This is not scaled, and needs to be scaled by the (uniform) density of the object.
#[derive(Clone, Debug)]
pub struct Moments<Scalar: RealField> {
    pub v: Scalar,
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
    pub xx: Scalar,
    pub xy: Scalar,
    pub xz: Scalar,
    pub yy: Scalar,
    pub yz: Scalar,
    pub zz: Scalar,
}

/// Stores the coefficients of the compacted inverse mass matrix (before the Kronecker product with I_3).
#[derive(Clone, Debug)]
pub struct CompactInvMass<Scalar: RealField> {
    m11: Scalar,
    m12: Scalar,
    m13: Scalar,
    m14: Scalar,
    m22: Scalar,
    m23: Scalar,
    m24: Scalar,
    m33: Scalar,
    m34: Scalar,
    m44: Scalar,
}

impl<Scalar: RealField> Moments<Scalar> {
    pub fn compact_inv_m(self) -> CompactInvMass<Scalar> {
        let Moments {
            v, x, y, z, xx, xy, xz, yy, yz, zz
        } = self;
        //  This was derived with sympy
        let two = Scalar::from_u32(2).unwrap();
        let x0 = yz.clone().powi(2);
        let x1 = x0.clone()*xx.clone();
        let x2 = xy.clone().powi(2);
        let x3 = x2.clone()*zz.clone();
        let x4 = xz.clone().powi(2);
        let x5 = x4.clone()*yy.clone();
        let x6 = yy.clone()*zz.clone();
        let x7 = x6.clone()*xx.clone();
        let x8 = xz.clone()*yz.clone();
        let x9 = two.clone() *x8.clone();
        let x10 = x.clone().powi(2);
        let x11 = x10.clone()*zz.clone();
        let x12 = y.clone().powi(2);
        let x13 = x12.clone()*zz.clone();
        let x14 = z.clone().powi(2);
        let x15 = x14.clone()*yy.clone();
        let x16 = v.clone()*xy.clone();
        let x17 = y.clone()*zz.clone();
        let x18 = x17.clone()*xy.clone();
        let x19 = two.clone()*x.clone();
        let x20 = xy.clone()*z.clone();
        let x21 = x20.clone()*yz.clone();
        let x22 = x8.clone()*y.clone();
        let x23 = yy.clone()*z.clone();
        let x24 = x23.clone()*xz.clone();
        let x25 = xx.clone()*yz.clone();
        let x26 = x25.clone()*z.clone();
        let x27 = two.clone()*y.clone();
        let x28 = x20.clone()*xz.clone();
        let x29 = (
            v.clone()*x1.clone() + v.clone()*x3.clone() + v.clone()*x5.clone() - v.clone()*x7.clone() - x0.clone()*x10.clone() + x11.clone()*yy.clone()
                - x12.clone()*x4.clone() + x13.clone()*xx.clone() - x14.clone()*x2.clone() + x15.clone()*xx.clone() - x16.clone()*x9.clone()
                - x18.clone()*x19.clone() + x19.clone()*x21.clone() + x19.clone()*x22.clone() - x19.clone()*x24.clone() - x26.clone()*x27.clone() + x27.clone()*x28.clone()
        ).recip();
        let x30 = x29.clone()*(-x.clone()*x0.clone() + x.clone()*x6.clone() - x18.clone() + x21.clone() + x22.clone() - x24.clone());
        let x31 = x.clone()*xy.clone();
        let x32 = x29.clone()*(x.clone()*x8.clone() + x17.clone()*xx.clone() - x26.clone() + x28.clone() - x31.clone()*zz.clone() - x4.clone()*y.clone());
        let x33 = xz.clone()*y.clone();
        let x34 = xz.clone()*yy.clone();
        let x35 = x29.clone()*(
            -x.clone()*x34.clone() - x2.clone()*z.clone() + x23.clone()*xx.clone() - x25.clone()*y.clone() + x31.clone()*yz.clone() + x33.clone()*xy.clone()
        );
        let x36 = y.clone()*z.clone();
        let x37 = x.clone()*yz.clone();
        let x38 = x29.clone()*(
            -v.clone()*x8.clone() - x.clone()*x17.clone() - x14.clone()*xy.clone() + x16.clone()*zz.clone() + x33.clone()*z.clone() + x37.clone()*z.clone()
        );
        let x39 = x29.clone()*(
            v.clone()*x34.clone() - x.clone()*x23.clone() - x12.clone()*xz.clone() - x16.clone()*yz.clone() + x20.clone()*y.clone() + x37.clone()*y.clone()
        );
        let x40 = v.clone()*xx.clone();
        let x41 = x29.clone()*(
            v.clone()*x25.clone() + x.clone()*x20.clone() + x.clone()*x33.clone() - x10.clone()*yz.clone() - x16.clone()*xz.clone() - x36.clone()*xx.clone()
        );
        let m11 = x29.clone() * (x1.clone() + x3.clone() + x5.clone() - x7.clone() - x9.clone() * xy.clone());
        let m22 = x29.clone() * (v.clone() * x0.clone() - v.clone() * x6.clone() + x13.clone() + x15.clone() - two.clone() * x36.clone() * yz.clone());
        let m33 = x29.clone() * (v.clone() * x4.clone() + x11.clone() + x14.clone() * xx.clone() - x19.clone() * xz.clone() * z.clone() - x40.clone() * zz.clone());
        let m44 = x29.clone() * (v.clone() * x2.clone() + x10.clone() * yy.clone() + x12.clone() * xx.clone() - x27.clone() * x31.clone() - x40.clone() * yy.clone());
        CompactInvMass {
            m11,
            m12: x30.clone(),
            m13: x32.clone(),
            m14: x35.clone(),
            m22,
            m23: x38.clone(),
            m24: x39.clone(),
            m33,
            m34: x41.clone(),
            m44
        }
    }
    pub fn inv_m(self) -> OMatrix<Scalar, Const<12>, Const<12>> {
        self.compact_inv_m().inv_m()
    }
}

impl<Scalar: RealField> CompactInvMass<Scalar> {

    fn to_matrix(self) -> OMatrix<Scalar, Const<4>, Const<4>> {
        let CompactInvMass {
            m11, m12, m13, m14, m22, m23, m24, m33, m34, m44
        } = self;
        Matrix4::new(
            m11, m12.clone(), m13.clone(), m14.clone(),
            m12, m22, m23.clone(), m24.clone(),
            m13, m23, m33, m34.clone(),
            m14, m24, m34, m44
        )
    }

    pub fn inv_m(self) -> OMatrix<Scalar, Const<12>, Const<12>> {
        /*  The following is equivalent to this: (checked with sympy)
        Matrix4::new(
            v, x.clone(), y.clone(), z.clone(),
            x, xx, xy.clone(), xz.clone(),
            y, xy, yy, yz.clone(),
            z, xz, yz, zz
        ).kronecker(&Matrix3::identity()).inv().unwrap()
        */
        self.to_matrix().kronecker(&Matrix3::identity())
    }
}

#[cfg(test)]
#[test]
fn compact_inv_mass_conversions() {
    //  Arbitrarily chosen values
    let moments = Moments {
        v: 1.0f64,
        x: 11.,
        y: 3.,
        z: 23.,
        xx: 5.,
        xy: 6.,
        xz: 7.,
        yy: 8.,
        yz: 9.,
        zz: 10.,
    };
    let inverse = moments.clone().compact_inv_m();
    //  Check to make sure that values are unique for later checking.
    //  If the test fails here, change the values of `moments` until it doesn't.
    let CompactInvMass { m11, m12, m13, m14, m22, m23, m24, m33, m34, m44 } = inverse.clone();
    let compact_elements = [m11, m12, m13, m14, m22, m23, m24, m33, m34, m44];
    let compact_names = ["m11", "m12", "m13", "m14", "m22", "m23", "m24", "m33", "m34", "m44"];
    for (element, name) in compact_elements.clone().into_iter().zip(compact_names.clone()) {
        assert!(element.is_finite(), "{name} = {element}");
    }
    for elements in [m11, m12, m13, m14, m22, m23, m24, m33, m34, m44].clone().into_iter()
        .zip(["m11", "m12", "m13", "m14", "m22", "m23", "m24", "m33", "m34", "m44"].clone())
        .permutations(2)
    {
        let (e1, name1) = elements[0];
        let (e2, name2) = elements[1];
        assert_ne!(e1, e2, "Pre-test sanity checking of selected values {name1}, {name2}");
    }
    let mat4 = inverse.clone().to_matrix();
    assert_eq!(mat4, mat4.transpose());
    assert_eq!(mat4[(0, 0)], inverse.m11);
    assert_eq!(mat4[(0, 1)], inverse.m12);
    assert_eq!(mat4[(0, 2)], inverse.m13);
    assert_eq!(mat4[(0, 3)], inverse.m14);
    assert_eq!(mat4[(1, 1)], inverse.m22);
    assert_eq!(mat4[(1, 2)], inverse.m23);
    assert_eq!(mat4[(1, 3)], inverse.m24);
    assert_eq!(mat4[(2, 2)], inverse.m33);
    assert_eq!(mat4[(2, 3)], inverse.m34);
    assert_eq!(mat4[(3, 3)], inverse.m44);
    let mat12 = inverse.inv_m();
    assert_eq!(mat12, moments.inv_m());
    for (r, row) in mat12.row_iter().enumerate() {
        for (c, &elm) in row.iter().enumerate() {
            assert!(elm.is_finite(), "inv_M[({r}, {c})] = {elm}");
            if r % 3 == c % 3 {
                assert_eq!(elm, mat4[(r / 3, c / 3)], "inv_M[({r}, {c}]");
            } else {
                assert_eq!(elm, 0.0, "inv_M[{}][{}] = {}", r, c, elm);
            }
        }
    }
}
