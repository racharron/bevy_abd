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

impl<Scalar: RealField> Moments<Scalar> {
    pub fn compact_inv_m(self) -> OMatrix<Scalar, Const<4>, Const<4>> {
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
        Matrix4::new(
            m11, x30.clone(), x32.clone(), x35.clone(),
            x30.clone(), m22, x38.clone(), x39.clone(),
            x32.clone(), x38.clone(), m33, x41.clone(),
            x35.clone(), x39.clone(), x41.clone(), m44
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
        self.compact_inv_m().kronecker(&Matrix3::identity())
    }
}
