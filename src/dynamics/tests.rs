use crate::dynamics::moments::CompactInvMass;
use super::*;

/*
#[test]
fn square_mass() {
    todo!()
}
*/
#[test]
fn compact_inv_mass_conversions() {
    use itertools::Itertools;
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
    let inverse = moments.clone().compact_inv_m(1.0);
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
    assert_eq!(mat12, moments.inv_m(1.0));
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
