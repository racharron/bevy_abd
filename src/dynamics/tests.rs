use petgraph::matrix_graph::Zero;
use crate::dynamics::moments::Moments;

#[test]
fn inv_m_structure() {
    let moments = Moments {
        v: 1.0f32,
        x: 2.0,
        y: 3.0,
        z: 4.0,
        xx: 5.0,
        xy: 6.0,
        xz: 7.0,
        yy: 8.0,
        yz: 9.0,
        zz: 10.0,
    };
    let inv_m = moments.inv_m();
    for (r, row) in inv_m.row_iter().enumerate() {
        for (c, elm) in row.iter().enumerate() {
            assert!(elm.is_finite());
            if r % 3 != c % 3 {
                assert!(elm.is_zero(), "inv_M[{}][{}] = {}", r, c, elm);
            } else {
                assert!(!elm.is_zero(), "inv_M[{}][{}] == 0", r, c)
            }
        }
    }
}
