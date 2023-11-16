//! Assorted geometric primitives.

mod line_line;
mod seg_seg;
#[cfg(test)]
mod tests;
mod tri_point;

pub use seg_seg::seg_seg_squared_distance;
pub use tri_point::tri_pt_squared_distance;
