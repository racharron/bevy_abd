//! Assorted geometric primitives.


#[cfg(test)]
mod tests;
mod tri_point;
mod seg_seg;
mod line_line;

pub use tri_point::tri_pt_squared_distance;
pub use seg_seg::seg_seg_squared_distance;

