use ndarray::prelude::*;

pub struct NeighborList {
    pub query_point_indices: Array1<u32>,
    pub point_indices: Array1<u32>,
    pub neighbor_counts: Array1<u32>,
    pub segments: Array1<u32>,
    pub distances: Array1<f32>,
}