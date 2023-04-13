use ndarray::prelude::*;

pub struct NeighborList {
    pub query_point_indices: Array1<u32>,
    pub point_indices: Array1<u32>,
    pub neighbor_counts: Array1<u32>,
    pub segments: Array1<u32>,
    pub distances: Array1<f32>,
}

pub fn particle_to_grid_cube(
    points: ArrayView2<f32>,
    values: ArrayView1<f32>,
    l: f32,  // assume cubic box
    bins: usize  // bins per dimensions
) -> Array3<f32> {
    assert_eq!(points.shape()[0], values.shape()[0]);

    let mut grid = Array3::<f32>::zeros((bins, bins, bins));
    let mut counts = Array3::<u32>::zeros((bins, bins, bins));
    let min = -l / 2.0;
    // let max = l / 2.0;
    let bin_size = l / bins as f32;
    for i in 0..points.shape()[0] {
        let x = (points[[i, 0]] - min) / bin_size;
        let y = (points[[i, 1]] - min) / bin_size;
        let z = (points[[i, 2]] - min) / bin_size;
        let x = x as usize;
        let y = y as usize;
        let z = z as usize;
        grid[[x, y, z]] += values[i];
        counts[[x, y, z]] += 1;
    }
    grid / counts.map(|x| if *x == 0 { 1.0 } else { *x as f32 })
}