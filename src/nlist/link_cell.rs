use glam::{UVec3, Vec3};
use std::collections::HashMap;

use crate::boxdim::BoxDim;

const LINK_CELL_TERMINATOR: u32 = u32::MAX;

pub struct LinkCell {
    cell_width: f32,
    cell_dim: UVec3,
    size: usize,
    starts: Vec<u32>, // TODO this needs a better name
    cell_list: Vec<u32>,
    neighbors: HashMap<u32, Vec<u32>>,
    boxdim: BoxDim,
    points: Vec<Vec3>,
}

fn compute_cell_dim(boxdim: &BoxDim, l: Vec3, cell_width: f32) -> UVec3 {
    let mut dim = UVec3::ZERO;

    dim.x = (l.x / cell_width) as u32;
    dim.y = (l.y / cell_width) as u32;
    if boxdim.is_2d() {
        dim.z = 1;
    } else {
        dim.z = (l.z / cell_width) as u32;
    }

    dim
}

fn get_cell_idx(point: Vec3, boxdim: &BoxDim, cell_dim: UVec3) -> usize {
    let alpha = boxdim.fractional(&point);
    let mut c = UVec3::ZERO;

    c.x = (alpha.x * cell_dim.x as f32).floor() as u32;
    c.x %= cell_dim.x;
    c.y = (alpha.y * cell_dim.y as f32).floor() as u32;
    c.y %= cell_dim.y;
    c.z = (alpha.z * cell_dim.z as f32).floor() as u32;
    c.z %= cell_dim.z;

    coord_to_index(c, cell_dim)
}

fn coord_to_index(c: UVec3, cell_dim: UVec3) -> usize {
    (c.z + c.y * cell_dim.z + c.x * cell_dim.z * cell_dim.y) as usize
}

fn index_to_coord(idx: usize, cell_dim: UVec3) -> UVec3 {
    let idx = idx as u32;
    let z = idx % cell_dim.z;
    let y = (idx / cell_dim.z) % cell_dim.y;
    let x = idx / (cell_dim.z * cell_dim.y);
    UVec3::new(x, y, z)
}

fn get_neigh_loop_bounds(cell_idx: UVec3, cell_dim: UVec3) -> (UVec3, UVec3) {
    let x_bounds = _get_neigh_loop_bounds_dim(cell_idx.x, cell_dim.x);
    let y_bounds = _get_neigh_loop_bounds_dim(cell_idx.y, cell_dim.y);
    let z_bounds = _get_neigh_loop_bounds_dim(cell_idx.z, cell_dim.z);

    (
        UVec3::new(x_bounds.0, y_bounds.0, z_bounds.0),
        UVec3::new(x_bounds.1, y_bounds.1, z_bounds.1),
    )
}

fn _get_neigh_loop_bounds_dim(idx: u32, dim: u32) -> (u32, u32) {
    let lower;
    let upper;

    if dim < 3 {
        lower = idx;
    } else {
        lower = (idx as i32 - 1).rem_euclid(dim as i32) as u32;
    }
    if dim < 2 {
        upper = idx;
    } else {
        upper = (idx + 1).rem_euclid(dim);
    }

    (lower, upper)
}

impl LinkCell {
    pub fn new(boxdim: BoxDim, points: Vec<Vec3>, cell_width: f32) -> Self {
        assert!(!points.is_empty());
        assert!(cell_width > 0.0);
        assert!(boxdim.periodic().all());

        if boxdim.is_2d() {
            for p in points.iter() {
                assert!(p.z.abs() < 1e-6) // arbitrary?
            }
        }

        let l = boxdim.nearest_plane_difference();
        let cell_dim = compute_cell_dim(&boxdim, l, cell_width);
        if cell_width * 2.0 > l.x || cell_width * 2.0 > l.y || cell_width * 2.0 > l.z {
            panic!();
        }
        let size = (cell_dim.x * cell_dim.y * cell_dim.z) as usize;
        assert!(size > 0);

        let mut starts = vec![0; points.len()];
        let mut cell_list = vec![LINK_CELL_TERMINATOR; size];

        for i in (0..starts.len()).rev() {
            let cell_idx = get_cell_idx(points[i], &boxdim, cell_dim);
            starts[i] = cell_list[cell_idx];
            cell_list[cell_idx] = starts[i]
        }

        let neighbors = HashMap::new();

        Self {
            cell_width,
            cell_dim,
            size,
            starts,
            cell_list,
            neighbors,
            boxdim,
            points,
        }
    }

    fn get_cell_neighbors(&mut self, cell_idx: usize) -> &[u32] {
        let key = cell_idx as u32;

        if self.neighbors.contains_key(&key) {
            self.neighbors.get(&key).unwrap()
        } else {
            self.compute_cell_neighbors(cell_idx)
        }
    }

    fn compute_cell_neighbors(&mut self, cell_idx: usize) -> &[u32] {
        let mut neighbors = Vec::new();
        let coord = index_to_coord(cell_idx, self.cell_dim);

        let bounds = get_neigh_loop_bounds(coord, self.cell_dim);

        for z in bounds.0.z..=bounds.1.z {
            for y in bounds.0.y..=bounds.1.y {
                for x in bounds.0.x..=bounds.1.x {
                    let c = UVec3::new(x, y, z);
                    let idx = coord_to_index(c, self.cell_dim);
                    if idx != cell_idx {
                        neighbors.push(idx as u32);
                    }
                }
            }
        }

        self.neighbors.insert(cell_idx as u32, neighbors);
        self.neighbors.get(&(cell_idx as u32)).unwrap()
    }
}
