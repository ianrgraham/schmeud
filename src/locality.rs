use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::vec;

use itertools::iproduct;
use ndarray::prelude::*;

use numpy::*;
use pyo3::prelude::*;

#[pyclass]
pub struct NeighborList {
    pub query_point_indices: Array1<u32>,
    pub point_indices: Array1<u32>,
    pub neighbor_counts: Array1<u32>,
    pub segments: Array1<u32>,
    pub distances: Array1<f32>,
}

#[pymethods]
impl NeighborList {
    #[new]
    pub fn new(
        query_point_indices: PyReadonlyArray1<u32>,
        point_indices: PyReadonlyArray1<u32>,
        neighbor_counts: PyReadonlyArray1<u32>,
        segments: PyReadonlyArray1<u32>,
        distances: PyReadonlyArray1<f32>,
    ) -> Self {
        let query_point_indices = query_point_indices.as_array().to_owned();
        let point_indices = point_indices.as_array().to_owned();
        let neighbor_counts = neighbor_counts.as_array().to_owned();
        let segments = segments.as_array().to_owned();
        let distances = distances.as_array().to_owned();

        NeighborList {
            query_point_indices,
            point_indices,
            neighbor_counts,
            segments,
            distances,
        }
    }
}

pub fn particle_to_grid_cube(
    points: ArrayView2<f32>,
    values: ArrayView1<f32>,
    l: f32,      // assume cubic box
    bins: usize, // bins per dimensions
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

pub fn particle_to_grid_cube_cic(
    points: ArrayView2<f32>,
    values: ArrayView1<f32>,
    l: f32,      // assume cubic box
    bins: usize, // bins per dimensions
) -> Array3<f32> {
    assert_eq!(points.shape()[0], values.shape()[0]);

    let mut grid = Array3::<f32>::zeros((bins, bins, bins));
    let mut weights = Array3::<f32>::zeros((bins, bins, bins));
    let min = -l / 2.0;
    let bin_size = l / bins as f32;
    for i in 0..points.shape()[0] {
        let p = points.row(i);
        let i_xo = (p[0] - min) / bin_size;
        let i_yo = (p[1] - min) / bin_size;
        let i_zo = (p[2] - min) / bin_size;
        for (i_xd, i_yd, i_zd) in iproduct!(-1isize..=1, -1isize..=1, -1isize..=1) {
            let i_x = (i_xo as usize + i_xd as usize) % bins;
            let i_y = (i_yo as usize + i_yd as usize) % bins;
            let i_z = (i_zo as usize + i_zd as usize) % bins;
            let x = i_x as f32 * bin_size + min;
            let y = i_y as f32 * bin_size + min;
            let z = i_z as f32 * bin_size + min;

            let dx = (x - p[0]).abs() / bin_size;
            let dy = (y - p[1]).abs() / bin_size;
            let dz = (z - p[2]).abs() / bin_size;

            if dx > 1.0 || dy > 1.0 || dz > 1.0 {
                continue;
            }

            let weight = (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
            grid[[i_x, i_y, i_z]] += values[i] * weight;
            weights[[i_x, i_y, i_z]] += weight;
        }
    }
    grid / weights.map(|x| if *x == 0.0 { 1.0 } else { *x as f32 })
}

pub fn particle_to_grid_cube_with_counts(
    points: ArrayView2<f32>,
    values: ArrayView1<f32>,
    l: f32,      // assume cubic box
    bins: usize, // bins per dimensions
) -> (Array3<f32>, Array3<u32>) {
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
    (
        grid / counts.map(|x| if *x == 0 { 1.0 } else { *x as f32 }),
        counts,
    )
}

pub fn particle_to_grid_cube_cic_with_weights(
    points: ArrayView2<f32>,
    values: ArrayView1<f32>,
    l: f32,      // assume cubic box
    bins: usize, // bins per dimensions
) -> (Array3<f32>, Array3<f32>) {
    assert_eq!(points.shape()[0], values.shape()[0]);

    let mut grid = Array3::<f32>::zeros((bins, bins, bins));
    let mut weights = Array3::<f32>::zeros((bins, bins, bins));
    let min = -l / 2.0;
    let bin_size = l / bins as f32;
    for i in 0..points.shape()[0] {
        let p = points.row(i);
        let i_xo = (p[0] - min) / bin_size;
        let i_yo = (p[1] - min) / bin_size;
        let i_zo = (p[2] - min) / bin_size;
        for (i_xd, i_yd, i_zd) in iproduct!(-1isize..=1, -1isize..=1, -1isize..=1) {
            let i_x = (i_xo as usize + i_xd as usize) % bins;
            let i_y = (i_yo as usize + i_yd as usize) % bins;
            let i_z = (i_zo as usize + i_zd as usize) % bins;
            let x = i_x as f32 * bin_size + min;
            let y = i_y as f32 * bin_size + min;
            let z = i_z as f32 * bin_size + min;

            let dx = (x - p[0]).abs() / bin_size;
            let dy = (y - p[1]).abs() / bin_size;
            let dz = (z - p[2]).abs() / bin_size;

            if dx > 1.0 || dy > 1.0 || dz > 1.0 {
                continue;
            }

            let weight = (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
            grid[[i_x, i_y, i_z]] += values[i] * weight;
            weights[[i_x, i_y, i_z]] += weight;
        }
    }
    (
        grid / weights.map(|x| if *x == 0.0 { 1.0 } else { *x as f32 }),
        weights,
    )
}

#[pyclass]
#[derive(Debug, Default)]
pub struct BlockTree {
    site2block: HashMap<(u16, u16, u16), (usize, (i8, i8, i8))>,
    id2block: HashMap<usize, BlockNode>,
    roots: HashSet<usize>,
    leafs: HashSet<usize>,
    periodic: bool,
    shape: Box<[usize]>,
}

#[derive(Debug, PartialEq)]
enum Percolation {
    None,
    PercAt(usize),
    Perced,
}

#[pyclass]
#[derive(Debug)]
struct BlockNode {
    id: usize,
    parent_id: Option<(usize, (i8, i8, i8))>,
    children_ids: Vec<(usize, (i8, i8, i8))>,
    sites: Vec<(u16, u16, u16)>,
    vals: Vec<f32>,
    percolated: Percolation,
}

// TODO refactor images to use glam or nalgebra
fn add_images(image1: (i8, i8, i8), image2: (i8, i8, i8)) -> (i8, i8, i8) {
    let mut image = image1;
    image.0 += image2.0;
    image.1 += image2.1;
    image.2 += image2.2;
    image
}

fn neg(image: (i8, i8, i8)) -> (i8, i8, i8) {
    (-image.0, -image.1, -image.2)
}

fn shift(site: (u16, u16, u16), image: (i8, i8, i8), shape: &[usize]) -> (i16, i16, i16) {
    let mut site = (site.0 as i16, site.1 as i16, site.2 as i16);
    site.0 += image.0 as i16 * shape[0] as i16;
    site.1 += image.1 as i16 * shape[1] as i16;
    if shape.len() > 2 {
        site.2 += image.2 as i16 * shape[2] as i16;
    }
    site
}

#[pymethods]
impl BlockTree {
    #[new]
    pub fn new(grid: PyReadonlyArrayDyn<f32>, periodic: bool) -> Self {
        let shape = grid.shape();
        assert!(grid.is_c_contiguous());
        assert!(shape.len() == 3 || shape.len() == 2);
        let is_2d = shape.len() == 2;

        shape.iter().for_each(|&x| assert!(x > 2));

        let mut tree = BlockTree::default();
        tree.periodic = periodic;
        tree.shape = shape.into();

        let arr = grid.as_array();
        let flat_arr = arr.view().into_shape((arr.len(),)).unwrap();

        let mut indices: Vec<usize> = (0..arr.len()).collect();
        indices.sort_unstable_by(move |&i, &j| flat_arr[i].total_cmp(&flat_arr[j]));

        // unravel indices
        let mut unraveled_indices = vec![(0, 0, 0); arr.len()];
        for (i, &idx) in indices.iter().enumerate() {
            let (x, y, z) = if is_2d {
                (idx % shape[0], idx / shape[0], 0)
            } else {
                (
                    idx % shape[0],
                    (idx / shape[0]) % shape[1],
                    idx / (shape[0] * shape[1]),
                )
            };
            unraveled_indices[i] = (x as u16, y as u16, z as u16);
        }

        let neigh = if is_2d {
            vec![[-1i16, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
        } else {
            vec![
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ]
        };

        let mut found_blocks = Vec::new();
        let mut root_blocks = HashMap::new();

        // build tree
        let mut next_block_id = 0;
        for ((x, y, z), i) in unraveled_indices.into_iter().zip(indices.into_iter()) {
            let v = flat_arr[i];
            found_blocks.clear();
            root_blocks.clear();

            let mut percolation = false;

            for n in &neigh {
                let mut image = (0i8, 0, 0);
                let x = x as i16 + n[0];
                let x = if x < 0 {
                    if !periodic {
                        continue;
                    }
                    image.0 += 1;
                    x.rem_euclid(shape[0] as i16) as u16
                } else if x >= shape[0] as i16 {
                    if !periodic {
                        continue;
                    }
                    image.0 -= 1;
                    x.rem_euclid(shape[0] as i16) as u16
                } else {
                    x as u16
                };

                let y = y as i16 + n[1];
                let y = if y < 0 {
                    if !periodic {
                        continue;
                    }
                    image.1 += 1;
                    y.rem_euclid(shape[1] as i16) as u16
                } else if y >= shape[1] as i16 {
                    if !periodic {
                        continue;
                    }
                    image.1 -= 1;
                    y.rem_euclid(shape[1] as i16) as u16
                } else {
                    y as u16
                };

                let z = if is_2d {
                    0
                } else {
                    let z = z as i16 + n[2];
                    if z < 0 {
                        if !periodic {
                            continue;
                        }
                        image.2 += 1;
                        // dbg!(image);
                        z.rem_euclid(shape[2] as i16) as u16
                    } else if z >= shape[2] as i16 {
                        if !periodic {
                            continue;
                        }
                        image.2 -= 1;
                        // dbg!(image);
                        z.rem_euclid(shape[2] as i16) as u16
                    } else {
                        z as u16
                    }
                };

                if let Some(block) = tree.site2block.get(&(x, y, z)) {
                    found_blocks.push((block.0, add_images(image, block.1)));
                }
            }

            for f in &found_blocks {
                let mut node = tree.id2block.get(&f.0).unwrap();
                let mut image = f.1;
                while let Some(p) = node.parent_id {
                    node = tree.id2block.get(&p.0).unwrap();
                    image = add_images(image, p.1);
                }
                if let Some(root_image) = root_blocks.get(&node.id) {
                    if *root_image != image || node.percolated != Percolation::None {
                        percolation = true;
                    }
                } else {
                    root_blocks.insert(node.id, image);
                }
            }

            if root_blocks.is_empty() {
                // create new node
                let node = BlockNode {
                    id: next_block_id,
                    parent_id: None,
                    children_ids: Vec::new(),
                    sites: vec![(x, y, z)],
                    vals: vec![v],
                    percolated: Percolation::None,
                };

                tree.site2block
                    .insert((x, y, z), (next_block_id, (0, 0, 0)));
                tree.id2block.insert(next_block_id, node);
                tree.roots.insert(next_block_id);
                tree.leafs.insert(next_block_id);
                next_block_id += 1;
            } else if root_blocks.len() == 1 {
                // add to existing node
                let (root_id, image) = root_blocks.iter().next().unwrap();
                tree.site2block.insert((x, y, z), (*root_id, *image));
                let node = tree.id2block.get_mut(root_id).unwrap();
                node.sites.push((x, y, z));
                node.vals.push(v);
                if percolation && node.percolated == Percolation::None {
                    node.percolated = Percolation::PercAt(node.vals.len() - 1);
                }
            } else {
                // merge nodes into parent
                let perc = if percolation {
                    Percolation::Perced
                } else {
                    Percolation::None
                };
                let mut parent_node = BlockNode {
                    id: next_block_id,
                    parent_id: None,
                    children_ids: Vec::new(),
                    sites: vec![(x, y, z)],
                    vals: vec![v],
                    percolated: perc,
                };
                tree.site2block
                    .insert((x, y, z), (next_block_id, (0, 0, 0)));
                tree.roots.insert(next_block_id);

                for old_root in root_blocks.iter() {
                    let (root_id, image) = old_root;
                    let child_node = tree.id2block.get_mut(root_id).unwrap();
                    child_node.parent_id = Some((next_block_id, *image));
                    // child_node.percolated = percolation;
                    parent_node.children_ids.push((*root_id, neg(*image)));
                    tree.roots.remove(root_id);
                }

                tree.id2block.insert(next_block_id, parent_node);
                next_block_id += 1;
            }
        }
        tree
    }

    pub fn mass_and_msd(&self, filt: PyReadonlyArray1<f32>) -> Vec<Vec<(f32, f32, f32, f32)>> {
        let filt = filt.as_array();

        let mut output = vec![vec![]; filt.len()];
        let mut idx = (filt.len() - 1) as isize;
        let mut nodes = self.roots.clone();

        while idx >= 0 {
            let filt_val = &filt[idx as usize];
            let mut new_nodes = HashSet::new();

            // walk down until we find a node that is at the filter value
            for node_id in nodes.iter() {
                let mut children = VecDeque::from([*node_id]);
                while let Some(next_node) = children.pop_front() {
                    let block = self.id2block.get(&next_node).unwrap();
                    if block.vals.first().unwrap() > filt_val {
                        // add all children
                        for child in &block.children_ids {
                            children.push_back(child.0);
                        }
                    } else {
                        new_nodes.insert(next_node);
                    }
                }
            }

            // now we have all the nodes that are at the filter value
            for node_id in new_nodes.iter() {
                let block = self.id2block.get(&node_id).unwrap();
                if block.percolated == Percolation::Perced {
                    continue;
                }
                let mut data = VecDeque::new();
                let mut birth = *block.vals.first().unwrap();
                let mut death = if let Some(parent) = block.parent_id {
                    let parent = self.id2block.get(&parent.0).unwrap();
                    *parent.vals.first().unwrap()
                } else {
                    f32::INFINITY
                };
                let mut cur_idx = 0;
                for (site, val) in block.sites.iter().zip(block.vals.iter()) {
                    if val > filt_val {
                        death = *val;
                        break;
                    }
                    let site = shift(*site, self.site2block[site].1, &self.shape);
                    data.push_back((site.0 as i16, site.1 as i16, site.2 as i16));
                    birth = *val;
                    cur_idx += 1;
                }
                if let Percolation::PercAt(perc_idx) = block.percolated {
                    if perc_idx < cur_idx {
                        continue;
                    }
                }
                let mut children = VecDeque::from(block.children_ids.clone());
                while let Some(next_node) = children.pop_front() {
                    let block = self.id2block.get(&next_node.0).unwrap();
                    let site2block = &self.site2block;
                    let sites = block
                        .sites
                        .iter()
                        .map(|x| shift(*x, add_images(next_node.1, site2block[x].1), &self.shape))
                        .collect::<Vec<_>>();
                    data.extend(sites);
                    for child in &block.children_ids {
                        children.push_back((child.0, add_images(next_node.1, child.1)));
                    }
                }

                // compute mass and msd and append to output
                let mass = data.len() as f32;
                let mean = data.iter().fold((0.0, 0.0, 0.0), |acc, x| {
                    (acc.0 + x.0 as f32, acc.1 + x.1 as f32, acc.2 + x.2 as f32)
                });
                let mean = (mean.0 / mass, mean.1 / mass, mean.2 / mass);
                let msd = data.iter().fold(0.0, |acc, x| {
                    let dx = x.0 as f32 - mean.0;
                    let dy = x.1 as f32 - mean.1;
                    let dz = x.2 as f32 - mean.2;
                    acc + dx * dx + dy * dy + dz * dz
                }) / mass;

                output[idx as usize].push((birth, death, mass, msd));
            }

            nodes = new_nodes;
            idx -= 1;
        }
        output
    }

    pub fn get_sites(&self, filt: PyReadonlyArray1<f32>) -> Vec<Vec<Vec<(f32, f32, f32)>>> {
        let filt = filt.as_array();

        let mut output = vec![vec![]; filt.len()];
        let mut idx = (filt.len() - 1) as isize;

        // let mut nodes = self.roots.iter().map(|x| (*x (0, VecDeque::<(u16, u16, u16)>::new()))).collect::<HashMap<_, _>>();
        let mut nodes = self.roots.clone();

        while idx >= 0 {
            let filt_val = &filt[idx as usize];
            let mut new_nodes = HashSet::new();

            // walk down until we find a node that is at the filter value
            for node_id in nodes.iter() {
                let mut children = VecDeque::from([*node_id]);
                while let Some(next_node) = children.pop_front() {
                    let block = self.id2block.get(&next_node).unwrap();
                    if block.vals.first().unwrap() > filt_val {
                        // add all children
                        for child in &block.children_ids {
                            children.push_back(child.0);
                        }
                    } else {
                        new_nodes.insert(next_node);
                    }
                }
            }

            // now we have all the nodes that are at the filter value
            for node_id in new_nodes.iter() {
                let block = self.id2block.get(&node_id).unwrap();
                if block.percolated == Percolation::Perced {
                    continue;
                }
                let mut data = Vec::new();

                let mut cur_idx = 0;
                for (site, val) in block.sites.iter().zip(block.vals.iter()) {
                    if val > filt_val {
                        break;
                    }
                    let site = shift(*site, self.site2block[site].1, &self.shape);
                    data.push((site.0 as f32, site.1 as f32, site.2 as f32));
                    cur_idx += 1;
                }
                if let Percolation::PercAt(perc_idx) = block.percolated {
                    if perc_idx < cur_idx {
                        continue;
                    }
                }
                let mut children = VecDeque::from(block.children_ids.clone());
                while let Some(next_node) = children.pop_front() {
                    let block = self.id2block.get(&next_node.0).unwrap();
                    let site2block = &self.site2block;
                    let sites = block
                        .sites
                        .iter()
                        .map(|x| shift(*x, add_images(next_node.1, site2block[x].1), &self.shape))
                        .map(|x| (x.0 as f32, x.1 as f32, x.2 as f32))
                        .collect::<Vec<_>>();
                    data.extend(sites);
                    for child in &block.children_ids {
                        children.push_back((child.0, add_images(next_node.1, child.1)));
                    }
                }

                output[idx as usize].push(data);
            }

            nodes = new_nodes;
            idx -= 1;
        }
        output
    }
}

#[pyclass]
#[derive(Debug, Default)]
pub struct NeighBlockTree {
    site2block: HashMap<(u16, u16, u16), (usize, (i8, i8, i8))>,
    id2block: HashMap<usize, BlockNode>,
    roots: HashSet<usize>,
    leafs: HashSet<usize>,
    periodic: bool,
    boxdim: crate::boxdim::BoxDim,
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn build_tree() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let orig_array = array![
                [4.0, 3.0, -13.0, -6.0, -5.0],
                [5.0, 2.0, -14.0, -4.0, -7.0],
                [6.0, 1.0, -15.0, -8.0, -3.0],
                [7.0, 0.0, 9.0, -9.0, -2.0],
                [8.0, -1.0, -12.0, -10.0, -11.0],
            ]
            .into_dyn();
            let np_array = PyArrayDyn::from_owned_array(py, orig_array).readonly();

            let tree = BlockTree::new(np_array, true);

            assert!(tree.id2block[&2].percolated == Percolation::PercAt(2));
            assert!(tree.id2block[&4].percolated == Percolation::Perced);
            assert!(tree.id2block.len() == 5);

            let filt = array![-16.0f32, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, 10.0];
            let np_filt = PyArray1::from_owned_array(py, filt).readonly();
            let output = tree.mass_and_msd(np_filt);

            assert!(output[0].len() == 0);
            assert!(output.last().unwrap().len() == 0);
            assert!(output[1].len() == 1);
        });
    }
}
