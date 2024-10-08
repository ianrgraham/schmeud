// pub mod aabb;
pub mod link_cell;
#[cfg(any(feature = "voro-static", feature = "voro-system"))]
pub mod voro;

use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass]
#[derive(Clone)]
pub struct NeighborList {
    #[pyo3(get)]
    pub query_point_indices: Vec<u32>,
    #[pyo3(get)]
    pub point_indices: Vec<u32>,
    #[pyo3(get)]
    pub counts: Vec<u32>,
    #[pyo3(get)]
    pub segments: Vec<u32>,
    #[pyo3(get)]
    pub distances: Vec<f32>,
    #[pyo3(get)]
    pub weights: Vec<f32>,
}

#[pymethods]
impl NeighborList {
    #[new]
    pub fn new(
        query_point_indices: Vec<u32>,
        point_indices: Vec<u32>,
        counts: Vec<u32>,
        segments: Vec<u32>,
        distances: Vec<f32>,
        weights: Vec<f32>,
    ) -> Self {
        Self {
            query_point_indices,
            point_indices,
            counts,
            segments,
            distances,
            weights,
        }
    }

    #[classmethod]
    #[pyo3(name = "from_freud")]
    pub fn py_from_freud<'p>(_cls: &'p PyType, freud_nlist: &'p PyAny) -> PyResult<Self> {
        let query_point_indices: Vec<u32> =
            freud_nlist.getattr("query_point_indices")?.extract()?;
        let point_indices: Vec<u32> = freud_nlist.getattr("point_indices")?.extract()?;
        let counts: Vec<u32> = freud_nlist.getattr("counts")?.extract()?;
        let segments: Vec<u32> = freud_nlist.getattr("segments")?.extract()?;
        let distances: Vec<f32> = freud_nlist.getattr("distances")?.extract()?;
        let weights: Vec<f32> = freud_nlist.getattr("weights")?.extract()?;
        Ok(Self {
            query_point_indices,
            point_indices,
            counts,
            segments,
            distances,
            weights,
        })
    }
}

#[derive(Clone)]
pub struct NeighborBond {
    pub query_point_idx: u32,
    pub point_idx: u32,
    pub distance: f32,
    pub weight: f32,
}

impl NeighborBond {
    fn new(query_point_idx: u32, point_idx: u32, distance: f32) -> Self {
        Self {
            query_point_idx,
            point_idx,
            distance,
            weight: 1.0,
        }
    }

    fn new_weighted(query_point_idx: u32, point_idx: u32, distance: f32, weight: f32) -> Self {
        Self {
            query_point_idx,
            point_idx,
            distance,
            weight,
        }
    }

    fn partial_cmp_id_ref_weight(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.query_point_idx != other.query_point_idx {
            self.query_point_idx.partial_cmp(&other.query_point_idx)
        } else if self.point_idx != other.point_idx {
            self.point_idx.partial_cmp(&other.point_idx)
        } else {
            self.weight.partial_cmp(&other.weight)
        }
    }
}

impl PartialEq for NeighborBond {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl PartialOrd for NeighborBond {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
