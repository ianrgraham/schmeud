use pyo3::prelude::*;
// use pyo3::types::PyType;
use ndarray::prelude::*;

use numpy::*;

enum SoftnessCalculator {
    ParrinelloRadial{
        rad: f32
    }
}

#[pyclass]
struct MSD {}

#[pyclass]
struct SISF {
    k: Option<f32>,
    ref_pos: Option<Array2<f32>>
}

#[pymethods]
impl SISF {

    #[new]
    fn new() -> Self {
        Self{k: None, ref_pos: None}
    }

    fn compute_iter<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        let arr = Array1::<f32>::zeros(10);
        arr.into_pyarray(py)
    }
}
