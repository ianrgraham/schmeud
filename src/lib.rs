#![allow(dead_code)]

use pyo3::prelude::*;

mod dynamics;
mod python;
mod statics;
mod softness;


#[pymodule]
fn schmeud(py: Python, m: &PyModule) -> PyResult<()> {

    // register submodules
    python::register_dynamics(py, m)?;
    python::register_statics(py, m)?;
    python::register_ml(py, m)?;
    
    Ok(())
}

mod utils {
    #[inline(always)]
    pub fn digitize_lin(x: f32, arr: &[f32], l: f32) -> usize {

        let ub = arr.len() + 1;
        let lb = 0;

        let mut j = ((x-arr[0])/l) as usize + 1;
        if j < lb { j = lb }
        else if j >= ub { j = ub }
        else if arr[j+1]-x < x-arr[j] { j += 1 }

        j
    }
}