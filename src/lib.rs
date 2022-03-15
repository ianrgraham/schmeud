#![deny(missing_docs,
    missing_debug_implementations, missing_copy_implementations,
    trivial_casts, trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces, unused_qualifications)]

//! This crate suplies a number of high-performance functions to be called though
//! FFI from Python.


use pyo3::prelude::*;

mod dynamics;
mod python;
mod statics;
mod softness;

#[pyfunction]
fn test_rust_func_py(_py: Python, x: f32) -> PyResult<f32> {
    Ok(x)
}


#[pymodule]
fn schmeud(py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(
        wrap_pyfunction!(test_rust_func_py, m)?
    )?;

    // register submodules
    python::register_dynamics(py, m)?;
    python::register_statics(py, m)?;
    python::register_ml(py, m)?;
    
    Ok(())
}

mod utils {
    #[inline(always)]
    pub fn digitize_lin(x: f32, arr: &[f32], l: f32) -> usize {

        let ub = arr.len() - 1;
        let lb = 0;

        let mut j = ((x-arr[0])/l) as usize;
        if j < lb { j = lb }
        else if j >= ub { j = ub }
        else if arr[j+1]-x < x-arr[j] { j += 1 }

        j
    }

    #[inline(always)]
    pub fn try_digitize_lin(x: f32, arr: &[f32], l: f32) -> Option<usize> {

        let ub = arr.len() as isize - 1;
        let lb = 0;

        let j = ((x-arr[0])/l) as isize;
        if j < lb { None }
        else if j >= ub { None }
        else { Some(j as usize) }
        
    }
}