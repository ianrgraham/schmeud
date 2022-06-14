// #![deny(missing_docs,
//     missing_debug_implementations, missing_copy_implementations,
//     trivial_casts, trivial_numeric_casts,
//     unsafe_code,
//     unstable_features,
//     unused_import_braces, unused_qualifications)]

#![allow(dead_code)]

//! This crate suplies a number of high-performance functions to be called through
//! FFI from Python.


use pyo3::prelude::*;

mod dynamics;
mod bindings;
mod statics;
mod ml;
mod experimental;


#[pymodule]
fn _schmeud(py: Python, m: &PyModule) -> PyResult<()> {

    // register submodules
    bindings::register_dynamics(py, m)?;
    bindings::register_statics(py, m)?;
    bindings::register_ml(py, m)?;
    
    Ok(())
}

mod utils {
    #[inline(always)]
    pub fn digitize_lin(x: f32, arr: &[f32], l: f32) -> usize {

        let ub = arr.len() as isize - 1;
        let lb = 0;

        let mut j = ((x-arr[0])/l) as isize;
        if j < lb { j = lb }
        else if j >= ub { j = ub }
        else if arr[j as usize +1]-x < x-arr[j as usize] { j += 1 }

        j as usize
    }

    /// Tries to find the index which satisfies arr\[idx\] < x <= arr\[idx+1\].
    #[inline(always)]
    pub fn try_digitize_lin(x: f32, arr: &[f32], l: f32) -> Option<usize> {

        let ub = arr.len() as isize - 1;
        let lb = 0;

        let j = ((x-arr[0])/l) as isize;
        if j < lb { None }
        else if j >= ub { None }
        else { Some(j as usize) }
        
    }

    #[inline(always)]
    pub fn gauss_smear(dr: f32, mu: f32, l: f32) -> f32 {
        let term = (dr-mu)/l;
        (-term*term*0.5).exp()
    }
}