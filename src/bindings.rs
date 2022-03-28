
use pyo3::prelude::*;
use pyo3::exceptions::*;
use numpy::*;

pub fn register_dynamics(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "dynamics")?;
    child_module.add_function(
        wrap_pyfunction!(nonaffine_local_strain_py, child_module)?
    )?;
    child_module.add_function(
        wrap_pyfunction!(affine_local_strain_py, child_module)?
    )?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}

pub fn register_statics(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "statics")?;

    child_module.add_function(
        wrap_pyfunction!(spatially_smeared_local_rdfs_py, child_module)?
    )?;

    parent_module.add_submodule(child_module)?;
    Ok(())
}

pub fn register_ml(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "ml")?;

    child_module.add_function(
        wrap_pyfunction!(get_rad_sf_frame_py, child_module)?
    )?;

    child_module.add_function(
        wrap_pyfunction!(get_rad_sf_frame_subset_py, child_module)?
    )?;

    parent_module.add_submodule(child_module)?;
    Ok(())
}

#[pyfunction(name="nonaffine_local_strain")]
fn nonaffine_local_strain_py(
    _py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    crate::dynamics::nonaffine_local_strain(x, y)
        .map_err(|e| PyArithmeticError::new_err(format!("{}", e)))
}

#[pyfunction(name="affine_local_strain")]
fn affine_local_strain_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let x = x.as_array();
    let y = y.as_array();
    match crate::dynamics::affine_local_strain(x, y) {
        Ok(j) => Ok(j.into_pyarray(py)),
        Err(e) => Err(PyArithmeticError::new_err(format!("{}", e)))
    }
}

#[pyfunction(name="self_intermed_scatter_fn")]
fn self_intermed_scatter_fn_py<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray3<f32>,
    q: f32
) -> PyResult<&'py PyArray1<f32>> {
    let traj = traj.as_array().into_owned();
    let sisf = crate::dynamics::self_intermed_scatter_fn(traj, q);
    Ok(sisf.into_pyarray(py))
}

#[pyfunction(name="get_rad_sf_frame")]
fn get_rad_sf_frame_py<'py>(
    py: Python<'py>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    drs: PyReadonlyArray1<f32>,
    type_ids: PyReadonlyArray1<u8>,
    types: u8,
    mus: PyReadonlyArray1<f32>,
    spread: u8
) -> PyResult<&'py PyArray2<f32>> 
{
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let drs = drs.as_array();
    let type_ids = type_ids.as_array();
    let mus = mus.as_slice()?;

    let sfs = crate::ml::get_rad_sf_frame(
        nlist_i, nlist_j, drs, type_ids, types, mus, spread
    );
    Ok(sfs.into_pyarray(py))
}

#[pyfunction(name="get_rad_sf_frame_subset")]
fn get_rad_sf_frame_subset_py<'py>(
    py: Python<'py>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    drs: PyReadonlyArray1<f32>,
    type_ids: PyReadonlyArray1<u8>,
    types: u8,
    mus: PyReadonlyArray1<f32>,
    spread: u8,
    subset: PyReadonlyArray1<u32>
) -> PyResult<&'py PyArray2<f32>> 
{
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let drs = drs.as_array();
    let type_ids = type_ids.as_array();
    let mus = mus.as_slice()?;
    let subset = subset.as_array();

    let sfs = crate::ml::get_rad_sf_frame_subset(
        nlist_i, nlist_j, drs, type_ids, types, mus, spread, subset
    );
    Ok(sfs.into_pyarray(py))
} 

#[pyfunction(name="spatially_smeared_local_rdfs")]
fn spatially_smeared_local_rdfs_py<'py>(
    py: Python<'py>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    drs: PyReadonlyArray1<f32>,
    type_ids: PyReadonlyArray1<u8>,
    types: u8,
    r_max: f32,
    bins: usize,
    smear_rad: Option<f32>,
    smear_gauss: Option<f32>
) -> PyResult<&'py PyArray3<f32>> 
{
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let drs = drs.as_array();
    let type_ids = type_ids.as_array();

    let rdfs = crate::statics::spatially_smeared_local_rdfs(
        nlist_i, nlist_j, drs, type_ids, types, r_max, bins, smear_rad, smear_gauss
    );

    Ok(rdfs.into_pyarray(py))
}