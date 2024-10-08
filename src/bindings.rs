use numpy::*;
use pyo3::exceptions::*;
use pyo3::prelude::*;

pub fn register_boxdim(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "boxdim")?;

    child_module.add_class::<crate::boxdim::BoxDim>()?;

    parent_module.add_submodule(child_module)?;
    Ok(())
}

pub fn register_nlist(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "nlist")?;

    child_module.add_class::<crate::nlist::NeighborList>()?;
    #[cfg(any(feature = "voro-static", feature = "voro-system"))]
    {
        child_module.add_class::<crate::nlist::voro::Voronoi>()?;
    }

    parent_module.add_submodule(child_module)?;
    Ok(())
}

pub fn register_locality(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "locality")?;

    child_module.add_function(wrap_pyfunction!(particle_to_grid_cube_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        particle_to_grid_cube_cic_py,
        child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(
        particle_to_grid_square_cic_py,
        child_module
    )?)?;
    child_module.add_class::<crate::locality::NeighborList>()?;
    child_module.add_class::<crate::locality::BlockTree>()?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}

pub fn register_dynamics(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "dynamics")?;

    child_module.add_function(wrap_pyfunction!(nonaffine_local_strain_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        affine_local_strain_tensor_py,
        child_module
    )?)?;
    child_module.add_function(wrap_pyfunction!(d2min_frame_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(d2min_and_strain_frame_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(self_intermed_scatter_fn_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(p_hop_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(p_hop_stats_py, child_module)?)?;

    parent_module.add_submodule(child_module)?;
    Ok(())
}

pub fn register_statics(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "statics")?;

    child_module.add_function(wrap_pyfunction!(
        spatially_smeared_local_rdfs_py,
        child_module
    )?)?;

    parent_module.add_submodule(child_module)?;
    Ok(())
}

pub fn register_ml(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "ml")?;

    child_module.add_function(wrap_pyfunction!(get_rad_sf_frame_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(get_rad_sf_frame_subset_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(
        radial_sf_snap_generic_nlist_py,
        child_module
    )?)?;

    parent_module.add_submodule(child_module)?;
    Ok(())
}

#[pyfunction(name = "nonaffine_local_strain")]
fn nonaffine_local_strain_py(
    _py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
) -> PyResult<f32> {
    let x = x.as_array();
    let y = y.as_array();
    crate::dynamics::nonaffine_local_strain(x, y)
        .map_err(|e| PyArithmeticError::new_err(format!("{}", e)))
}

#[pyfunction(name = "affine_local_strain_tensor")]
fn affine_local_strain_tensor_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let x = x.as_array();
    let y = y.as_array();
    match crate::dynamics::affine_local_strain_tensor(x, y) {
        Ok(j) => Ok(j.into_pyarray(py)),
        Err(e) => Err(PyArithmeticError::new_err(format!("{}", e))),
    }
}

#[pyfunction(name = "d2min_frame")]
fn d2min_frame_py<'py>(
    py: Python<'py>,
    inital_pos: PyReadonlyArray2<f32>,
    final_pos: PyReadonlyArray2<f32>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    sboxs: Option<(PyReadonlyArray1<f32>, PyReadonlyArray1<f32>)>,
) -> PyResult<&'py PyArray1<f32>> {
    let inital_pos = inital_pos.as_array();
    let final_pos = final_pos.as_array();
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let sboxs: Option<([f32; 6], [f32; 6])> = match sboxs {
        Some(sboxs) => {
            let tbox;
            let tbox2;
            let arr = sboxs.0.as_array();
            let tmp: &[f32] = arr
                .as_slice()
                .ok_or(PyValueError::new_err("sbox is not contiguous"))?;
            let tmp2: [f32; 6] = tmp.try_into()?;
            tbox = tmp2.clone();
            let arr = sboxs.1.as_array();
            let tmp: &[f32] = arr
                .as_slice()
                .ok_or(PyValueError::new_err("sbox is not contiguous"))?;
            let tmp2: [f32; 6] = tmp.try_into()?;
            tbox2 = tmp2.clone();
            Some((tbox, tbox2))
        }
        None => None,
    };

    match crate::dynamics::d2min_frame(inital_pos, final_pos, nlist_i, nlist_j, sboxs) {
        Ok(d2min) => Ok(d2min.into_pyarray(py)),
        Err(e) => Err(PyArithmeticError::new_err(format!("{}", e))),
    }
}

#[pyfunction(name = "d2min_and_strain_frame")]
fn d2min_and_strain_frame_py<'py>(
    py: Python<'py>,
    inital_pos: PyReadonlyArray2<f32>,
    final_pos: PyReadonlyArray2<f32>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    sboxs: Option<(PyReadonlyArray1<f32>, PyReadonlyArray1<f32>)>,
) -> PyResult<(&'py PyArray1<f32>, &'py PyArray3<f32>)> {
    let inital_pos = inital_pos.as_array();
    let final_pos = final_pos.as_array();
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let sboxs: Option<([f32; 6], [f32; 6])> = match sboxs {
        Some(sboxs) => {
            let tbox;
            let tbox2;
            let arr = sboxs.0.as_array();
            let tmp: &[f32] = arr
                .as_slice()
                .ok_or(PyValueError::new_err("sbox is not contiguous"))?;
            let tmp2: [f32; 6] = tmp.try_into()?;
            tbox = tmp2.clone();
            let arr = sboxs.1.as_array();
            let tmp: &[f32] = arr
                .as_slice()
                .ok_or(PyValueError::new_err("sbox is not contiguous"))?;
            let tmp2: [f32; 6] = tmp.try_into()?;
            tbox2 = tmp2.clone();
            Some((tbox, tbox2))
        }
        None => None,
    };

    match crate::dynamics::d2min_and_strain_frame(inital_pos, final_pos, nlist_i, nlist_j, sboxs) {
        Ok((d2min, strain)) => Ok((d2min.into_pyarray(py), strain.into_pyarray(py))),
        Err(e) => Err(PyArithmeticError::new_err(format!("{}", e))),
    }
}

#[pyfunction(name = "self_intermed_scatter_fn")]
fn self_intermed_scatter_fn_py<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray3<f32>,
    q: f32,
) -> PyResult<&'py PyArray1<f32>> {
    let traj = traj.as_array();
    let sisf = crate::dynamics::self_intermed_scatter_fn(traj, q);
    let sisf = sisf.unwrap();
    Ok(sisf.into_pyarray(py))
}

#[pyfunction(name = "p_hop")]
fn p_hop_py<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray3<f32>,
    tr_frames: usize,
) -> PyResult<&'py PyArray2<f32>> {
    let traj = traj.as_array();
    let phop = crate::dynamics::p_hop(traj, tr_frames);
    Ok(phop.into_pyarray(py))
}

#[pyfunction(name = "p_hop_stats")]
fn p_hop_stats_py(
    traj: PyReadonlyArray3<f32>,
    tr_frames: usize,
    cutoff: f32,
) -> PyResult<Vec<(f32, f32, f32)>> {
    let traj = traj.as_array();
    let phop = crate::dynamics::p_hop_stats(traj, tr_frames, cutoff);
    Ok(phop)
}

#[pyfunction(name = "get_rad_sf_frame")]
fn get_rad_sf_frame_py<'py>(
    py: Python<'py>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    drs: PyReadonlyArray1<f32>,
    type_ids: PyReadonlyArray1<u8>,
    types: u8,
    mus: PyReadonlyArray1<f32>,
    spread: u8,
) -> PyResult<&'py PyArray2<f32>> {
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let drs = drs.as_array();
    let type_ids = type_ids.as_array();
    let mus = mus.as_slice()?;

    let sfs = crate::ml::get_rad_sf_frame(nlist_i, nlist_j, drs, type_ids, types, mus, spread);
    Ok(sfs.into_pyarray(py))
}

#[pyfunction(name = "radial_sf_snap_generic_nlist")]
fn radial_sf_snap_generic_nlist_py<'py>(
    py: Python<'py>,
    query_point_indices: PyReadonlyArray1<u32>,
    point_indices: PyReadonlyArray1<u32>,
    neighbor_counts: PyReadonlyArray1<u32>,
    segments: PyReadonlyArray1<u32>,
    distances: PyReadonlyArray1<f32>,
    type_id: PyReadonlyArray1<u8>,
    types: u8,
    mus: PyReadonlyArray1<f32>,
    spread: u8,
) -> PyResult<&'py PyArray2<f32>> {
    let query_point_indices = query_point_indices.as_array();
    let point_indices = point_indices.as_array();
    let neighbor_counts = neighbor_counts.as_array();
    let segments = segments.as_array();
    let distances = distances.as_array();
    let type_id = type_id.as_array();
    let mus = mus.as_slice()?;

    let nlist = crate::ml::FreudNeighborListView {
        query_point_indices,
        point_indices,
        neighbor_counts,
        segments,
        distances,
    };

    let sfs = crate::ml::radial_sf_snap_generic_nlist(&nlist, type_id, types, mus, spread);
    Ok(sfs.into_pyarray(py))
}

#[pyfunction(name = "get_rad_sf_frame_subset")]
fn get_rad_sf_frame_subset_py<'py>(
    py: Python<'py>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    drs: PyReadonlyArray1<f32>,
    type_ids: PyReadonlyArray1<u8>,
    types: u8,
    mus: PyReadonlyArray1<f32>,
    spread: u8,
    subset: PyReadonlyArray1<u32>,
) -> PyResult<&'py PyArray2<f32>> {
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let drs = drs.as_array();
    let type_ids = type_ids.as_array();
    let mus = mus.as_slice()?;
    let subset = subset.as_array();

    let sfs = crate::ml::get_rad_sf_frame_subset(
        nlist_i, nlist_j, drs, type_ids, types, mus, spread, subset,
    );
    Ok(sfs.into_pyarray(py))
}

#[pyfunction(name = "spatially_smeared_local_rdfs")]
fn spatially_smeared_local_rdfs_py<'py>(
    py: Python<'py>,
    nlist_i: PyReadonlyArray1<u32>,
    nlist_j: PyReadonlyArray1<u32>,
    drs: PyReadonlyArray1<f32>,
    type_ids: PyReadonlyArray1<u8>,
    types: u8,
    r_min: f32,
    r_max: f32,
    bins: usize,
    smear_rad: Option<f32>,
    smear_gauss: Option<f32>,
) -> PyResult<&'py PyArray3<f32>> {
    let nlist_i = nlist_i.as_array();
    let nlist_j = nlist_j.as_array();
    let drs = drs.as_array();
    let type_ids = type_ids.as_array();

    let rdfs = crate::statics::spatially_smeared_local_rdfs(
        nlist_i,
        nlist_j,
        drs,
        type_ids,
        types,
        r_min,
        r_max,
        bins,
        smear_rad,
        smear_gauss,
    );

    Ok(rdfs.into_pyarray(py))
}

#[pyfunction(name = "particle_to_grid_cube")]
fn particle_to_grid_cube_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    values: PyReadonlyArray1<f32>,
    l: f32,
    bins: usize,
) -> PyResult<&'py PyArray3<f32>> {
    let points = points.as_array();
    let values = values.as_array();

    let grid = crate::locality::particle_to_grid_cube(points, values, l, bins);
    Ok(grid.into_pyarray(py))
}

#[pyfunction(name = "particle_to_grid_cube_cic")]
fn particle_to_grid_cube_cic_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    values: PyReadonlyArray1<f32>,
    l: f32,
    bins: usize,
) -> PyResult<&'py PyArray3<f32>> {
    let points = points.as_array();
    let values = values.as_array();

    let grid = crate::locality::particle_to_grid_cube_cic(points, values, l, bins);
    Ok(grid.into_pyarray(py))
}

#[pyfunction(name = "particle_to_grid_square_cic")]
fn particle_to_grid_square_cic_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    values: PyReadonlyArray1<f32>,
    l: f32,
    bins: usize,
) -> PyResult<&'py PyArray2<f32>> {
    let points = points.as_array();
    let values = values.as_array();

    let grid = crate::locality::particle_to_grid_square_cic(points, values, l, bins);
    Ok(grid.into_pyarray(py))
}
