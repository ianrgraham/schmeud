use criterion::{black_box, criterion_group, criterion_main, Criterion};
use schmeud::ml;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};
use numpy::{PyArray1, PyReadonlyArray1};
use ndarray::Array;

use std::env;

fn build_nlist() -> PyResult<Py<PyTuple>> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {

        let nlist_shim = PyModule::from_code(py, r#"
import freud
import gsd.hoomd
import numpy as np

def unpack_nlist(nlist):
    query_point_indices = nlist.query_point_indices
    point_indices = nlist.point_indices
    neighbor_counts = nlist.neighbor_counts
    segments = nlist.segments
    distances = nlist.distances
    return query_point_indices, point_indices, neighbor_counts, segments, distances

def nlist_data(filename):
    traj = gsd.hoomd.open(filename)
    snap = traj[0]
    nlist_query = freud.locality.AABBQuery.from_system(snap)
    nlist = nlist_query.query(snap.particles.position, dict(mode='ball', r_max=2.0)).toNeighborList()
    typeid = snap.particles.typeid.astype(np.uint8)
    return unpack_nlist(nlist) + (typeid,)
            "#, "nlist_shim.py", "nlist_shim")?;

        let nlist_data =
            nlist_shim.getattr("nlist_data")?.call1(("tests/data/traj.gsd",))?;
        Ok(nlist_data.into_py(py).extract(py).unwrap())
    })
}

pub fn criterion_benchmark(c: &mut Criterion) {

    let nlist_data = build_nlist().unwrap();
    Python::with_gil(|py| {
        let nlist_data: &PyTuple = nlist_data.as_ref(py);
        let query_point_indices: PyReadonlyArray1<u32> = nlist_data.get_item(0).unwrap().extract().unwrap();
        let point_indices: PyReadonlyArray1<u32> = nlist_data.get_item(1).unwrap().extract().unwrap();
        let neighbor_counts: PyReadonlyArray1<u32> = nlist_data.get_item(2).unwrap().extract().unwrap();
        let segments: PyReadonlyArray1<u32> = nlist_data.get_item(3).unwrap().extract().unwrap();
        let distances: PyReadonlyArray1<f32> = nlist_data.get_item(4).unwrap().extract().unwrap();
        let typeid: PyReadonlyArray1<u8> = nlist_data.get_item(5).unwrap().extract().unwrap();
        let nlist = ml::FreudNeighborListView {
            query_point_indices: query_point_indices.as_array(),
            point_indices: point_indices.as_array(),
            neighbor_counts: neighbor_counts.as_array(),
            segments: segments.as_array(),
            distances: distances.as_array()
        };
        let typeid = typeid.as_array();

        let mus = Array::linspace(0.3, 3.0, 28);
        let mus = mus.as_slice().unwrap();

        c.bench_function("sf-prod", |b| b.iter(||
            ml::radial_sf_snap_generic_nlist(
                black_box(&nlist),
                black_box(typeid),
                black_box(2u8),
                black_box(mus),
                black_box(3u8)
            )
        ));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);