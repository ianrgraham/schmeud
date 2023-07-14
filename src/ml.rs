use ndarray::prelude::*;
use ndarray::Zip;
use std::collections::HashMap;

#[inline(always)]
fn update_rad_sf(
    dr: f32,
    mus: &[f32],
    l: f32,
    mu_idx: isize,
    spread: isize,
    type_id: usize,
    types: usize,
    sf: &mut ArrayViewMut1<f32>,
) {
    for pre_idx in (-spread)..=spread {
        let idx = mu_idx + pre_idx;
        if idx >= 0 && idx < mus.len() as isize {
            let uidx = (idx as usize) * types + type_id;
            let other_idx = idx as usize;
            sf[uidx] += crate::utils::gaussian(dr, mus[other_idx], l);
        } else {
            continue;
        }
    }
}

/// Computes structure functions for a given set of particles.
///
/// __Note:__ This function assumes that that the labels among `nlist_i` and
/// `nlist_j` point to the same particles.
///
/// Useful when the structure function
#[inline(always)]
pub fn get_rad_sf_frame(
    nlist_i: ArrayView1<u32>,
    nlist_j: ArrayView1<u32>,
    drs: ArrayView1<f32>,
    type_ids: ArrayView1<u8>,
    types: u8,
    mus: &[f32],
    spread: u8,
) -> Array2<f32> {
    let l = mus[1] - mus[0];
    let mut features = Array2::<f32>::zeros((type_ids.len(), (types as usize) * mus.len()));

    for idx in 0..nlist_i.len() {
        let i = nlist_i[idx] as usize;
        let j = nlist_j[idx] as usize;
        let dr = drs[idx];

        let type_id = type_ids[j];
        let mu_idx = crate::utils::digitize_lin(dr, mus, l);

        update_rad_sf(
            dr,
            mus,
            l,
            mu_idx as isize,
            spread as isize,
            type_id as usize,
            types as usize,
            &mut features.index_axis_mut(Axis(0), i),
        );
    }

    features
}

pub struct FreudNeighborListView<'a> {
    pub query_point_indices: ArrayView1<'a, u32>,
    pub point_indices: ArrayView1<'a, u32>,
    pub neighbor_counts: ArrayView1<'a, u32>,
    pub segments: ArrayView1<'a, u32>,
    pub distances: ArrayView1<'a, f32>,
}

/// Compute structure functions from
#[inline(always)]
pub fn radial_sf_snap_generic_nlist(
    nlist: &FreudNeighborListView,
    type_id: ArrayView1<u8>,
    types: u8,
    mus: &[f32],
    spread: u8,
) -> Array2<f32> {
    let l = mus[1] - mus[0];
    let mut features =
        Array2::<f32>::zeros((nlist.segments.raw_dim()[0], (types as usize) * mus.len()));

    if cfg!(feature = "rayon") {
        Zip::from(features.rows_mut())
            .and(nlist.segments)
            .and(nlist.neighbor_counts)
            .par_for_each(|mut sf, &head, &nn| {
                for j in head..head + nn {
                    let dr = nlist.distances[j as usize];
                    let type_id = type_id[nlist.point_indices[j as usize] as usize];
                    let mu_idx = crate::utils::digitize_lin(dr, mus, l);
                    update_rad_sf(
                        dr,
                        mus,
                        l,
                        mu_idx as isize,
                        spread as isize,
                        type_id as usize,
                        types as usize,
                        &mut sf,
                    );
                }
            });
    } else {
        for (i, (&head, &nn)) in nlist.segments.iter().zip(nlist.neighbor_counts).enumerate() {
            for j in head..head + nn {
                let dr = nlist.distances[j as usize];
                let type_id = type_id[nlist.point_indices[j as usize] as usize];
                let mu_idx = crate::utils::digitize_lin(dr, mus, l);

                update_rad_sf(
                    dr,
                    mus,
                    l,
                    mu_idx as isize,
                    spread as isize,
                    type_id as usize,
                    types as usize,
                    &mut features.index_axis_mut(Axis(0), i),
                );
            }
        }
    }

    features
}

#[inline(always)]
pub fn get_rad_sf_frame_subset(
    nlist_i: ArrayView1<u32>,
    nlist_j: ArrayView1<u32>,
    drs: ArrayView1<f32>,
    type_ids: ArrayView1<u8>,
    types: u8,
    mus: &[f32],
    spread: u8,
    subset: ArrayView1<u32>,
) -> Array2<f32> {
    let hash_subset: HashMap<_, _> = subset.iter().enumerate().map(|(idx, i)| (i, idx)).collect();

    let l = mus[1] - mus[0];
    let mut features = Array2::<f32>::zeros((subset.len(), (types as usize) * mus.len()));

    for idx in 0..nlist_i.len() {
        let i = nlist_i[idx];
        if let Some(feat_idx) = hash_subset.get_key_value(&i) {
            let j = nlist_j[idx] as usize;
            let dr = drs[idx];

            let type_id = type_ids[j];
            let mu_idx = crate::utils::digitize_lin(dr, mus, l);

            update_rad_sf(
                dr,
                mus,
                l,
                mu_idx as isize,
                spread as isize,
                type_id as usize,
                types as usize,
                &mut features.index_axis_mut(Axis(0), *feat_idx.1),
            )
        }
    }

    features
}

#[cfg(test)]
mod test {
    use super::*;
    use pyo3::{prelude::*, types::PyModule};

    #[test]
    fn test_freud_in_rust() -> PyResult<()> {
        // initialize freud python nlist in rust
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let gsd_hoomd = PyModule::import(py, "gsd.hoomd").expect("Failed to import gsd.hoomd");
            let traj = gsd_hoomd
                .getattr("open")
                .unwrap()
                .call1(("tests/data/traj.gsd",))
                .unwrap();
            let snap = traj.get_item(0).unwrap();
            println!("{:?} {:?}", traj, snap);
            let freud = PyModule::import(py, "freud").expect("Failed to import freud");
            let nlist_query = freud
                .getattr("locality")?
                .getattr("AABBQuery")?
                .getattr("from_system")?
                .call1((snap,))?;
            println!("{:?}", nlist_query);

            let nlist_shim = PyModule::from_code(
                py,
                r#"
import freud
import gsd.hoomd

def nlist_query(filename):
    traj = gsd.hoomd.open(filename)
    snap = traj[0]
    return freud.locality.AABBQuery.from_system(snap)
                "#,
                "nlist_shim.py",
                "nlist_shim",
            )?;

            let nlist_query = nlist_shim
                .getattr("nlist_query")?
                .call1(("tests/data/traj.gsd",))?;
            println!("{:?}", nlist_query);

            Ok(())
        })
    }

    #[test]
    fn generic_sf() {
        let query_point_indices: Array1<u32> = vec![0, 0, 1, 1].into();
        let point_indices: Array1<u32> = vec![0, 1, 2, 3].into();
        let neighbor_counts: Array1<u32> = vec![2, 2].into();
        let segments: Array1<u32> = vec![0, 2].into();
        let distances: Array1<f32> = vec![0.2, 1.1, 1.9, 0.5].into();
        let nlist = FreudNeighborListView {
            query_point_indices: query_point_indices.view(),
            point_indices: point_indices.view(),
            neighbor_counts: neighbor_counts.view(),
            segments: segments.view(),
            distances: distances.view(),
        };

        let type_id: Array1<u8> = vec![0, 1, 0, 1, 0, 0, 1].into();

        radial_sf_snap_generic_nlist(&nlist, type_id.view(), 2, &[0.0, 0.5, 1.0, 1.5, 2.0], 0);
    }
}
