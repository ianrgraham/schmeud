use ndarray::prelude::*;
use std::collections::HashMap;

#[inline(always)]
fn update_rad_sf(dr: f32, mus: &[f32], l: f32, mu_idx: isize, spread: isize, type_id: usize, types: usize, sf: &mut ArrayViewMut1<f32>) {
    for pre_idx in (-spread)..=spread {
        let idx = mu_idx + pre_idx;
        if idx >= 0 && idx < mus.len() as isize {
            let uidx = (idx as usize)*types + type_id;
            let other_idx = idx as usize;
            sf[uidx] += crate::utils::gaussian(dr, mus[other_idx], l);
        }
        else {
            continue
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
    spread: u8
) -> Array2<f32> {
    let l = mus[1] - mus[0];
    let mut features = Array2::<f32>::zeros((type_ids.len(), (types as usize)*mus.len()));

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
            &mut features.index_axis_mut(Axis(0), i)
        );
    }

    features
}

pub struct NeighborList<'a> {
    pub query_point_indices: ArrayView1<'a, u32>,
    pub point_indices: ArrayView1<'a, u32>,
    pub neighbor_counts: ArrayView1<'a, u32>,
    pub segments: ArrayView1<'a, u32>,
    pub distances: ArrayView1<'a, f32>,
}

/// Compute structure functions from 
#[inline(always)]
pub fn radial_sf_snap_generic_nlist(
    nlist: &NeighborList,
    type_id: ArrayView1<u8>,
    types: u8,
    mus: &[f32],
    spread: u8
) -> Array2<f32> {

    let l = mus[1] - mus[0];
    let mut features = Array2::<f32>::zeros((nlist.query_point_indices.raw_dim()[0], (types as usize)*mus.len()));

    for (i, (&head, &nn)) in nlist.segments.iter().zip(nlist.neighbor_counts).enumerate() {
        for j in head..head+nn {

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
                &mut features.index_axis_mut(Axis(0), i)
            );
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
    subset: ArrayView1<u32>
) -> Array2<f32> {

    let hash_subset: HashMap<_,_> = subset.iter().enumerate().map(|(idx, i)| (i, idx)).collect();

    let l = mus[1] - mus[0];
    let mut features = Array2::<f32>::zeros((subset.len(), (types as usize)*mus.len()));

    for idx in 0..nlist_i.len() {
        let i = nlist_i[idx];
        if let Some(feat_idx) = hash_subset.get_key_value(&i) {
            let j = nlist_j[idx] as usize;
            let dr = drs[idx];

            let type_id = type_ids[j];
            let mu_idx = crate::utils::digitize_lin(dr, mus, l);

            update_rad_sf(dr, mus, l, mu_idx as isize, spread as isize, type_id as usize, types as usize, &mut features.index_axis_mut(Axis(0), *feat_idx.1))
        }
    }

    features
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn generic_sf() {
        let query_point_indices: Array1<u32> = vec![0, 0, 1, 1].into();
        let point_indices: Array1<u32> = vec![0, 1, 2, 3].into();
        let neighbor_counts: Array1<u32> = vec![2, 2].into();
        let segments: Array1<u32> = vec![0, 2].into();
        let distances: Array1<f32> = vec![0.2, 1.1, 1.9, 0.5].into();
        let nlist = NeighborList {
            query_point_indices: query_point_indices.view(),
            point_indices: point_indices.view(),
            neighbor_counts: neighbor_counts.view(),
            segments: segments.view(),
            distances: distances.view(),
        };

        let type_id: Array1<u8> = vec![0, 1, 0, 1, 0, 0, 1].into();

        radial_sf_snap_generic_nlist(
            &nlist,
            type_id.view(),
            2,
            &[0.0, 0.5, 1.0, 1.5, 2.0],
            0
        );
    }
}