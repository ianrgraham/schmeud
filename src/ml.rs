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

        update_rad_sf(dr, mus, l, mu_idx as isize, spread as isize, type_id as usize, types as usize, &mut features.index_axis_mut(Axis(0), i))
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