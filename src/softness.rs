use ndarray::prelude::*;
use std::collections::HashMap;

#[inline(always)]
fn rad_sf(dr: f32, mu: f32, l: f32) -> f32 {
    let term = (dr-mu)/l;
    (-term*term*0.5).exp()
}

#[inline(always)]
fn update_rad_sf(dr: f32, mus: &[f32], l: f32, mu_idx: isize, spread: isize, type_id: usize, types: usize, sf: &mut ArrayViewMut1<f32>) {
    for pre_idx in (-spread)..=spread {
        let idx = mu_idx + pre_idx;
        if idx >= 0 && idx < mus.len() as isize {
            let uidx = (idx as usize)*types + type_id;
            let other_idx = idx as usize;
            sf[uidx] += rad_sf(dr, mus[other_idx], l);
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

#[inline(always)]
fn update_rdf_gauss_smear(dr: f32, rads: &[f32], gauss_smear: f32, rad_idx: isize, spread: isize, rdf: &mut ArrayViewMut1<f32>) {
    let max_idx = rads.len() as isize - 1;
    for pre_idx in (-spread)..=spread {
        let idx = rad_idx + pre_idx;
        if idx >= 0 && idx < max_idx {
            let uidx = idx as usize;
            rdf[uidx] += rad_sf(dr, rads[uidx], gauss_smear);
        }
        else {
            continue
        }
    }
}

pub fn spatially_smeared_local_rdfs(
    nlist_i: ArrayView1<u32>,
    nlist_j: ArrayView1<u32>,
    drs: ArrayView1<f32>,
    type_ids: ArrayView1<u8>,
    types: u8,
    r_max: f32,
    bins: usize,
    smear_rad: f32,
    smear_gauss: Option<f32>
) -> Array3<f32> {

    let rads = itertools_num::linspace::<f32>(0., r_max, bins + 1).collect::<Array1<_>>();

    let rads_slice = rads.as_slice().unwrap();
    
    // Allocate output Array
    let l: f32 = rads[1] - rads[0];
    let mut rdfs = Array3::<f32>::zeros((type_ids.len(), rads.len() - 1, types as usize));

    let gauss_smear_tup = if let Some(smear_gauss) = smear_gauss {
        let smear_n = (3.0*smear_gauss/l) as u32;
        Some((smear_gauss, smear_n))
    }
    else {
        None
    };

    // Build initial RDFs
    for idx in 0..nlist_i.len() {
        let i = nlist_i[idx] as usize;
        let j = nlist_j[idx] as usize;
        let dr = drs[idx];

        let type_id = type_ids[j];
        if let Some((gauss_smear, gauss_n)) = gauss_smear_tup {
            let rad_idx = crate::utils::digitize_lin(dr, rads_slice, l);

            let mut rdf_i = rdfs.slice_mut(s![i, .., type_id as usize]);

            update_rdf_gauss_smear(dr, rads_slice, gauss_smear, rad_idx as isize, gauss_n as isize, &mut rdf_i);
        }
        else {
            if let Some(rad_idx) = crate::utils::try_digitize_lin(dr, rads_slice, l) {
                rdfs[(i, rad_idx, type_id as usize)] += 1.0;
            }
        }
    }

    // Smear RDFs
    for idx in 0..nlist_i.len() {
        let i = nlist_i[idx] as usize;
        let j = nlist_j[idx] as usize;
        let dr = drs[idx];
        if dr > smear_rad { continue }


        let rdf_j = rdfs.index_axis(Axis(0), j).to_owned();
        let mut rdf_i = rdfs.index_axis_mut(Axis(0), i);

        azip!((rdf_i in &mut rdf_i, &rdf_j in &rdf_j) *rdf_i += rdf_j);
    }

    rdfs
}