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

pub fn spatially_smeared_local_rdfs(
    nlist_i: ArrayView1<u32>,
    nlist_j: ArrayView1<u32>,
    drs: ArrayView1<f32>,
    type_ids: ArrayView1<u8>,
    types: u8,
    r_max: f32,
    bins: usize,
    smear_rad: f32
) -> Array3<f32> {

    let rads = itertools_num::linspace::<f32>(0., r_max, bins + 1).collect::<Array1<_>>();

    let rads_slice = rads.as_slice().unwrap();
    
    // Allocate output Array
    let l: f32 = rads[1] - rads[0];
    let mut rdfs = Array3::<f32>::zeros((type_ids.len(), rads.len(), types as usize));

    // todo: apply the RDF rescaling step in Rust (instead of in python)
    // let bin_centers = rads.slice(s![..-1]).to_owned() + 0.5*l;
    // let div = 4.0*f32::pi*bin_centers.map(|x| x*x)*l;
    
    // let hull = 4.0*f32::pi*r_max*r_max*r_max/3.0;

    // Build initial RDFs
    for idx in 0..nlist_i.len() {
        let i = nlist_i[idx] as usize;
        let j = nlist_j[idx] as usize;
        let dr = drs[idx];

        let type_id = type_ids[j];
        let rad_idx = crate::utils::digitize_lin(dr, rads_slice, l);
        rdfs[(i, rad_idx, type_id as usize)] += 1.0;
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