use num::{Float, Zero};
use ndarray::prelude::*;
use ndarray::Zip;

#[inline(always)]
fn update_rdf_gauss_smear(dr: f32, rads: &[f32], l: f32, gauss_smear: f32, rad_idx: isize, spread: isize, rdf: &mut ArrayViewMut1<f32>) {
    let max_idx = rdf.len() as isize - 1;
    let max_uidx = max_idx as usize;
    let mut uidx = 0;
    for pre_idx in (-spread)..=spread {
        let idx = rad_idx + pre_idx;
        if idx >= 0 && idx <= max_idx {
            uidx = idx as usize;
            rdf[uidx] += crate::utils::gauss_smear(dr, rads[uidx], gauss_smear);
        }
        
    }
}

#[inline(always)]
pub fn spatially_smeared_local_rdfs(
    nlist_i: ArrayView1<u32>,
    nlist_j: ArrayView1<u32>,
    drs: ArrayView1<f32>,
    type_ids: ArrayView1<u8>,
    types: u8,
    r_min: f32,
    r_max: f32,
    bins: usize,
    smear_rad: Option<f32>,
    smear_gauss: Option<f32>
) -> Array3<f32> {

    let rads = itertools_num::linspace::<f32>(r_min, r_max, bins + 1).collect::<Array1<_>>();

    let rads_slice = rads.as_slice().unwrap();
    
    // Allocate output Array
    let l: f32 = rads[1] - rads[0];
    let l2 = l/2.0;
    let mid_points = rads.slice(s![..-1]).map(|x| x + l2);
    let mid_point_slice = rads.as_slice().unwrap();

    let mut rdfs = Array3::<f32>::zeros((type_ids.len(), mid_points.len(), types as usize));

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
            let rad_idx = crate::utils::digitize_lin(dr, mid_point_slice, l);

            let mut rdf_i = rdfs.slice_mut(s![i, .., type_id as usize]);

            update_rdf_gauss_smear(dr, mid_point_slice, l, gauss_smear, rad_idx as isize, gauss_n as isize, &mut rdf_i);
        }
        else {
            if let Some(rad_idx) = crate::utils::try_digitize_lin(dr, rads_slice, l) {
                rdfs[(i, rad_idx, type_id as usize)] += 1.0;
            }
        }
    }

    // Smear RDFs
    if let Some(smear_rad) = smear_rad {
        for idx in 0..nlist_i.len() {
            let i = nlist_i[idx] as usize;
            let j = nlist_j[idx] as usize;
            let dr = drs[idx];
            if dr > smear_rad { continue }


            let rdf_j = rdfs.index_axis(Axis(0), j).to_owned();
            let mut rdf_i = rdfs.index_axis_mut(Axis(0), i);

            azip!((rdf_i in &mut rdf_i, &rdf_j in &rdf_j) *rdf_i += rdf_j);
        }
    }

    rdfs
}

// This is pretty inefficient
fn naive_structure_factor<T: Float>(
    pos: ArrayView2<T>,
    q: T
) -> T {

    let mut output = Zero::zero();

    let n = pos.len_of(Axis(0));
    let two = T::from(2.).unwrap();
    let flt_n = T::from(n).unwrap();
    for i in 0..(n-1) {
        for j in (i+1)..n {
            let term = q*(&pos.slice(s![i, ..]) - &pos.slice(s![j, ..]))
                .fold(Zero::zero(), |sum: T, x| sum + x.powi(2)).sqrt();

            output = output + term.sin()/term;
        }
    }

    output = two*output/flt_n;
    output
}

pub fn structure_factor<T: Float>(
    nlist_i: ArrayView1<usize>,
    nlist_j: ArrayView1<usize>,
    pos: ArrayView2<T>,
    q: T
) -> T {

    let mut output = Zero::zero();

    let n = pos.len_of(Axis(0));
    let flt_n = T::from(n).unwrap();
    Zip::from(&nlist_i).and(&nlist_j).for_each(|&i, &j| {
        let term = q*(&pos.slice(s![i, ..]) - &pos.slice(s![j, ..]))
            .fold(Zero::zero(), |sum: T, x| sum + x.powi(2)).sqrt();

        output = output + term.sin()/term;
    });

    output = output/flt_n;
    output
}