use fastapprox::faster::erfc;
use ndarray::prelude::*;

#[inline(always)]
fn update_rdf_gauss_smear(
    dr: f32,
    rads: &[f32],
    _l: f32,
    gauss_smear: f32,
    rad_idx: isize,
    spread: isize,
    rdf: &mut ArrayViewMut1<f32>,
) {
    let max_idx = rdf.len() as isize - 1;
    for pre_idx in (-spread)..=spread {
        let idx = rad_idx + pre_idx;
        if idx >= 0 && idx <= max_idx {
            let uidx = idx as usize;
            rdf[uidx] += crate::utils::gaussian(dr, rads[uidx], gauss_smear);
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
    smear_gauss: Option<f32>,
) -> Array3<f32> {
    let rads = itertools_num::linspace::<f32>(r_min, r_max, bins + 1).collect::<Array1<_>>();

    let rads_slice = rads.as_slice().unwrap();

    // Allocate output Array
    let l: f32 = rads[1] - rads[0];
    let l2 = l / 2.0;
    let mid_points = rads.slice(s![..-1]).map(|x| x + l2);
    let mid_point_slice = rads.as_slice().unwrap();

    let mut rdfs = Array3::<f32>::zeros((type_ids.len(), mid_points.len(), types as usize));

    let gauss_smear_tup = if let Some(smear_gauss) = smear_gauss {
        let smear_n = (3.0 * smear_gauss / l) as u32;
        Some((smear_gauss, smear_n))
    } else {
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

            update_rdf_gauss_smear(
                dr,
                mid_point_slice,
                l,
                gauss_smear,
                rad_idx as isize,
                gauss_n as isize,
                &mut rdf_i,
            );
        } else {
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
            if dr > smear_rad {
                continue;
            }

            let rdf_j = rdfs.index_axis(Axis(0), j).to_owned();
            let mut rdf_i = rdfs.index_axis_mut(Axis(0), i);

            azip!((rdf_i in &mut rdf_i, &rdf_j in &rdf_j) *rdf_i += rdf_j);
        }
    }

    rdfs
}

#[inline(always)]
fn excess_entropy(
    rdf: ArrayView1<f32>,
    r: ArrayView1<f32>,
    dr: f32,
    cutoff: Option<(f32, f32)>,
) -> f32 {
    if let Some((y, z)) = cutoff {
        let mut terms = ((&rdf * &rdf.mapv(f32::ln) - &rdf + 1.0) * &r.mapv(|x| x.powi(2)))
            .mapv_into(|x| if x.is_nan() { 0.0 } else { x });
        terms.zip_mut_with(&r.view(), |x, w| *x *= 0.5 * erfc((*w - y) / z));

        terms.sum() * dr
    } else {
        ((&rdf * &rdf.mapv(f32::ln) - &rdf + 1.0) * &r.mapv(|x| x.powi(2)))
            .mapv_into(|x| if x.is_nan() { 0.0 } else { x })
            .sum()
            * dr
    }
}

pub fn delaunay_2d(points: ArrayView2<f32>, _sbox: [f32; 3]) -> Array2<f32> {
    points.to_owned() // temporary
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_excess_entropy() {
        let rdf = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let r = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let dr = 1.0;

        let entropy = excess_entropy(rdf.view(), r.view(), dr, Some((10.0, 0.1)));
        let entropy2 = excess_entropy(rdf.view(), r.view(), dr, None);

        approx::assert_relative_eq!(entropy, entropy2);
    }
}
