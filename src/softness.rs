use ndarray::prelude::*;

#[inline(always)]
fn rad_sf(dr: f32, mu: f32, l: f32) -> f32 {
    let term = (dr-mu)/l;
    (-term*term*0.5).exp()
}

#[inline(always)]
fn update_rad_sf(dr: f32, mus: &[f32], l: f32, mu_idx: isize, spread: isize, type_id: usize, types: usize, sf: &mut ArrayViewMut1<f32>) {
    for pre_idx in (-spread)..spread {
        let idx = mu_idx + pre_idx;
        if idx >= 0 || idx < mus.len() as isize {
            let uidx = (idx as usize)*types + type_id;
            sf[uidx] += rad_sf(dr, mus[uidx], l);
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
