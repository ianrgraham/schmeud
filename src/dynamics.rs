use ndarray::prelude::*;
use ndarray::{Slice, Zip};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::least_squares::LeastSquaresSvd;
use ndarray_linalg::*;
use num::{Float, Zero};
use glam::*;

struct BoxDim {
    l: Vec3,
    tilt: Vec3,
    l_inv: Vec3,
    is_2d: bool
}

impl BoxDim {
    pub fn new(sbox: &[f32; 6]) -> Self {
        let is_2d = sbox[2] == 0.0;
        let l = Vec3::from_slice(&sbox[..3]);
        let tilt = Vec3::from_slice(&sbox[3..]);
        let l_inv = Vec3::new(1.0/l.x, 1.0/l.y, 1.0/l.z);
        Self {
            l,
            tilt,
            l_inv,
            is_2d
        }
    }

    #[inline(always)]
    pub fn min_image_array(&self, w: &mut ArrayViewMut1<f32>) {

        if !self.is_2d {
            let img = (w[2] * self.l_inv.z).round();
            w[2] -= self.l.z * img;
            w[1] -= self.l.z * self.tilt.z * img;
            w[0] -= self.l.z * self.tilt.y * img;
        }

        let img = (w[1] * self.l_inv.y).round();
        w[1] -= self.l.y * img;
        w[0] -= self.l.y * self.tilt.x * img;

        let img = (w[0] * self.l_inv.x).round();
        w[0] -= self.l.x * img;
    }

    #[inline(always)]
    pub fn min_image(&self, w: &mut Vec3) {
        let mut img = 0.0;

        if !self.is_2d {
            img = (w.z * self.l_inv.z).round();
            w.z -= self.l.z * img;
            w.y -= self.l.z * self.tilt.z * img;
            w.x -= self.l.z * self.tilt.y * img;
        }

        img = (w.y * self.l_inv.y).round();
        w.y -= self.l.y * img;
        w.x -= self.l.y * self.tilt.x * img;

        img = (w.x * self.l_inv.x).round();
        w.x -= self.l.x * img;
    }

    #[inline(always)]
    pub fn min_image_2d(&self, w: &mut Vec2) {
        let mut img = 0.0;

        img = (w.y * self.l_inv.y).round();
        w.y -= self.l.y * img;
        w.x -= self.l.y * self.tilt.x * img;

        img = (w.x * self.l_inv.x).round();
        w.x -= self.l.x * img;
    }
}

/// Get D2min for an entire configuration
///
/// This is a fairly general implementation of D2min. It accepts an arbitrary numbers of neighbors per particle.
// This implementation is quite rough though.
// I'd like to do a similar refactor job with a couple of the other codes that
// depend up a NeighborList.
// It might be a good idea to just completely disentangle the query_point and
// point positions, since sometimes they are different and sometimes they are
// the same and I'd rather make it explicit that they may be the same or
// different.
// There's also an opportunity to parallelize this loop if we use
// nlist.segments and nlist.neighbor_counts
pub fn d2min_frame(
    initial_pos: ArrayView2<f32>,
    final_pos: ArrayView2<f32>,
    nlist_i: ArrayView1<u32>,
    nlist_j: ArrayView1<u32>,
    sboxs: Option<([f32; 6], [f32; 6])>
) -> Result<Array1<f32>, LinalgError> {
    // Get sizes and allocate space
    let dim2 = initial_pos.raw_dim();
    let mut out = Array1::<f32>::zeros(dim2[0]);
    let dim = dim2[1];
    let mut nlist = Vec::<Vec<u32>>::with_capacity(dim2[0]);
    nlist.push(vec![]);

    let box_dim = match sboxs {
        Some(sboxs) => Some((BoxDim::new(&sboxs.0), BoxDim::new(&sboxs.1))),
        None => None
    };

    // Build up Vec-based nlist from indices
    let mut cidx = 0;
    for (i, j) in nlist_i.iter().zip(nlist_j) {
        while cidx < (*i as usize) {
            nlist.push(vec![]);
            cidx += 1;
        }
        nlist[cidx].push(*j)
    }

    // Loop over nlist, build bonds, and compute least squared
    for (idx, ids) in nlist.into_iter().enumerate() {
        let pos_i_init = initial_pos.row(idx);
        let pos_i_final = final_pos.row(idx);
        let mut init_bonds = Array2::<f32>::zeros((ids.len(), dim));
        let mut final_bonds = Array2::<f32>::zeros((ids.len(), dim));
        if let Some((box_dim_i, box_dim_f)) = &box_dim {
            for (jdx, j) in ids.into_iter().enumerate() {
                Zip::from(initial_pos.row(j as usize))
                    .and(pos_i_init)
                    .and(&mut init_bonds.row_mut(jdx))
                    .for_each(|x1, x2, y| *y = x1 - x2);
                
                box_dim_i.min_image_array(&mut init_bonds.row_mut(jdx).view_mut());
                

                Zip::from(final_pos.row(j as usize))
                    .and(pos_i_final)
                    .and(final_bonds.row_mut(jdx))
                    .for_each(|x1, x2, y| *y = x1 - x2);

                box_dim_f.min_image_array(&mut final_bonds.row_mut(jdx).view_mut());
            }
        }
        else {
            for (jdx, j) in ids.into_iter().enumerate() {
                Zip::from(initial_pos.row(j as usize))
                    .and(pos_i_init)
                    .and(&mut init_bonds.row_mut(jdx))
                    .for_each(|x1, x2, y| *y = x1 - x2);

                Zip::from(final_pos.row(j as usize))
                    .and(pos_i_final)
                    .and(final_bonds.row_mut(jdx))
                    .for_each(|x1, x2, y| *y = x1 - x2);
            }
        }
        let result = init_bonds.least_squares(&final_bonds)?;
        out[idx] = match result.residual_sum_of_squares {
            Some(x) => x.sum(),
            None => 0.0,
        }
    }

    return Ok(out);
}

#[inline(always)]
pub fn affine_local_strain_tensor<T: Float + Scalar + Lapack>(
    initial_vectors: ArrayView2<T>,
    final_vectors: ArrayView2<T>,
) -> Result<Array2<T>, LinalgError> {
    let v = initial_vectors.t().dot(&initial_vectors);
    let w = initial_vectors.t().dot(&final_vectors);
    Ok(v.inv()?.dot(&w))
}

#[inline(always)]
pub fn nonaffine_and_affine_local_strain<T: Float + Scalar + Lapack>(
    initial_vectors: ArrayView2<T>,
    final_vectors: ArrayView2<T>,
) -> Result<(T, Array2<T>), LinalgError> {
    let j = affine_local_strain_tensor(initial_vectors, final_vectors)?;
    let non_affine = initial_vectors.dot(&j) - initial_vectors;
    let d2min = non_affine
        .iter()
        .fold(Zero::zero(), |sum: T, x| sum + (*x) * (*x));

    Ok((d2min, j))
}

#[inline(always)]
pub fn nonaffine_local_strain<T: Float + Scalar + Lapack>(
    initial_bonds: ArrayView2<T>,
    final_bonds: ArrayView2<T>,
) -> Result<T, LinalgError> {
    let (d2min, _) = nonaffine_and_affine_local_strain::<T>(initial_bonds, final_bonds)?;
    Ok(d2min)
}

pub fn p_hop(pos: ArrayView3<f32>, tr_frames: usize) -> Array2<f32> {
    if tr_frames % 2 == 0 {
        panic!("tr_frames must be odd");
    }

    let shape = pos.shape();
    let n_frames = shape[0];
    let half = tr_frames / 2;

    let mut phop_result = Array2::<f32>::zeros((n_frames - tr_frames, shape[1]));

    for i in 0..(phop_result.shape()[0]) {
        let r_a = pos.slice_axis(Axis(0), Slice::from(i..i + half + 1));
        let r_b = pos.slice_axis(Axis(0), Slice::from(i + half..=i + tr_frames));

        let r_a_mean = r_a.mean_axis(Axis(0)).unwrap();
        let r_b_mean = r_b.mean_axis(Axis(0)).unwrap();

        let phop = ((&r_a - &r_b_mean)
            .mapv(f32::square)
            .sum_axis(Axis(2))
            .mean_axis(Axis(0))
            .unwrap()
            * (&r_b - &r_a_mean)
                .mapv(f32::square)
                .sum_axis(Axis(2))
                .mean_axis(Axis(0))
                .unwrap())
        .mapv(f32::sqrt);

        phop_result.row_mut(i).assign(&phop);
    }

    return phop_result;
}

pub fn self_intermed_scatter_fn<T: Float>(pos: ArrayView3<T>, q: T) -> Result<Array1<T>, ()> {
    let time = pos.len_of(Axis(0));

    let mut output = Array1::<T>::zeros(time - 1);

    let n = pos.len_of(Axis(1));
    let flt_n = T::from(n).unwrap();
    for i in 0..n {
        for t in 1..time {
            let term = q
                * (&pos.slice(s![t, i, ..]) - &pos.slice(s![0, i, ..]))
                    .fold(Zero::zero(), |sum: T, x| sum + x.powi(2))
                    .sqrt();

            output[t] = output[t] + term.sin() / term;
        }
    }

    output.mapv_inplace(|y| y / flt_n);
    Ok(output)
}

/// The Van Hove self-correlation function.
///
pub fn self_van_hove_corr_fn<T: Float>(
    _traj: ArrayView3<T>,
    _bins: ArrayView1<T>,
) -> Result<Array2<T>, ()> {
    Ok(arr2(&[[Zero::zero()]]))
}

/// Mean-squared displacement.
///
pub fn msd_normal<T: Float>(_traj: ArrayView3<T>) -> Result<Array1<T>, ()> {
    Ok(arr1(&[Zero::zero()]))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn phop_one_particle() {
        let traj = [vec![[[1.0, 2.0]]; 20], vec![[[2.0, 0.0]]; 20]].concat();

        let traj = Array3::from(traj);

        let phop_res = p_hop(traj.view(), 11);

        assert_eq!(phop_res.shape(), &[29, 1]);
    }

    #[test]
    fn phop_two_particles() {
        let traj = [
            vec![[[1.0, 2.0], [1.0, 1.0]]; 20],
            vec![[[2.0, 0.0], [1.0, 1.0]]; 20],
        ]
        .concat();

        let traj = Array3::from(traj);

        let phop_res = p_hop(traj.view(), 11);

        assert_eq!(phop_res.shape(), &[29, 2]);
    }
}
