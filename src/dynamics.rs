use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::least_squares::LeastSquaresSvd;
use ndarray_linalg::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::{Float, Zero};
use std::time;

/// Just an example calculating D2min in highly parallel manner.
/// 
/// Don't actually use this!
fn _d2min_system_example() {
    // fake system
    let n = 10000usize;
    let a_big = Array::random((n, 10, 2), Uniform::<f64>::new(-2., 2.));
    let b_big = &a_big + Array::random((n, 10, 2), Uniform::new(-0.5, 0.5));
    let mut out = Array1::<f64>::zeros(n);

    let start = time::Instant::now();
    Zip::from(&mut out)
        .and(a_big.axis_iter(Axis(0)))
        .and(b_big.axis_iter(Axis(0)))
        .par_for_each(|o, a, b| {
            let result = a.least_squares(&b).unwrap();
            *o = result.residual_sum_of_squares.unwrap().sum();
        });

    println!("{:?}", start.elapsed());
    println!("{}", out)
}

#[inline(always)]
pub fn affine_local_strain<T: Float + Scalar + Lapack>(
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
    let j = affine_local_strain(initial_vectors, final_vectors)?;
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
/// 
pub fn self_van_hove_corr_fn<T: Float>(
    _traj: ArrayView3<T>,
    _bins: ArrayView1<T>,
) -> Result<Array2<T>, ()> {
    Ok(arr2(&[[Zero::zero()]]))
}

pub fn msd<T: Float>(_traj: ArrayView3<T>) -> Result<Array1<T>, ()> {
    Ok(arr1(&[Zero::zero()]))
}
