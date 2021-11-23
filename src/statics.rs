use num::{Float, Zero};
use ndarray::prelude::*;
use ndarray::Zip;

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