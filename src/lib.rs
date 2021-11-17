use pyo3::prelude::*;
use pyo3::exceptions::*;
use numpy::*;


#[pymodule]
fn schmeud(py: Python, m: &PyModule) -> PyResult<()> {

    register_dynamics(py, m)?;

    register_statics(py, m)?;

    register_ml(py, m)?;
    
    Ok(())
}

fn register_dynamics(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "dynamics")?;
    child_module.add_function(
        wrap_pyfunction!(nonaffine_local_strain_py, child_module)?
    )?;
    child_module.add_function(
        wrap_pyfunction!(affine_local_strain_py, child_module)?
    )?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}

fn register_statics(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "statics")?;
    child_module.add_function(
        wrap_pyfunction!(nonaffine_local_strain_py, child_module)?
    )?;
    child_module.add_function(
        wrap_pyfunction!(affine_local_strain_py, child_module)?
    )?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}

fn register_ml(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "ml")?;
    child_module.add_function(
        wrap_pyfunction!(nonaffine_local_strain_py, child_module)?
    )?;
    child_module.add_function(
        wrap_pyfunction!(affine_local_strain_py, child_module)?
    )?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}

#[pyfunction(name="nonaffine_local_strain")]
fn nonaffine_local_strain_py(
        _py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        ) -> PyResult<f64> {
    let x = x.as_array();
    let y = y.as_array();
    dynamics::nonaffine_local_strain(x, y)
        .map_err(|e| PyArithmeticError::new_err(format!("{}", e)))
}

#[pyfunction(name="affine_local_strain")]
fn affine_local_strain_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        ) -> PyResult<&'py PyArray2<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    match dynamics::affine_local_strain(x, y) {
        Ok(j) => Ok(j.into_pyarray(py)),
        Err(e) => Err(PyArithmeticError::new_err(format!("{}", e)))
    }
}

#[pyfunction(name="self_intermed_scatter_fn")]
fn self_intermed_scatter_fn_py<'py>(
    py: Python<'py>,
    traj: PyReadonlyArray3<f64>,
    ) -> PyResult<&'py PyArray3<f64>> {
        let x = traj.as_array().into_owned();
        Ok(x.into_pyarray(py))
    }

mod dynamics {
    use num::{Float, Zero};
    use ndarray::prelude::*;
    use ndarray_linalg::*;
    use ndarray_linalg::error::LinalgError;
    use ndarray_linalg::least_squares::LeastSquaresSvd;
    use ndarray::Zip;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use std::time;

    #[inline(always)]
    fn p_hop() -> Result<(), ()> {
        Ok(())
    }

    fn _d2min_system_example() {

        // fake system
        let n = 10000usize;
        let a_big= Array::random((n, 10, 2), Uniform::<f64>::new(-2., 2.));

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

    /// Calculates the affine local strain of particles given the local
    /// configurations at two times
    /// 
    /// # Arguments
    /// 
    /// * `initial_vectors` - Initial distance vectors of particles
    /// * `final_vectors` - Final distance vectors of particles
    /// 
    /// # Returns
    /// 
    /// * `J` - Best affine fit matrix
    /// 
    #[inline(always)]
    pub fn affine_local_strain<T: Float + Scalar + Lapack>(
        initial_vectors: ArrayView2<T>,
        final_vectors: ArrayView2<T>)
    -> Result<Array2<T>, LinalgError> {
        let v = initial_vectors.t().dot(&initial_vectors);
        let w = initial_vectors.t().dot(&final_vectors);
        Ok(v.inv()?.dot(&w))
    }

    /// Calculates the nonaffine and affine local strain of particles given the
    /// local configurations at two times
    ///
    /// # Arguments
    /// 
    /// * `initial_vectors` - Initial distance vectors of particles
    /// * `final_vectors` - Final distance vectors of particles
    /// 
    /// # Returns
    /// 
    /// * `D^2_{min}` - Least-square difference to best affine fit
    /// * `J` - Best affine fit matrix
    ///
    #[inline(always)]
    pub fn nonaffine_and_affine_local_strain<T: Float + Scalar + Lapack>(
        initial_vectors: ArrayView2<T>,
        final_vectors: ArrayView2<T>)
    -> Result<(T, Array2<T>), LinalgError> {
        
        let j = affine_local_strain(initial_vectors, final_vectors)?;
        let non_affine = initial_vectors.dot(&j) - initial_vectors;
        let d2min = non_affine
            .iter()
            .fold(Zero::zero(), |sum: T, x| sum + (*x)*(*x));

        Ok((d2min, j))
    }

    /// Calculates the nonaffine local strain of particles given the
    /// local configurations at two times
    ///
    /// # Arguments
    /// 
    /// * `initial_vectors` - Initial distance vectors of particles
    /// * `final_vectors` - Final distance vectors of particles
    /// 
    /// # Returns
    /// 
    /// * `D^2_{min}` - Least-square difference to best affine fit
    ///
    #[inline(always)]
    pub fn nonaffine_local_strain<T: Float + Scalar + Lapack>(
        initial_bonds: ArrayView2<T>,
        final_bonds: ArrayView2<T>)
    -> Result<T, LinalgError> {
        let (d2min, _) = nonaffine_and_affine_local_strain::<T>(initial_bonds, final_bonds)?;
        Ok(d2min)
    }

    /// Calculates the self-intermediate scatter function seen in the literature
    /// of liquid, super-cooled liquids, and glasses
    /// 
    /// # Arguments
    /// 
    /// * `traj` - Particles positions pulled from the simulation. (T,N,D) shaped array where T is the number of time steps, 
    /// N is the number of particles, and D is the number of spatial dimensions
    // / * `time` - Times of the (T) shaped array where T is again the number of time steps recorded
    pub fn self_intermed_scatter_fn<T: Float>(
        pos: ArrayView3<T>,
        q: T
    ) -> Result<Array1<T>, ()> {

        let time = pos.len_of(Axis(0));

        let mut output = Array1::<T>::zeros(time-1);

        let n = pos.len_of(Axis(1));
        let flt_n = T::from(n).unwrap();
        for i in 0..n {
            for t in 1..time {
                let term = q*(&pos.slice(s![t, i, ..]) - &pos.slice(s![0, i, ..]))
                    .fold(Zero::zero(), |sum: T, x| sum + x.powi(2)).sqrt();

                output[t] = output[t] + term.sin()/term;
            }
        }

        output.mapv_inplace( |y| { y/flt_n });
        Ok(output)
    }

    pub fn self_van_hove_corr_fn<T: Float>(
        traj: ArrayView3<T>,
        bins: ArrayView1<T>
    ) -> Result<Array2<T>, ()> {

        Ok(arr2(&[[Zero::zero()]]))
    }

    pub fn msd<T: Float>(
        traj: ArrayView3<T>
    ) -> Result<Array1<T>, ()> {
        
        Ok(arr1(&[Zero::zero()]))
    }
}

mod statics {
    use num::{Float, Zero};
    use ndarray::prelude::*;
    use ndarray::Zip;

    // This is pretty inefficient
    pub fn naive_structure_factor<T: Float>(
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
}

mod softness {
    use ndarray::prelude::*;


    #[inline(always)]
    fn digitize_lin(x: f64, arr: &[f64]) -> usize {

        let l = arr[1] - arr[0];  // arr must have at least two elements

        let ub = arr.len() + 1;
        let lb = 0;

        let mut j = ((x-arr[0])/l) as usize + 1;
        if j < lb { j = lb }
        else if j >= ub { j = ub }
        else if arr[j+1]-x < x-arr[j] { j += 1 }

        j
    }

    #[inline(always)]
    pub fn rad_sf(dr: f64, mu: f64, l: f64) -> f64 {
        let term = (dr-mu)/l;
        (-term*term*0.5).exp()
    }

    #[inline(always)]
    pub fn ang_sf(dr: f64, mu: f64, l: f64) -> f64 {
        let term = (dr-mu)/l;
        (-term*term*0.5).exp()
    }
    
    #[inline(always)]
    pub fn get_sf_rad(
        drs: ArrayView1<f64>,
        type_ids: ArrayView1<u8>,
        types: u8,
        mus: ArrayView1<f64>,
        spread: u8
    ) -> Array1<f64> {
        let l = mus[1] - mus[0];
        let mut feature = Array1::<f64>::zeros((types as usize)*mus.len());

        for i in 0..drs.len() {
            let dr = drs[i];
            
        }

        feature
    }


}

mod utils {

}