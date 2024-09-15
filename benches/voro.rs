
use glam::Vec3;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use schmeud::boxdim;
#[cfg(any(feature = "voro-static", feature = "voro-system"))]
use schmeud::nlist::voro::Voronoi;

fn main() {
    let boxdim = boxdim::BoxDim::cube(10.0);

    // random ndarray of shape (n_points, 3)
    let points = ndarray::Array2::<f32>::random((100_000, 3), Uniform::new(-5.0, 5.0));
    let shape = points.shape();
    let points = points.as_slice().unwrap();
    // SAFETY: reinterpret numpy array slice into slice Vec3
    let ptr = points.as_ptr() as *const Vec3;
    let points = unsafe { std::slice::from_raw_parts(ptr, shape[0]) };
    
    #[cfg(any(feature = "voro-static", feature = "voro-system"))] {
        let _voro = Voronoi::new(boxdim.clone(), points);
    }
}
