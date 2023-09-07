use glam::{BVec3, IVec2, IVec3, Vec2, Vec3};
use ndarray::prelude::*;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::ops::VecOps;

/// Periodic triclinic box.
#[pyclass]
#[derive(PartialEq, Debug, Clone)]
pub struct BoxDim {
    lo: Vec3,
    hi: Vec3,
    l: Vec3,
    l_inv: Vec3,
    tilt: Vec3, // xy, xz, yz
    periodic: BVec3,
    is_2d: bool,
}

impl Default for BoxDim {
    fn default() -> Self {
        Self {
            lo: Vec3::ZERO,
            hi: Vec3::ZERO,
            l: Vec3::ZERO,
            l_inv: Vec3::INFINITY,
            tilt: Vec3::ZERO,
            periodic: BVec3::TRUE,
            is_2d: false,
        }
    }
}

#[pymethods]
impl BoxDim {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }

    #[classmethod]
    #[pyo3(name = "from_freud")]
    pub fn py_from_freud<'p>(
        _cls: &'p PyType,
        py: Python<'p>,
        freud_box: Py<PyAny>,
    ) -> PyResult<Self> {
        // TODO Not the most effecient.
        // Need to figure out how to access the cython fields
        // from pyo3.
        let l = freud_box.getattr(py, "L")?;
        let l = l.extract::<PyReadonlyArray1<f64>>(py)?;
        let l: [f64; 3] = l.as_slice()?.try_into()?;
        let periodic = freud_box.getattr(py, "periodic")?;
        let periodic = periodic.extract::<PyReadonlyArray1<bool>>(py)?;
        let periodic: [bool; 3] = periodic.as_slice()?.try_into()?;

        let xy = freud_box.getattr(py, "xy")?;
        let xy = xy.extract::<f32>(py)?;
        let xz = freud_box.getattr(py, "xz")?;
        let xz = xz.extract::<f32>(py)?;
        let yz = freud_box.getattr(py, "yz")?;
        let yz = yz.extract::<f32>(py)?;
        let sbox = [l[0] as f32, l[1] as f32, l[2] as f32, xy, xz, yz];

        Ok(Self::from_arrays(&sbox, periodic))
    }

    #[classmethod]
    #[pyo3(name = "cube")]
    fn py_cube(_cls: &PyType, l: f32) -> Self {
        Self::cube(l)
    }

    #[classmethod]
    #[pyo3(name = "from_array")]
    fn py_from_array(
        _cls: &PyType,
        sbox: PyReadonlyArray1<f32>,
        periodic: PyReadonlyArray1<bool>,
    ) -> PyResult<Self> {
        let sbox = sbox.as_slice()?.try_into()?;
        let periodic = periodic.as_slice()?.try_into()?;
        Ok(Self::from_arrays(&sbox, periodic))
    }

    #[getter(l)]
    fn py_l(&self) -> [f32; 3] {
        self.l.into()
    }

    #[getter(tilt)]
    fn py_tilt(&self) -> [f32; 3] {
        self.tilt.into()
    }

    #[pyo3(name = "periodic")]
    fn py_periodic(&self) -> [bool; 3] {
        self.periodic.into()
    }

    #[pyo3(name = "is_2d")]
    fn py_is_2d(&self) -> bool {
        self.is_2d
    }

    #[pyo3(name = "volume")]
    fn py_volume(&self) -> f32 {
        self.volume()
    }

    #[pyo3(name = "fractional")]
    fn py_fractional(&self, v: [f32; 3]) -> [f32; 3] {
        let v = Vec3::from(v);
        self.fractional(&v).into()
    }

    #[pyo3(name = "absolute")]
    fn py_absolute(&self, v: [f32; 3]) -> [f32; 3] {
        let v = Vec3::from(v);
        self.absolute(&v).into()
    }

    #[pyo3(name = "wrap")]
    fn py_wrap(&self, v: [f32; 3]) -> [f32; 3] {
        let v = Vec3::from(v);
        self.wrap(&v).into()
    }

    #[pyo3(name = "image")]
    fn py_image(&self, v: [f32; 3]) -> [i32; 3] {
        let v = Vec3::from(v);
        self.image(&v).into()
    }
}

pub trait BoxVec: VecOps + Copy + Clone {
    type Image;
    type Inner: num_traits::Float;

    fn fractional(&self, boxdim: &BoxDim) -> Self;

    fn absolute(&self, boxdim: &BoxDim) -> Self;

    fn wrap(&self, boxdim: &BoxDim) -> Self;

    fn image(&self, boxdim: &BoxDim) -> Self::Image;

    fn length(&self) -> Self::Inner;
}

impl BoxVec for Vec3 {
    type Image = IVec3;
    type Inner = f32;

    #[inline(always)]
    fn fractional(&self, boxdim: &BoxDim) -> Self {
        let tilt = boxdim.tilt;
        let mut w = *self - boxdim.lo;
        w.x -= (tilt.y - tilt.z * tilt.x) * self.z + tilt.x * self.y;
        w.y -= tilt.z * self.z;
        w /= boxdim.l;

        if boxdim.is_2d {
            w.z = 0.0;
        }
        w
    }

    #[inline(always)]
    fn absolute(&self, boxdim: &BoxDim) -> Self {
        let tilt = boxdim.tilt;
        let mut w = boxdim.lo + *self * boxdim.l;
        w.x += tilt.x * w.y + tilt.y * w.z;
        w.y += tilt.z * w.z;

        if boxdim.is_2d {
            w.z = 0.0
        }
        w
    }

    #[inline(always)]
    fn wrap(&self, boxdim: &BoxDim) -> Self {
        // TODO check that this implementation has the same behaviour as freud
        // in test cases
        if (!boxdim.periodic).all() {
            return *self;
        }

        let mut w = *self;
        if !boxdim.is_2d && boxdim.periodic.z {
            let img = (w.z * boxdim.l_inv.z).round();
            w.z -= boxdim.l.z * img;
            w.y -= boxdim.l.z * boxdim.tilt.y * img;
            w.x -= boxdim.l.z * boxdim.tilt.x * img;
        }

        if boxdim.periodic.y {
            let img = (w.y * boxdim.l_inv.y).round();
            w.y -= boxdim.l.y * img;
            w.x -= boxdim.l.y * boxdim.tilt.x * img;
        }

        if boxdim.periodic.x {
            let img = (w.x * boxdim.l_inv.x).round();
            w.x -= boxdim.l.x * img;
        }

        w
    }

    #[inline(always)]
    fn image(&self, boxdim: &BoxDim) -> Self::Image {
        // TODO check that this implementation has the same behaviour as freud
        // in test cases
        let mut img = IVec3::ZERO;
        if (!boxdim.periodic).all() {
            return img;
        }

        if !boxdim.is_2d && boxdim.periodic.z {
            img.z = (self.z * boxdim.l_inv.z).round() as i32;
        }

        if boxdim.periodic.y {
            img.z = (self.y * boxdim.l_inv.y).round() as i32;
        }

        if boxdim.periodic.x {
            img.z = (self.x * boxdim.l_inv.x).round() as i32;
        }
        img
    }

    #[inline(always)]
    fn length(&self) -> Self::Inner {
        Vec3::length(*self)
    }
}

// TODO need to benchmark whether a specialized impl is negessary performance wise
impl BoxVec for Vec2 {
    type Image = IVec2;
    type Inner = f32;

    #[inline(always)]
    fn fractional(&self, boxdim: &BoxDim) -> Self {
        let w = self.extend(0.0);
        w.fractional(boxdim).truncate()
    }

    #[inline(always)]
    fn absolute(&self, boxdim: &BoxDim) -> Self {
        let w = self.extend(0.0);
        w.absolute(boxdim).truncate()
    }

    #[inline(always)]
    fn wrap(&self, boxdim: &BoxDim) -> Self {
        let w = self.extend(0.0);
        w.wrap(boxdim).truncate()
    }

    #[inline(always)]
    fn image(&self, boxdim: &BoxDim) -> Self::Image {
        let w = self.extend(0.0);
        w.image(boxdim).truncate()
    }

    #[inline(always)]
    fn length(&self) -> Self::Inner {
        Vec2::length(*self)
    }
}

impl BoxDim {
    #[inline(always)]
    pub fn fractional<T: BoxVec>(&self, v: &T) -> T {
        v.fractional(self)
    }

    #[inline(always)]
    pub fn absolute<T: BoxVec>(&self, v: &T) -> T {
        v.absolute(self)
    }

    #[inline(always)]
    pub fn wrap<T: BoxVec>(&self, v: &T) -> T {
        v.wrap(self)
    }

    #[inline(always)]
    pub fn image<T: BoxVec>(&self, v: &T) -> T::Image {
        v.image(self)
    }

    #[inline(always)]
    pub fn distance<T: BoxVec>(&self, v1: &T, v2: &T) -> T::Inner {
        let d = self.wrap(&(*v1 - *v2));
        d.length()
    }
}

impl BoxDim {
    pub fn cube(l: f32) -> Self {
        let hi = Vec3::splat(l / 2.0);
        Self {
            lo: -hi,
            hi: hi,
            l: Vec3::splat(l),
            l_inv: Vec3::splat(1.0 / l),
            tilt: Vec3::ZERO,
            periodic: BVec3::TRUE,
            is_2d: false,
        }
    }

    pub fn from_arrays(sbox: &[f32; 6], periodic: [bool; 3]) -> Self {
        let is_2d = sbox[2] == 0.0;
        let periodic = BVec3::new(periodic[0], periodic[1], periodic[2]);
        let arrays = sbox.split_at(3);
        let l = Vec3::from_slice(arrays.0);
        let tilt = Vec3::from_slice(arrays.1);
        let l_inv = 1.0 / l;
        let hi = l / 2.0;
        let lo = -hi;
        Self {
            lo,
            hi,
            l,
            l_inv,
            tilt,
            periodic,
            is_2d,
        }
    }

    pub fn set_l(&mut self, l: Vec3) {
        self.l = l;
        self.l_inv = 1.0 / l;
        self.hi = l / 2.0;
        self.lo = -self.hi;
    }

    pub fn l(&self) -> Vec3 {
        self.l
    }

    pub fn tilt(&self) -> Vec3 {
        self.tilt
    }

    #[inline(always)]
    pub fn min_image_ndarray(&self, w: &mut ArrayViewMut1<f32>) {
        // if self.periodic.iter().all(|x| !x) {
        //     return;
        // }

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
    pub fn min_image_array3(&self, w: &mut [f32; 3]) {
        // if self.periodic.iter().all(|x| !x) {
        //     return;
        // }

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
    pub fn min_image_array2(&self, w: &mut [f32; 2]) {
        assert!(self.is_2d);

        let img = (w[1] * self.l_inv.y).round();
        w[1] -= self.l.y * img;
        w[0] -= self.l.y * self.tilt.x * img;

        let img = (w[0] * self.l_inv.x).round();
        w[0] -= self.l.x * img;
    }

    #[inline(always)]
    pub fn min_image_vec3(&self, w: &mut Vec3) {
        if !self.is_2d {
            let img = (w.z * self.l_inv.z).round();
            w.z -= self.l.z * img;
            w.y -= self.l.z * self.tilt.y * img;
            w.x -= self.l.z * self.tilt.x * img;
        }

        let img = (w.y * self.l_inv.y).round();
        w.y -= self.l.y * img;
        w.x -= self.l.y * self.tilt.x * img;

        let img = (w.x * self.l_inv.x).round();
        w[0] -= self.l.x * img;
    }

    #[inline(always)]
    pub fn min_image_vec2(&self, w: &mut Vec2) {
        assert!(self.is_2d);

        let img = (w.y * self.l_inv.y).round();
        w.y -= self.l.y * img;
        w.x -= self.l.y * self.tilt.x * img;

        let img = (w.x * self.l_inv.x).round();
        w[0] -= self.l.x * img;
    }

    pub fn volume(&self) -> f32 {
        if self.is_2d {
            self.l.x * self.l.y
        } else {
            self.l.x * self.l.y * self.l.z
        }
    }

    pub fn is_2d(&self) -> bool {
        self.is_2d
    }

    pub fn periodic(&self) -> BVec3 {
        self.periodic
    }

    pub fn nearest_plane_difference(&self) -> Vec3 {
        let mut dist = Vec3::ZERO;
        let l = self.l;
        let tilt = self.tilt;
        let term = tilt.x * tilt.z - tilt.y;

        dist.x = l.x / (1.0 + tilt.x * tilt.x + term * term);
        dist.y = l.y / (1.0 + tilt.z * tilt.z);
        dist.z = l.z;

        dist
    }

    pub fn lattice_vector(&self, i: usize) -> Vec3 {
        if i == 0 {
            Vec3::new(self.l.x, 0.0, 0.0)
        } else if i == 1 {
            Vec3::new(self.l.y * self.tilt.x, self.l.y, 0.0)
        } else if i == 2 && !self.is_2d {
            Vec3::new(self.l.z * self.tilt.y, self.l.z * self.tilt.z, self.l.z)
        } else {
            panic!(
                "index out of bounds: the box has {} lattice vectors but the index is {}",
                if self.is_2d { 2 } else { 3 },
                i
            );
        }
    }
}
