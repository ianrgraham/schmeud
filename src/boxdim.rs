use glam::{Vec2, Vec3};
use ndarray::prelude::*;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass]
#[derive(PartialEq, Debug)]
pub struct BoxDim {
    lo: Vec3,
    hi: Vec3,
    l: Vec3,
    l_inv: Vec3,
    tilt: Vec3,  // xy, xz, yz
    periodic: [bool; 3],
    is_2d: bool
}

impl Default for BoxDim {
    fn default() -> Self {
        Self {
            lo: Vec3::ZERO,
            hi: Vec3::ZERO,
            l: Vec3::ZERO,
            l_inv: Vec3::ZERO,
            tilt: Vec3::ZERO,
            periodic: [true; 3],
            is_2d: false
        }
    }
}

#[pymethods]
impl BoxDim {

    #[classmethod]
    pub fn from_freud<'p>(cls: &'p PyType, py: Python<'p>, freud_box: Py<PyAny>) -> PyResult<Self> {
        let l = freud_box.getattr(py, "L")?;
        let l = l.extract::<PyReadonlyArray1<f32>>(py)?;
        let l: [f32; 3] = l.as_slice()?.try_into()?;

        let periodic = freud_box.getattr(py, "periodic")?;
        let periodic = periodic.extract::<PyReadonlyArray1<bool>>(py)?;
        let periodic: [bool; 3] = periodic.as_slice()?.try_into()?;

        let xy = freud_box.getattr(py, "xy")?;
        let xy = xy.extract::<f32>(py)?;
        let xz = freud_box.getattr(py, "xz")?;
        let xz = xz.extract::<f32>(py)?;
        let yz = freud_box.getattr(py, "yz")?;
        let yz = yz.extract::<f32>(py)?;
        let sbox = [l[0], l[1], l[2], xy, xz, yz];

        Ok(Self::from_array(&sbox, periodic))
    }
}


impl BoxDim {

    pub fn from_array(sbox: &[f32; 6], periodic: [bool; 3]) -> Self {
        let is_2d = sbox[2] == 0.0;
        let arrays = sbox.split_at(3);
        let l = Vec3::from_slice(arrays.0);
        let tilt = Vec3::from_slice(arrays.1);
        let l_inv = 1.0/l;
        let hi = l/2.0;
        let lo = -hi;
        Self {
            lo,
            hi,
            l,
            l_inv,
            tilt,
            periodic,
            is_2d
        }
    }

    #[inline(always)]
    pub fn min_image_ndarray(&self, w: &mut ArrayViewMut1<f32>) {

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

    fn volume(&self) -> f32 {
        if self.is_2d {
            self.l.x * self.l.y
        } else {
            self.l.x * self.l.y * self.l.z
        }
    }
}