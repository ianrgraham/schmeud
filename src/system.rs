use crate::boxdim::BoxDim;
use ndarray::prelude::*;
use numpy::*;
use pyo3::prelude::*;

pub trait FrameTrait {
    fn pos(&self) -> ArrayView2<f32>;
    fn boxdim(&self) -> Option<&BoxDim>;
}

pub struct Frame {
    pub pos: Array2<f32>,
    pub boxdim: Option<BoxDim>,
}

pub struct FrameView<'a> {
    pub pos: ArrayView2<'a, f32>,
    pub boxdim: Option<&'a BoxDim>,
}

pub struct FrameViewMut<'a> {
    pub pos: ArrayViewMut2<'a, f32>,
    pub boxdim: Option<&'a mut BoxDim>,
}

pub struct PyFrame(Py<PyAny>);

// impl FrameTrait for PyFrame {

pub trait TrajTrait {
    fn frame(&self, i: usize) -> FrameView;
}

#[pyclass]
pub struct Traj {
    pub frames: Vec<Frame>,
}

/// A trajectory
#[pyclass]
pub struct TrackpyTraj {

}

pub struct PyTraj(Py<PyAny>);

// impl TrajTrait for PyTraj {
