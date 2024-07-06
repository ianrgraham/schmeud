//! Defines traits to accept generic types for frames and trajectories.
//! Frame objects contain all of the information

#![allow(unused)]

use std::mem::ManuallyDrop;
use std::{borrow::BorrowMut, mem::MaybeUninit};

use crate::boxdim::BoxDim;
use ndarray::prelude::*;
use numpy::*;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;

pub trait FrameTrait {
    fn pos(&self) -> ArrayView2<f32>;
    fn boxdim(&self) -> Option<&BoxDim>;
    // fn as_any(&self) -> &dyn Any {
    //     &self
    // }
    // fn as_any_mut(&mut self) -> &mut dyn Any {
    //     &mut self
    // }
    // fn downcast_ref<T: 'static>(&self) -> Option<&T> {
    //     self.as_any().downcast_ref::<T>()
    // }
    // fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
    //     self.as_any_mut().downcast_mut::<T>()
    // }
}

#[pyclass]
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

#[pyclass]
pub struct PyFrame(Py<PyAny>);

impl FrameTrait for PyFrame {
    fn pos(&self) -> ArrayView2<f32> {
        unimplemented!()
    }

    fn boxdim(&self) -> Option<&BoxDim> {
        unimplemented!()
    }
}

/// Macros
///
/// Identifiers: foo, Bambous, we_can_dance, self
/// Literals: 42, 73u32, 1e-4, "hi!"
/// Keywords: _, fn, self, match, yield, macro
/// Symbols: [, :, ::, ?, ~, @
///
/// Syntax extensions
///
/// Patterns
/// Statements
/// Expressions
/// Items
/// Types
///
/// But NOT
///
/// Identifiers
/// Match arms
/// Struct fields
///
/// Metavariables
///
/// block: block of statements surrounded by curly braces
/// expr: an expression
/// ident: an identifier
/// item: an item, like a funciton, struct, module, impl, etc
/// lifetime: 'foo, 'static
/// literal: string literal
/// meta: a meta item, things that go inside #[...] or #![...]
/// pat: a pattern
/// path: a path (e.g. foo, ::std::mem::replace, transmute::<_, int>, ...)
/// stmt: a statement
/// tt: a single token tree
/// ty: a type
/// vis: a possible empy visability qualifier (e.g. pub, pub(in crate))

macro_rules! extract_frame_ref_and_call {
    ($py: ident, $f: ident, $func: ident) => {
        if let Ok(f) = $f.extract::<PyRef<Frame>>() {
            f.$func()
        } else if let Ok(f) = $f.extract::<PyRefMut<Frame>>() {
            f.$func()
        } else if let Ok(f) = $f.extract::<Py<PyAny>>() {
            let f = *f.as_any().downcast_ref::<&dyn FrameTrait>().unwrap();
            f.$func()
        } else {
            panic!("Could not extract frame")
        }
    };
}

pub trait TrajTrait {
    fn frame(&self, i: usize) -> FrameView;
}

#[pyclass]
pub struct Traj {
    pub frames: Vec<Frame>,
}

/// A trajectory
#[pyclass]
pub struct TrackpyTraj {}

pub struct PyTraj(Py<PyAny>);

impl TrajTrait for PyTraj {
    fn frame(&self, i: usize) -> FrameView {
        unimplemented!()
    }
}

#[pyclass]
#[derive(Clone)]
struct A {
    i: i32,
}

#[pyclass]
#[derive(Clone)]
struct B {
    i: i32,
}

#[pyclass]
#[derive(Clone)]
struct C {
    i: i32,
}

#[derive(FromPyObject)]
enum TTMatch<'a> {
    #[pyo3(transparent)]
    A(PyRef<'a, A>),
    #[pyo3(transparent)]
    B(PyRef<'a, B>),
}

trait TT {
    fn t(&self) -> i32;
}

impl TT for A {
    fn t(&self) -> i32 {
        self.i
    }
}

impl TT for B {
    fn t(&self) -> i32 {
        self.i * 2
    }
}

impl TT for C {
    fn t(&self) -> i32 {
        self.i * 3
    }
}

impl<'a, T: TT + pyo3::PyClass> From<&'a PyRef<'_, T>> for &'a dyn TT {
    fn from(value: &'a PyRef<'_, T>) -> Self {
        value as &'a T
    }
}

// impl<'a, T: TT + pyo3::PyClass> From<PyCell<T>> for &'a dyn TT {
//     fn from(value: PyCell<T>) -> Self {
//         value.as_ref() as &'a T
//     }
// }

// struct DynTT<'a> {
//     inner: &'a dyn TT,
//     marker: std::marker::PhantomData<>
// }

macro_rules! extract_dyn_from_py {
    ($py:ident, $trt:ident; $($typ:ty),*) => {
        {
            let mut tmp: MaybeUninit<std::cell::UnsafeCell<&dyn $trt>> =
                std::mem::MaybeUninit::uninit();
            let owner: Box<dyn std::any::Any> = if false {
                unreachable!()

            }
            $(
                else if let Ok(res) = $py.extract::<PyRef<$typ>>() {
                    let ref_t = &res as &$typ as *const $typ;
                    let o: &dyn $trt = unsafe { &*ref_t as &$typ };
                    std::cell::UnsafeCell::raw_get(tmp.as_ptr()).write(o);
                    Box::new(res)
                }
            )*
            else {
                panic!()
            };
            let tmp = tmp.assume_init();
            let tmp = tmp.into_inner();
            (tmp, owner)
        }
    }
}

macro_rules! extract_tc_ref_and_call {
    ($f: ident, $o: ident, $func: expr) => {
        if let Ok($o) = $f.extract::<PyRef<A>>() {
            $func()
        } else if let Ok($o) = $f.extract::<PyRef<B>>() {
            $func()
        } else if let Ok($o) = $f.extract::<PyRef<C>>() {
            $func()
        }
        // else if let Ok($o) = $f.extract::<&dyn TT>() {
        //     $func()
        // } else {
        //     panic!("No match found")
        // }
    };
}

mod tmp {
    use super::*;
    
    #[pyfunction]
    fn test(a: A) -> A {
        println!("hi");
        return a;
    }
}

// fn extract_tc<T: TT, O, F: FnOnce(T) -> O>(f: &PyAny, func: F) -> O {
//     if let Ok(o) = f.extract::<PyRef<A>>() {
//         func(o)
//     } else if Ok(o) = f.extract::<PyRef<B>>() {
//         func(o)
//     } else {
//         panic!("No types matched")
//     }
// }

#[test]
fn test1() -> PyResult<()> {
    // initialize interpreter
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let class: &PyAny = Py::new(py, A { i: 0 }).unwrap().into_ref(py);

        let class_cell: &PyCell<A> = class.downcast()?;

        class_cell.borrow_mut().i += 1;

        // Alternatively you can get a `PyRefMut` directly
        let class_ref: PyRefMut<'_, A> = class.extract()?;
        assert_eq!(class_ref.i, 1);
        println!("A.c() = {}", class_ref.t());

        let b = B { i: 0 };
        println!("{:?}", &b as *const _);

        let class: &PyAny = Py::new(py, b).unwrap().into_ref(py);

        let class_cell: &PyCell<B> = class.downcast()?;

        println!("Python type of PyCell<B>: {:?}", class_cell.get_type());

        println!("Type name of B: {:?}", std::any::type_name::<B>());

        // println!("Type name {}", PyCell::)

        class_cell.borrow_mut().i += 1;

        // Alternatively you can get a `PyRefMut` directly
        let mut class_ref: PyRefMut<'_, B> = class.extract()?;
        // println!("Python type of PyCell<B>: {:?}", class_ref);
        println!("{:?}", &class_ref as *const _);
        let tmp: &B = &class_ref;
        // class_ref.i += 1;
        assert_eq!(class_ref.i, 1);
        println!("{:?}", tmp as *const B);
        assert_eq!(tmp.i, 1);
        let tmp: &B = &class_ref;
        println!("{:?}", tmp as *const B);

        // drop(tmp);
        class_ref.i += 1;
        println!("B.c() = {}", class_ref.t());
        drop(class_ref);
        let class_ref: PyRef<'_, B> = class.extract()?;
        {
            let ref_class_ref = &class_ref;
            let tmp: &dyn TT = ref_class_ref.into();
            println!(
                "Able to downcast to trait object by first finding concrete type {}",
                tmp.t()
            );
        }

        let tmp = &class_ref as &B;
        println!("{:?}", tmp as *const B);
        let tmp: &dyn TT = tmp;
        println!("{:?}", tmp as *const dyn TT);
        println!("dyn_B.c() = {}", tmp.t());

        Ok(())
    })
}

#[test]
fn test2() -> PyResult<()> {
    // initialize interpreter
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let classes: Vec<&PyAny> = vec![
            Py::new(py, A { i: 1 }).unwrap().into_ref(py),
            Py::new(py, B { i: 2 }).unwrap().into_ref(py),
        ];

        for class in classes {
            if let Ok(class_cell) = class.extract::<PyRef<A>>() {
                println!("A.c() = {}", class_cell.t());
            } else if let Ok(class_cell) = class.extract::<PyRef<B>>() {
                println!("B.c() = {}", class_cell.t());
            }
        }
        Ok(())
    })
}

#[test]
fn test3() -> PyResult<()> {
    // initialize interpreter
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let classes: Vec<&PyAny> = vec![
            Py::new(py, A { i: 1 }).unwrap().into_ref(py),
            Py::new(py, B { i: 2 }).unwrap().into_ref(py),
            Py::new(py, C { i: 2 }).unwrap().into_ref(py),
        ];

        for class in &classes {
            extract_tc_ref_and_call!(class, downcast_class, || println!(
                "X.c() = {}",
                downcast_class.t()
            ))
        }

        // for class in &classes {
        //     let tmp = extract_dyn_from_py!(class, TT; A, B, C);
        //     tmp.t();
        // }

        Ok(())
    })
}

#[test]
fn test4() -> PyResult<()> {
    // initialize interpreter
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let classes: Vec<&PyAny> = vec![
            Py::new(py, A { i: 1 }).unwrap().into_ref(py),
            Py::new(py, B { i: 2 }).unwrap().into_ref(py),
            Py::new(py, C { i: 2 }).unwrap().into_ref(py),
        ];

        // transfrom refs into 'static
        let classes: Vec<&'static PyAny> = classes
            .into_iter()
            .map(|x| unsafe { std::mem::transmute(x) })
            .collect();

        for class in &classes {
            extract_tc_ref_and_call!(class, downcast_class, || println!(
                "X.c() = {}",
                downcast_class.t()
            ))
        }

        for &class in &classes {
            let mut tmp: MaybeUninit<std::cell::UnsafeCell<&dyn TT>> =
                std::mem::MaybeUninit::uninit();
            dbg!(tmp.as_ptr());
            if let Ok(res) = class.extract::<PyRef<A>>() {
                let ref_t = &res as &A as *const A; // erase original lifetime
                let o: &dyn TT = unsafe { &*ref_t as &A }; // case into trait obj
                                                           // SAFETY: the reference to the inner type of PyRef<A> (A),
                                                           // lasts as long as the reference to inner type in &PyAny
                unsafe {
                    std::cell::UnsafeCell::raw_get(tmp.as_ptr()).write(o);
                }
            } else if let Ok(res) = class.extract::<PyRef<B>>() {
                let ref_t = &res as &B as *const B;
                let o: &dyn TT = unsafe { &*ref_t as &B };
                unsafe {
                    std::cell::UnsafeCell::raw_get(tmp.as_ptr()).write(o);
                }
            } else if let Ok(res) = class.extract::<PyRef<C>>() {
                let ref_t = &res as &C as *const C;
                let o: &dyn TT = unsafe { &*ref_t as &C };
                unsafe {
                    std::cell::UnsafeCell::raw_get(tmp.as_ptr()).write(o);
                }
            } else {
                panic!()
            };
            let tmp = unsafe { tmp.assume_init() };
            let tmp = tmp.into_inner();
            println!("X.c() = {}", tmp.t());
            // println!("hey!")
        }

        for &class in &classes {
            let (tmp, owner) = unsafe { extract_dyn_from_py!(class, TT; A, B, C) };
            // drop(*class);
            println!("X.c() = {}", tmp.t());
            match class.extract::<PyRefMut<'_, C>>() {
                Ok(mut cmut) => {
                    println!("hey!");
                    cmut.i += 1;
                },
                Err(e) => {
                    e.print(py)
                }
            }
            println!("X.c() = {}", tmp.t());
            tmp.t();
        }

        Ok(())
    })
}

// #[test]
// fn takes_a_frame(f: &PyAny) {

//     let f = *f.as_any().downcast_ref::<&dyn FrameTrait>().unwrap();

//     let pos = f.pos();
//     let boxdim = f.boxdim();
//     println!("{:?} {:?}", pos, boxdim);
// }
