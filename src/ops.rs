use glam::{DVec2, DVec3, Vec2, Vec3};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub trait VecOps:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
where
    Self: std::marker::Sized,
{
}

impl VecOps for Vec3 {}

impl VecOps for Vec2 {}

impl VecOps for DVec3 {}

impl VecOps for DVec2 {}
