use cgmath::{Point3, Vector3};

pub trait Interpolatable {
    fn interpolate(self, other: Self, factor: f32) -> Self;
}

impl Interpolatable for Point3<f32> {
    fn interpolate(self, other: Self, factor: f32) -> Self {
        let diff = other - self;
        self + diff * factor
    }
}

impl Interpolatable for Vector3<f32> {
    fn interpolate(self, other: Self, factor: f32) -> Self {
        let diff = other - self;
        self + diff * factor
    }
}
