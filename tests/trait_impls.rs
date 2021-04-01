use cgmath::Vector3;
use dmc::McNode;

#[derive(Clone, Copy)]
pub struct Node {
    pub density: f32,
    pub normal: Vector3<f32>,
}

impl McNode for Node {
    fn density(&self) -> f32 {
        self.density
    }

    fn normal(&self) -> Vector3<f32> {
        self.normal
    }
}
