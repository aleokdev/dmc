use cgmath::Vector3;

pub trait McNode: Copy + Send + Sync {
    fn density(&self) -> f32;
    fn normal(&self) -> Vector3<f32>;
}
