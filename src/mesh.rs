#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub position: cgmath::Point3<f32>,
    pub normal: cgmath::Vector3<f32>,
}
