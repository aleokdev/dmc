//! Contains the Dual Marching Cubes implementation, along with a function
//! ([`dmc::dmc::mesh_from_octree`]) to interface with it.
//!
//! # Explanation
//! The goal here is to adapt the Marching Cubes algorithm to hashed octrees using Morton keys.
//! The original Marching Cubes algorithm used an uniform grid to generate the final mesh. However,
//! with this approach, we use the dual grid generated from the octree given. The dual grid is
//! composed of vertices (One at the center of each leaf node) and edges, which connect vertices of
//! adjacent leaves.
//!
//! # Visual Example
//! Simplified in a quadtree, https://imgur.com/7YJrNLK
//!
//! # Possible Implementation
//! Simplified in a quadtree; not pretty code: https://editor.p5js.org/alexinfdev/sketches/4I0506NqA
//!
//! # References
//! Refer to the comments at the start of [`src/lib.rs`].

use crate::duals::DualGrid;
use crate::octree::*;
use crate::tables;
use crate::util::*;

use cgmath::{Point3, Vector3};
use rayon::prelude::*;

/// Creates a mesh from an octree full of values sampled from a Signed Distance Function (SDF).
/// The triangles of the resulting mesh will be in **counter-clockwise** order.
/// # Panics
/// Should never panic. If it does, it's an error in the crate; please report it.
pub fn mesh_from_octree<T: McNode>(octree: &HashedOctree<T>, scale: f32) -> Mesh {
    let dual = DualGrid::from_octree(octree);

    const VOLUME_TO_MC_VERTEX: [usize; 8] = [3, 2, 7, 6, 0, 1, 4, 5];
    const MC_CUBE_EDGES: [(usize, usize); 12] = [
        (4, 5),
        (1, 5),
        (0, 1),
        (0, 4),
        (6, 7),
        (3, 7),
        (2, 3),
        (2, 6),
        (4, 6),
        (5, 7),
        (1, 3),
        (0, 2),
    ];

    let vertices = dual
        .volumes
        .into_par_iter()
        .map(|volume| {
            let mut densities: [f32; 8] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            let mut normals: [Vector3<f32>; 8] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };

            let mut cube_index = 0;
            volume.iter().enumerate().for_each(|(volume_i, &key)| {
                let mc_i = VOLUME_TO_MC_VERTEX[volume_i];
                if key == MortonKey::none() {
                    return;
                }
                #[cfg(test)]
                {
                    dbg!(mc_i, key, octree.value(key).unwrap().density());
                    assert!(!octree.is_subdivided(key));
                };

                match octree.value(key) {
                    Some(node) => {
                        let density = node.density();
                        if density > 0. {
                            cube_index |= 1 << mc_i;
                        }
                        densities[volume_i] = density;
                        normals[volume_i] = node.normal();
                    }
                    _ => (),
                }
            });

            (volume, densities, cube_index, normals)
        })
        .filter(|(_, _, cube_index, _)| cube_index != &0 && cube_index != &0xFF)
        .flat_map_iter(|(volume, densities, cube_index, normals)| {
            let (volume_vertices, volume_normals) = {
                let factor = |i1, i2| {
                    let p1: f32 = densities[i1];
                    let p2: f32 = densities[i2];
                    if p1.abs() < 0.00001 {
                        p1
                    } else if p2.abs() < 0.00001 {
                        p2
                    } else if (p1 - p2).abs() < 0.00001 {
                        p1
                    } else {
                        (0. - p1) / (p2 - p1)
                    }
                };

                let mut verts: [[f32; 3]; 12] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                let mut norms: [[f32; 3]; 12] =
                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                let edges = tables::EDGE_TABLE[cube_index];
                (0..12)
                    .into_iter()
                    .filter(|i| (edges & (1 << i)) > 0)
                    .for_each(|i| {
                        let (v1_i, v2_i) = MC_CUBE_EDGES[i];
                        let factor = factor(v1_i, v2_i);
                        verts[i] = volume[v1_i]
                            .position()
                            .interpolate(volume[v2_i].position(), factor)
                            .into();
                        norms[i] = normals[v1_i].interpolate(normals[v2_i], factor).into();
                    });
                (verts, norms)
            };

            tables::TRI_TABLE[cube_index]
                .chunks(3)
                .take_while(|chunk| chunk[0] != -1)
                .flat_map(move |chunk| {
                    let (vtx1, vtx2, vtx3) = (
                        Point3::from(volume_vertices[chunk[0] as usize]) * scale,
                        Point3::from(volume_vertices[chunk[1] as usize]) * scale,
                        Point3::from(volume_vertices[chunk[2] as usize]) * scale,
                    );
                    let (n1, n2, n3) = (
                        volume_normals[chunk[0] as usize],
                        volume_normals[chunk[1] as usize],
                        volume_normals[chunk[2] as usize],
                    );

                    std::array::IntoIter::new([(vtx1, n1), (vtx2, n2), (vtx3, n3)]).map(
                        move |(vtx, norm)| Vertex {
                            position: vtx.into(),
                            normal: norm.into(),
                        },
                    )
                })
        })
        .collect::<Vec<_>>();

    Mesh {
        indices: (0..vertices.len() as u32).collect(),
        vertices,
    }
}

/// A simple mesh type that holds a vector of vertices and another one of indices.
/// This type is meant to be converted to your own mesh type via the [`std::convert::From`] trait.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    /// The vertices of the mesh.
    pub vertices: Vec<Vertex>,
    /// The vertex indices of the mesh.
    pub indices: Vec<u32>,
}

/// A 3D vertex that holds a position and a normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    /// The position of the vertex.
    pub position: cgmath::Point3<f32>,
    /// The normalized normal of the vertex.
    pub normal: cgmath::Vector3<f32>,
}

/// Provides the functions that a node must define to create a mesh out of a group of them.
pub trait McNode: Copy + Send + Sync {
    /// This function should return the distance from this node's center to its contour. This can be
    /// precalculated using a Signed Distance Function (SDF).
    ///
    /// # Resources
    /// [Here](https://iquilezles.org/www/articles/distfunctions/distfunctions.htm) is a link with
    /// basic SDFs and operations over them.
    fn density(&self) -> f32;

    /// This function should return the gradient of the SDF sampled, that is, the normal of the node
    /// if it were on the surface of the SDF.
    ///
    /// # How To Calculate The Gradient
    /// The gradient of a SDF is its derivative. However, if you don't want to calculate it (Which
    /// you won't, because it will get complex really quickly), you can approximate it with the
    /// following formula:
    /// ```ignore
    /// pub fn sdf_gradient(pos: Point3<f32>) -> Vector3<f32> {
    ///     vec3(
    ///        sdf(point3(pos.x + f32::EPSILON, pos.y, pos.z))
    ///            - sdf(point3(pos.x - f32::EPSILON, pos.y, pos.z)),
    ///        sdf(point3(pos.x, pos.y + f32::EPSILON, pos.z))
    ///             - sdf(point3(pos.x, pos.y - f32::EPSILON, pos.z)),
    ///        sdf(point3(pos.x, pos.y, pos.z + f32::EPSILON))
    ///             - sdf(point3(pos.x, pos.y, pos.z - f32::EPSILON)),
    ///     )
    ///     .normalize()
    /// }
    /// ```
    fn normal(&self) -> Vector3<f32>;
}
