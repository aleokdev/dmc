// dmc.rs: Dual Marching Cubes implementation for octree structures
// By aleok, 2021
// Based on:
// Schaefer, Scott & Warren, Joe. (2004).
// Dual Marching Cubes: Primal Contouring of Dual Grids.
// Computer Graphics Forum. 24. 10.1111/j.1467-8659.2005.00843.x.
//
// Lewiner, Thomas & Mello, Vinícius & Peixoto, Adelailson & Pesco, Sinesio & Lopes, Hélio. (2010).
// Fast Generation of Pointerless Octree Duals.
// Comput. Graph. Forum. 29. 1661-1669. 10.1111/j.1467-8659.2010.01775.x.
//
// Stocco, Leo & Schrack, Guenther. (1995).
// Integer dilation and contraction for quadtrees and octrees.
// 426 - 428. 10.1109/PACRIM.1995.519560.
//
// Bourke, Paul. (1994).
// Polygonising a scalar field
// http://paulbourke.net/geometry/polygonise/

// The goal here is to adapt the Marching Cubes algorithm to hashed octrees using Morton keys.
// The original Marching Cubes algorithm used an uniform grid to generate the final mesh. However,
// with this approach, we use the dual grid generated from the octree given. The dual grid is composed
// of vertices (One at the center of each leaf node) and edges, which connect vertices of adjacent leaves.
// e.g. Simplified in a quadtree, https://imgur.com/7YJrNLK
// Or, if you prefer an example implementation (with a quadtree; not pretty code):
//   https://editor.p5js.org/alexinfdev/sketches/4I0506NqA

use crate::duals::DualGrid;
use crate::mesh::*;
use crate::octree::*;
use crate::tables;
use crate::util::*;
use crate::McNode;

use cgmath::{Point3, Vector3};
use rayon::prelude::*;

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
