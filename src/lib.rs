// Made by Alejandro Perea (aleok), 2021
// Based on:
//
// - Schaefer, Scott & Warren, Joe. (2004).
// Dual Marching Cubes: Primal Contouring of Dual Grids.
// Computer Graphics Forum. 24. 10.1111/j.1467-8659.2005.00843.x.
//
// - Lewiner, Thomas & Mello, Vinícius & Peixoto, Adelailson & Pesco, Sinesio & Lopes, Hélio. (2010).
// Fast Generation of Pointerless Octree Duals.
// Comput. Graph. Forum. 29. 1661-1669. 10.1111/j.1467-8659.2010.01775.x.
//
// - Stocco, Leo & Schrack, Guenther. (1995).
// Integer dilation and contraction for quadtrees and octrees.
// 426 - 428. 10.1109/PACRIM.1995.519560.
//
// Bourke, Paul. (1994).
// Polygonising a scalar field
// http://paulbourke.net/geometry/polygonise/

// Licensed under the MIT License. You may find a copy of this license in the root directory of the
// crate.

//! Dual Marching Cubes implementation for octree structures

#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

pub mod dmc;

pub mod duals;
pub mod octree;
pub mod prelude;
mod tables;
mod util;
