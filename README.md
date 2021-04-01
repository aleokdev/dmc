# Sampled DMC
This crate defines a fast implementation for the [Dual Marching Cubes technique](https://www.cs.rice.edu/~jwarren/papers/dmc.pdf),
also known as Linear Hashed Marching Cubes, along with a concurrent octree structure for storing node data.

Unlike [isosurface](https://docs.rs/isosurface/0.0.4/isosurface/), this crate does NOT expect Signed Distance Functions (SDFs) or use them in any way.
The input of the mesh creation functions is the data octree itself, not a sampling source, which makes it appropiate for situations where storing
the sampled points is appropiate (i.e. complex SDFs, destructible environments or objects with different LODs).

## TODO
- Allow for inputting a "LOD function" which determines how detailed each point should be (as in how deep into the octree it should go)
- Add benchmarks
