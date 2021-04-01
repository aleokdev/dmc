//! Defines the dual grid generation for a given octree with the [`DualGrid::from_octree`] function.

use crate::octree::*;
use rayon::prelude::*;

/// Type that defines a grid made from the dual edges of an octree.
pub struct DualGrid {
    /// The cells or volumes that make up this dual grid.
    /// Each one is composed by, at most, 8 different dual vertices.
    /// They are always presented in the same order: The same one as defined in
    /// [`dmc::octree::OctreeIndex`].
    ///
    /// # Notes
    /// Some cells will have shared vertices, however, their topology will be the same one as a
    /// cube, so the Marching Cubes algorithm will work over them.
    pub volumes: Vec<[MortonKey; 8]>,
}

impl DualGrid {
    /// Constructs a dual grid from a given octree.
    /// You won't require to create an object of this type yourself unless you plan on processing
    /// the duals. If you want to generate a mesh from an octree, use
    /// [`dmc::dmc::mesh_from_octree`].
    /// # Example
    /// ```rust
    /// use dmc::octree::*;
    /// use dmc::duals::*;
    ///
    /// let mut octree = HashedOctree::new(1);
    /// octree.subdivide(MortonKey::root()).for_each(drop);
    ///
    /// assert!(octree.is_subdivided(MortonKey::root()));
    /// assert_eq!(
    ///     octree.leaves(MortonKey::root()),
    ///     vec![
    ///         MortonKey(0b1111),
    ///         MortonKey(0b1110),
    ///         MortonKey(0b1101),
    ///         MortonKey(0b1100),
    ///         MortonKey(0b1011),
    ///         MortonKey(0b1010),
    ///         MortonKey(0b1001),
    ///         MortonKey(0b1000),
    ///     ]
    /// );
    ///
    /// let duals = DualGrid::from_octree(&octree);
    ///
    /// assert_eq!(
    ///     duals.volumes,
    ///     vec![[
    ///         MortonKey(0b1111),
    ///         MortonKey(0b1110),
    ///         MortonKey(0b1101),
    ///         MortonKey(0b1100),
    ///         MortonKey(0b1011),
    ///         MortonKey(0b1010),
    ///         MortonKey(0b1001),
    ///         MortonKey(0b1000),
    ///     ]]
    /// );
    /// ```
    pub fn from_octree<T: Copy + Send + Sync>(octree: &HashedOctree<T>) -> Self {
        let leaves = octree.leaves(MortonKey::root());

        let volumes = leaves
            .into_par_iter()
            .map(|leaf| {
                // Get the vertex codes of the leaf.
                leaf2vert(leaf)
            })
            .flat_map_iter(|(vertex_lv, vertex_keys)| {
                // To obtain all the vertices and cells contained in the dual grid, we need to execute and filter
                // everything in a precise order in order to only process each vertex exactly once.
                std::array::IntoIter::new(vertex_keys)
                    .enumerate()
                    .take_while(|(_vertex_i, vertex_k)| vertex_k != &MortonKey::none())
                    .flat_map(move |(vertex_i, vertex_k)| {
                        // Get all the neighbours to this vertex.
                        let neighbours = vert2leaf(vertex_k, vertex_lv);

                        // Filter the neighbours and only accept these with the same level as this leaf.
                        std::iter::once(())
                            .take_while(move |()| {
                                for neighbour_i in 0..8 {
                                    // Skip processing the node we came from!
                                    if neighbour_i == vertex_i {
                                        continue;
                                    }

                                    let neighbour_k = neighbours[neighbour_i];

                                    // Skip the neighbour if the leaf `leaf` is deeper.
                                    if !octree.node_exists(neighbour_k) {
                                        continue;
                                    }

                                    // Skip the whole vertex if the neighbour is deeper than the leaf,
                                    // since it will be processed by that neighbour.
                                    if octree.is_subdivided(neighbour_k) {
                                        return false;
                                    }

                                    // Neighbour has the same level as this leaf: It's a tie. Process the vertex
                                    // only if neighbour_i < vertex_i.
                                    if neighbour_i < vertex_i {
                                        return false;
                                    }
                                }
                                true
                            })
                            .map(move |()| {
                                let mut keys: [MortonKey; 8] =
                                    unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                                (0..8).for_each(|neighbour_i| {
                                    let mut neighbour_k = neighbours[neighbour_i];
                                    while neighbour_k != MortonKey::none()
                                        && !octree.node_exists(neighbour_k)
                                    {
                                        neighbour_k = neighbour_k.parent();
                                    }
                                    keys[neighbour_i] = neighbour_k;
                                });
                                keys
                            })
                    })
            })
            .collect();

        Self { volumes }
    }
}

// Dilation constants
const DIL_X: u64 = 0b001001001001001001001001001001001001001001001001001001001001001u64;
const DIL_Y: u64 = 0b010010010010010010010010010010010010010010010010010010010010010u64;
const DIL_Z: u64 = 0b100100100100100100100100100100100100100100100100100100100100100u64;

/// Returns morton-like keys for the 8 vertices belonging to a leaf node, along with their level.
fn leaf2vert(key: MortonKey) -> (u32, [MortonKey; 8]) {
    let level = key.level();
    let level_key: u64 = 1 << (3 * level);

    let mut keys: [MortonKey; 8] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    (0..8u64).into_iter().for_each(|vertex_i| {
        // Perform the dilated integer addition of k + i.
        let vertex_key = (((key.0 | !DIL_X) + (vertex_i & DIL_X)) & DIL_X)
            | (((key.0 | !DIL_Y) + (vertex_i & DIL_Y)) & DIL_Y)
            | (((key.0 | !DIL_Z) + (vertex_i & DIL_Z)) & DIL_Z);

        // Check if the vertex is within the volume of the octree and not on its surface
        // (Check for overflows)
        keys[vertex_i as usize] = if (vertex_key >= (level_key << 1))
            || ((vertex_key - level_key) & DIL_X) == 0
            || ((vertex_key - level_key) & DIL_Y) == 0
            || ((vertex_key - level_key) & DIL_Z) == 0
        {
            MortonKey::none()
        } else {
            MortonKey(vertex_key)
        };
    });

    (level, keys)
}

/// Returns the 8 Morton keys for the adjacent nodes to a vertex, given its Morton-like key and level.
fn vert2leaf(vertex_k: MortonKey, vertex_lv: u32) -> [MortonKey; 8] {
    let dc = vertex_k.0 << (vertex_lv - vertex_k.level()) * 3;

    let mut keys: [MortonKey; 8] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    (0..8u64).into_iter().for_each(|node_i| {
        // Perform the dilated integer substraction of dc - i.
        keys[node_i as usize] = MortonKey(
            (((dc & DIL_X) - (node_i & DIL_X)) & DIL_X)
                | (((dc & DIL_Y) - (node_i & DIL_Y)) & DIL_Y)
                | (((dc & DIL_Z) - (node_i & DIL_Z)) & DIL_Z),
        );
    });

    keys
}
