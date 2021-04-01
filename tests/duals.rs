use dmc::duals::*;
use dmc::octree::*;

#[test]
fn complex_dual_grid_ok() {
    let mut octree = HashedOctree::new(1);
    octree.subdivide(MortonKey::root()).for_each(drop);
    octree.subdivide(MortonKey(0b1000)).for_each(drop);
    octree.subdivide(MortonKey(0b1010)).for_each(drop);
    octree.subdivide(MortonKey(0b1000001)).for_each(drop);
    octree.subdivide(MortonKey(0b1010111)).for_each(drop);

    let duals = DualGrid::from_octree(&octree);

    duals.volumes.into_iter().for_each(|keys| {
        for &key in &keys {
            // All keys should be valid leaf nodes
            assert_ne!(key.0, 0);
            assert!(!octree.is_subdivided(key));
        }
    })
}
