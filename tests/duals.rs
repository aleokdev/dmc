use dmc::duals::*;
use dmc::octree::*;

#[test]
fn dual_grid_generation() {
    let mut octree = HashedOctree::new(1);
    octree.subdivide(MortonKey::root()).for_each(drop);

    assert!(octree.is_subdivided(MortonKey::root()));
    assert_eq!(
        octree.leaves(MortonKey::root()),
        vec![
            MortonKey(0b1111),
            MortonKey(0b1110),
            MortonKey(0b1101),
            MortonKey(0b1100),
            MortonKey(0b1011),
            MortonKey(0b1010),
            MortonKey(0b1001),
            MortonKey(0b1000),
        ]
    );

    let duals = DualGrid::from_octree(&octree);

    assert_eq!(
        duals.volumes,
        vec![[
            MortonKey(0b1111),
            MortonKey(0b1110),
            MortonKey(0b1101),
            MortonKey(0b1100),
            MortonKey(0b1011),
            MortonKey(0b1010),
            MortonKey(0b1001),
            MortonKey(0b1000),
        ]]
    );
}

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
