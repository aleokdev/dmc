use cgmath::vec3;
use dmc::dmc::*;
use dmc::octree::*;
mod trait_impls;
use trait_impls::Node;

#[test]
fn null_cube_marching() {
    let mut octree = HashedOctree::new(Node {
        density: 1.,
        normal: vec3(0., 0., 0.),
    });
    octree.subdivide(MortonKey::root()).for_each(drop);

    let mc = mesh_from_octree(&octree, 1.);

    assert_eq!(mc.vertices, vec![]);
}

#[test]
fn simple_single_point_cube_marching() {
    let mut octree = HashedOctree::new(Node {
        density: 1.,
        normal: vec3(0., 0., 0.),
    });
    octree.subdivide(MortonKey::root()).for_each(drop);
    // Change the value of the bottom left backmost node
    octree.value_mut(MortonKey(0b1000)).unwrap().density = -1.;

    let mc = mesh_from_octree(&octree, 1.);

    dbg!(&mc);
    assert_eq!(mc.vertices.len(), 3);
}

#[test]
fn simple_double_point_cube_marching() {
    let mut octree = HashedOctree::new(Node {
        density: 1.,
        normal: vec3(0., 0., 0.),
    });
    octree.subdivide(MortonKey::root()).for_each(drop);
    // Change the value of the bottom left backmost node and its opposite
    octree.value_mut(MortonKey(0b1000)).unwrap().density = -1.;
    octree.value_mut(MortonKey(0b1111)).unwrap().density = -1.;

    let mc = mesh_from_octree(&octree, 1.);

    dbg!(&mc);
    assert_eq!(mc.vertices.len(), 6);
}
