use ::dmc::octree::*;
use cgmath::Point3;

#[test]
fn node_positions() {
    assert_eq!(MortonKey(0b1).position(), Point3::new(0., 0., 0.));
    assert_eq!(MortonKey(0b1_111).position(), Point3::new(0.5, 0.5, 0.5));
    assert_eq!(
        MortonKey(0b1_111_010).position(),
        Point3::new(0.25, 0.75, 0.25)
    );
    assert_eq!(
        MortonKey(0b1_001_010).position(),
        Point3::new(0.25, -0.25, -0.75)
    );
}

#[test]
fn positions_to_nodes() {
    use cgmath::point3;

    assert_eq!(
        MortonKey::closest_to_position(point3(0., 0., 0.), 0),
        MortonKey(0b1)
    );
    assert_eq!(
        MortonKey::closest_to_position(point3(0.5, 0.5, 0.5), 1),
        MortonKey(0b1_111)
    );
    assert_eq!(
        MortonKey::closest_to_position(point3(0.25, 0.75, 0.25), 2),
        MortonKey(0b1_111_010)
    );
    assert_eq!(
        MortonKey::closest_to_position(point3(0.25, -0.25, -0.75), 2),
        MortonKey(0b1_001_010)
    );
}
