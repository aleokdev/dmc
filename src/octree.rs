use ahash::RandomState;
use bitflags::bitflags;
use cgmath::*;
use dashmap::{mapref::one::*, DashMap};

bitflags! {
    pub struct OctreeIndex : u8 {
        const RIGHT = 1 << 0;
        const TOP = 1 << 1;
        const FOREMOST = 1 << 2;
    }
}

impl<T> From<OctreeIndex> for Vector3<T>
where
    T: From<i8>,
{
    fn from(x: OctreeIndex) -> Self {
        let result = Vector3::new(
            if x.contains(OctreeIndex::RIGHT) {
                T::from(1)
            } else {
                T::from(-1)
            },
            if x.contains(OctreeIndex::TOP) {
                T::from(1)
            } else {
                T::from(-1)
            },
            if x.contains(OctreeIndex::FOREMOST) {
                T::from(-1)
            } else {
                T::from(1)
            },
        );

        result
    }
}

/// A 64-bit integer that indicates a single node in a hashed octree.
/// Its internal representation starts with 1 as the root node and appends 3 bits for each following
/// child node, indicating its position in the X, Y and Z axis.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct MortonKey(pub u64);

impl MortonKey {
    /// Returns a Morton key pointing to the octree root node.
    pub const fn root() -> Self {
        Self(1)
    }

    /// Returns a Morton key pointing to no node.
    pub const fn none() -> Self {
        Self(0)
    }

    /// Returns the parent of this Morton key.
    /// If this key is pointing to the root node, the node given will be equal to [`MortonKey::none()`].
    pub fn parent(self) -> Self {
        Self(self.0 >> 3)
    }

    /// Returns a child of the node this key is pointing to.
    /// # Panics
    /// Panics if the level of this key is equal to the maximum level possible.
    pub fn child(self, index: OctreeIndex) -> Self {
        assert!(self.level() < Self::max_level());
        Self(self.0 << 3 | index.bits as u64)
    }

    /// Returns the level or depth of this node, where 0 is the root node.
    /// # Panics
    /// Panics if this is a Morton key pointing to no node.
    pub fn level(self) -> u32 {
        assert_ne!(self.0, 0);
        (63 - self.0.leading_zeros()) / 3
    }

    /// Returns the maximum depth of the nodes that this type can point to.
    pub const fn max_level() -> u32 {
        63 / 3
    }

    /// Returns the position of the center of the node this key is pointing to.
    /// All axis are in the (0, 1) interval.
    ///
    pub fn position(self) -> Point3<f32> {
        let (mut x, mut y, mut z) = (0, 0, 0);
        let level = self.level();

        let mut bits = self.0;

        for i in 1..=level {
            x |= (bits & 1) << i;
            bits >>= 1;
            y |= (bits & 1) << i;
            bits >>= 1;
            z |= (bits & 1) << i;
            bits >>= 1;
        }

        let max_level_k = (1 << level) as f32;

        Point3::new(
            (x + 1) as f32 / max_level_k - 1.,
            (y + 1) as f32 / max_level_k - 1.,
            (z + 1) as f32 / max_level_k - 1.,
        )
    }

    /// Returns the Morton key with level `level` of an octree of bounds \[-1]³ to \[1]³ which is
    /// closest to the position `position`.
    /// # Panics
    /// Panics if `level < Self::max_level()`.
    pub fn closest_to_position(position: Point3<f32>, level: u32) -> Self {
        assert!(level < Self::max_level());

        let position = point3(
            position.x.clamp(-1., 1.),
            position.y.clamp(-1., 1.),
            position.z.clamp(-1., 1.),
        );

        let max_level_k = (1 << level) as f32;

        let (x, y, z) = (
            ((position.x + 1.) * max_level_k - 1.) as u64,
            ((position.y + 1.) * max_level_k - 1.) as u64,
            ((position.z + 1.) * max_level_k - 1.) as u64,
        );

        let mut bits = 1u64;

        for i in (1..=level).rev() {
            bits <<= 1;
            bits |= (z >> i) & 1;
            bits <<= 1;
            bits |= (y >> i) & 1;
            bits <<= 1;
            bits |= (x >> i) & 1;
        }

        Self(bits)
    }

    /// Returns a key that goes along the current key until a given level.
    /// # Panics
    /// If `self.level() < level`.
    pub fn until_level(self, level: u32) -> Self {
        assert!(self.level() >= level);
        Self(self.0 >> (self.level() - level) * 3)
    }
}

/// A hashed (linear) octree implementation which uses [`MortonKey`] for indexing.
/// This type supports concurrent access and uses a fast hashing algorithm for best performance.
/// # Example
/// ```rust
/// use dmc::octree::*;
///
/// let mut octree = HashedOctree::new(0usize);
///
/// let mut i = 0;
/// octree.subdivide(MortonKey::root()).for_each(|child| {
///     *octree.value_mut(child).unwrap() = i; i += 1;
/// });
/// dbg!(octree);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HashedOctree<T: Copy> {
    values: DashMap<MortonKey, T, RandomState>,
}

impl<T: Copy> HashedOctree<T> {
    pub fn new(root_value: T) -> Self {
        Self {
            values: {
                let x = DashMap::with_capacity_and_hasher(256, RandomState::new());
                x.insert(MortonKey::root(), root_value);
                x
            },
        }
    }

    pub fn value(&self, key: MortonKey) -> Option<Ref<'_, MortonKey, T, RandomState>> {
        self.values.get(&key)
    }

    pub fn value_mut(&mut self, key: MortonKey) -> Option<RefMut<'_, MortonKey, T, RandomState>> {
        self.values.get_mut(&key)
    }

    pub fn subdivide(&mut self, key: MortonKey) -> impl std::iter::Iterator<Item = MortonKey> {
        assert!(
            !self.is_subdivided(key),
            "Tried to subdivide already subdivided node"
        );
        let child_value = *self
            .value(key)
            .expect("Tried to subdivide null octree node");

        let vec = (0..8)
            .into_iter()
            .map(move |child| {
                let child_key = key.child(unsafe { OctreeIndex::from_bits_unchecked(child) });
                self.values.insert(child_key, child_value);
                child_key
            })
            .collect::<Vec<_>>();

        vec.into_iter()
    }

    pub fn children(&self, key: MortonKey) -> Option<impl std::iter::Iterator<Item = MortonKey>> {
        if self.is_subdivided(key) {
            Some((0..8).into_iter().map(move |child| {
                let child_key = key.child(unsafe { OctreeIndex::from_bits_unchecked(child) });
                child_key
            }))
        } else {
            None
        }
    }

    pub fn node_exists(&self, key: MortonKey) -> bool {
        self.values.contains_key(&key)
    }

    pub fn is_subdivided(&self, key: MortonKey) -> bool {
        self.values.contains_key(&(key.child(OctreeIndex::empty())))
    }

    /// Finds all the leaf nodes belonging to `parent`.
    pub fn leaves(&self, parent: MortonKey) -> Vec<MortonKey> {
        let mut leaves = Vec::with_capacity(128);
        let mut to_process = Vec::with_capacity(64);
        to_process.push(parent);

        while let Some(node_to_process) = to_process.pop() {
            if let Some(children) = self.children(node_to_process) {
                for child in children {
                    to_process.push(child);
                }
            } else {
                leaves.push(node_to_process);
            }
        }

        leaves
    }

    pub fn node_count(&self) -> usize {
        self.values.len()
    }
}
