use crate::{self as bevy_asset, InternalAssetId};
use crate::{
    Asset, AssetEvent, AssetHandleProvider, AssetId, AssetServer, Handle, LoadState, UntypedHandle,
};
use bevy_ecs::{
    prelude::EventWriter,
    system::{Res, ResMut, Resource},
};
use bevy_reflect::{Reflect, TypePath};
use bevy_utils::{hashbrown, EntityHash, HashMap};
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::{
    any::TypeId,
    hash::Hash,
    iter::Enumerate,
    sync::{atomic::AtomicU32, Arc},
};
use thiserror::Error;
use uuid::Uuid;

/// A generational runtime-only identifier for a specific [`Asset`] stored in [`Assets`]. This is optimized for efficient runtime
/// usage and is not suitable for identifying assets across app runs.
#[derive(Debug, Default, Copy, Clone, Reflect, Serialize, Deserialize)]
#[repr(C, align(8))]
pub struct AssetIndex {
    // Do not reorder the fields here. The ordering is explicitly used by repr(C)
    // to make this struct equivalent to a u64.
    #[cfg(target_endian = "little")]
    pub(crate) index: u32,
    pub(crate) generation: u32,
    #[cfg(target_endian = "big")]
    pub(crate) index: u32,
}

// By not short-circuiting in comparisons, we get better codegen.
// See <https://github.com/rust-lang/rust/issues/117800>
impl PartialEq for AssetIndex {
    #[inline]
    fn eq(&self, other: &AssetIndex) -> bool {
        // By using `to_bits`, the codegen can be optimised out even
        // further potentially. Relies on the correct alignment/field
        // order of `AssetIndex`.
        self.to_bits() == other.to_bits()
    }
}

impl Eq for AssetIndex {}

// The derive macro codegen output is not optimal and can't be optimised as well
// by the compiler. This impl resolves the issue of non-optimal codegen by relying
// on comparing against the bit representation of `AssetIndex` instead of comparing
// the fields. The result is then LLVM is able to optimise the codegen for AssetIndex
// far beyond what the derive macro can.
// See <https://github.com/rust-lang/rust/issues/106107>
impl PartialOrd for AssetIndex {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Make use of our `Ord` impl to ensure optimal codegen output
        Some(self.cmp(other))
    }
}

// The derive macro codegen output is not optimal and can't be optimised as well
// by the compiler. This impl resolves the issue of non-optimal codegen by relying
// on comparing against the bit representation of `AssetIndex` instead of comparing
// the fields. The result is then LLVM is able to optimise the codegen for AssetIndex
// far beyond what the derive macro can.
// See <https://github.com/rust-lang/rust/issues/106107>
impl Ord for AssetIndex {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This will result in better codegen for ordering comparisons, plus
        // avoids pitfalls with regards to macro codegen relying on property
        // position when we want to compare against the bit representation.
        self.to_bits().cmp(&other.to_bits())
    }
}

impl Hash for AssetIndex {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state);
    }
}

impl AssetIndex {
    pub const INVALID: Self = Self {
        generation: u32::MAX,
        index: u32::MAX,
    };

    /// Convert the [`AssetIndex`] into an opaque blob of bits to transport it in circumstances where carrying a strongly typed index isn't possible.
    ///
    /// The result of this function should not be relied upon for anything except putting it back into [`AssetIndex::from_bits`] to recover the index.
    pub fn to_bits(self) -> u64 {
        let Self { generation, index } = self;
        ((generation as u64) << 32) | index as u64
    }
    /// Convert an opaque `u64` acquired from [`AssetIndex::to_bits`] back into an [`AssetIndex`]. This should not be used with any inputs other than those
    /// derived from [`AssetIndex::to_bits`], as there are no guarantees for what will happen with such inputs.
    pub fn from_bits(bits: u64) -> Self {
        let index = bits as u32;
        let generation = (bits >> 32) as u32;
        Self { generation, index }
    }
}

/// A [`HashMap`](hashbrown::HashMap) pre-configured to use [`EntityHash`] hashing.
pub type AssetHashMap<V> = hashbrown::HashMap<AssetIndex, V, EntityHash>;

/// A [`HashSet`](hashbrown::HashSet) pre-configured to use [`EntityHash`] hashing.
pub type AssetHashSet = hashbrown::HashSet<AssetIndex, EntityHash>;

/// Allocates generational [`AssetIndex`] values and facilitates their reuse.
pub(crate) struct AssetIndexAllocator {
    /// A monotonically increasing index.
    next_index: AtomicU32,
    recycled_queue_sender: Sender<AssetIndex>,
    /// This receives every recycled [`AssetIndex`]. It serves as a buffer/queue to store indices ready for reuse.
    recycled_queue_receiver: Receiver<AssetIndex>,
    recycled_sender: Sender<AssetIndex>,
    recycled_receiver: Receiver<AssetIndex>,
}

impl Default for AssetIndexAllocator {
    fn default() -> Self {
        let (recycled_queue_sender, recycled_queue_receiver) = crossbeam_channel::unbounded();
        let (recycled_sender, recycled_receiver) = crossbeam_channel::unbounded();
        Self {
            recycled_queue_sender,
            recycled_queue_receiver,
            recycled_sender,
            recycled_receiver,
            next_index: Default::default(),
        }
    }
}

impl AssetIndexAllocator {
    /// Reserves a new [`AssetIndex`], either by reusing a recycled index (with an incremented generation), or by creating a new index
    /// by incrementing the index counter for a given asset type `A`.
    pub fn reserve(&self) -> AssetIndex {
        if let Ok(mut recycled) = self.recycled_queue_receiver.try_recv() {
            recycled.generation += 1;
            self.recycled_sender.send(recycled).unwrap();
            recycled
        } else {
            AssetIndex {
                index: self
                    .next_index
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                generation: 0,
            }
        }
    }

    /// Queues the given `index` for reuse. This should only be done if the `index` is no longer being used.
    pub fn recycle(&self, index: AssetIndex) {
        self.recycled_queue_sender.send(index).unwrap();
    }
}

/// A "loaded asset" containing the untyped handle for an asset stored in a given [`AssetPath`].
///
/// [`AssetPath`]: crate::AssetPath
#[derive(Asset, TypePath)]
pub struct LoadedUntypedAsset {
    #[dependency]
    pub handle: UntypedHandle,
}

// PERF: do we actually need this to be an enum? Can we just use an "invalid" generation instead
#[derive(Default)]
enum Entry<A: Asset> {
    /// None is an indicator that this entry does not have live handles.
    #[default]
    None,
    /// Some is an indicator that there is a live handle active for the entry at this [`AssetIndex`]
    Some { value: Option<A>, generation: u32 },
}

/// Stores [`Asset`] values in a Vec-like storage identified by [`AssetIndex`].
struct DenseAssetStorage<A: Asset> {
    storage: Vec<Entry<A>>,
    len: u32,
    allocator: Arc<AssetIndexAllocator>,
}

impl<A: Asset> Default for DenseAssetStorage<A> {
    fn default() -> Self {
        Self {
            len: 0,
            storage: Default::default(),
            allocator: Default::default(),
        }
    }
}

impl<A: Asset> DenseAssetStorage<A> {
    // Returns the number of assets stored.
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    // Returns `true` if there are no assets stored.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert the value at the given index. Returns true if a value already exists (and was replaced)
    pub(crate) fn insert(
        &mut self,
        index: AssetIndex,
        asset: A,
    ) -> Result<bool, InvalidGenerationError> {
        self.flush();
        let entry = &mut self.storage[index.index as usize];
        if let Entry::Some { value, generation } = entry {
            if *generation == index.generation {
                let exists = value.is_some();
                if !exists {
                    self.len += 1;
                }
                *value = Some(asset);
                Ok(exists)
            } else {
                Err(InvalidGenerationError {
                    index,
                    current_generation: *generation,
                })
            }
        } else {
            unreachable!("entries should always be valid after a flush");
        }
    }

    /// Removes the asset stored at the given `index` and returns it as [`Some`] (if the asset exists).
    /// This will recycle the id and allow new entries to be inserted.
    pub(crate) fn remove_dropped(&mut self, index: AssetIndex) -> Option<A> {
        self.remove_internal(index, |dense_storage| {
            dense_storage.storage[index.index as usize] = Entry::None;
            dense_storage.allocator.recycle(index);
        })
    }

    /// Removes the asset stored at the given `index` and returns it as [`Some`] (if the asset exists).
    /// This will _not_ recycle the id. New values with the current ID can still be inserted. The ID will
    /// not be reused until [`DenseAssetStorage::remove_dropped`] is called.
    pub(crate) fn remove_still_alive(&mut self, index: AssetIndex) -> Option<A> {
        self.remove_internal(index, |_| {})
    }

    fn remove_internal(
        &mut self,
        index: AssetIndex,
        removed_action: impl FnOnce(&mut Self),
    ) -> Option<A> {
        self.flush();
        let value = match &mut self.storage[index.index as usize] {
            Entry::None => return None,
            Entry::Some { value, generation } => {
                if *generation == index.generation {
                    value.take().map(|value| {
                        self.len -= 1;
                        value
                    })
                } else {
                    return None;
                }
            }
        };
        removed_action(self);
        value
    }

    pub(crate) fn get(&self, index: AssetIndex) -> Option<&A> {
        let entry = self.storage.get(index.index as usize)?;
        match entry {
            Entry::None => None,
            Entry::Some { value, generation } => {
                if *generation == index.generation {
                    value.as_ref()
                } else {
                    None
                }
            }
        }
    }

    pub(crate) fn get_mut(&mut self, index: AssetIndex) -> Option<&mut A> {
        let entry = self.storage.get_mut(index.index as usize)?;
        match entry {
            Entry::None => None,
            Entry::Some { value, generation } => {
                if *generation == index.generation {
                    value.as_mut()
                } else {
                    None
                }
            }
        }
    }

    pub(crate) fn flush(&mut self) {
        // NOTE: this assumes the allocator index is monotonically increasing.
        let new_len = self
            .allocator
            .next_index
            .load(std::sync::atomic::Ordering::Relaxed);
        self.storage.resize_with(new_len as usize, || Entry::Some {
            value: None,
            generation: 0,
        });
        while let Ok(recycled) = self.allocator.recycled_receiver.try_recv() {
            let entry = &mut self.storage[recycled.index as usize];
            *entry = Entry::Some {
                value: None,
                generation: recycled.generation,
            };
        }
    }

    pub(crate) fn get_index_allocator(&self) -> Arc<AssetIndexAllocator> {
        self.allocator.clone()
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = AssetId<A>> + '_ {
        self.storage
            .iter()
            .enumerate()
            .filter_map(|(i, v)| match v {
                Entry::None => None,
                Entry::Some { value, generation } => {
                    if value.is_some() {
                        Some(AssetId::from(AssetIndex {
                            index: i as u32,
                            generation: *generation,
                        }))
                    } else {
                        None
                    }
                }
            })
    }
}

/// Stores [`Asset`] values identified by their [`AssetId`].
///
/// Assets identified by [`AssetId::Index`] will be stored in a "dense" vec-like storage. This is more efficient, but it means that
/// the assets can only be identified at runtime. This is the default behavior.
///
/// Assets identified by [`AssetId::Uuid`] will be stored in a hashmap. This is less efficient, but it means that the assets can be referenced
/// at compile time.
///
/// This tracks (and queues) [`AssetEvent`] events whenever changes to the collection occur.
#[derive(Resource)]
pub struct Assets<A: Asset> {
    dense_storage: DenseAssetStorage<A>,
    // Uuid assets are stored in dense storage
    hash_map: HashMap<Uuid, AssetIndex>,
    handle_provider: AssetHandleProvider,
    queued_events: Vec<AssetEvent<A>>,
    /// Assets managed by the `Assets` struct with live strong `Handle`s
    /// originating from `get_strong_handle`.
    duplicate_handles: HashMap<AssetId<A>, u16>,
}

impl<A: Asset> Default for Assets<A> {
    fn default() -> Self {
        let dense_storage = DenseAssetStorage::default();
        let handle_provider =
            AssetHandleProvider::new(TypeId::of::<A>(), dense_storage.get_index_allocator());
        Self {
            dense_storage,
            handle_provider,
            hash_map: Default::default(),
            queued_events: Default::default(),
            duplicate_handles: Default::default(),
        }
    }
}

impl<A: Asset> Assets<A> {
    /// Retrieves an [`AssetHandleProvider`] capable of reserving new [`Handle`] values for assets that will be stored in this
    /// collection.
    pub fn get_handle_provider(&self) -> AssetHandleProvider {
        self.handle_provider.clone()
    }

    /// Reserves a new [`Handle`] for an asset that will be stored in this collection.
    pub fn reserve_handle(&self) -> Handle<A> {
        self.handle_provider.reserve_handle().typed::<A>()
    }

    /// Inserts the given `asset`, identified by the given `id`. If an asset already exists for `id`, it will be replaced.
    pub fn insert(&mut self, id: impl Into<AssetId<A>>, asset: A) {
        let id = id.into();
        if let Some(uuid) = id.uuid() {
            self.insert_with_uuid(uuid, asset);
        } else {
            self.insert_with_index(id.index(), asset).unwrap();
        }
    }

    /// Retrieves an [`Asset`] stored for the given `id` if it exists. If it does not exist, it will be inserted using `insert_fn`.
    // PERF: Optimize this or remove it
    pub fn get_or_insert_with(
        &mut self,
        id: impl Into<AssetId<A>>,
        insert_fn: impl FnOnce() -> A,
    ) -> &mut A {
        let id: AssetId<A> = id.into();
        if self.get(id).is_none() {
            self.insert(id, insert_fn());
        }
        self.get_mut(id).unwrap()
    }

    /// Returns `true` if the `id` exists in this collection. Otherwise it returns `false`.
    pub fn contains(&self, id: impl Into<AssetId<A>>) -> bool {
        self.dense_storage.get(id.into().index()).is_some()
    }

    // No longer necessary
    pub fn insert_with_uuid(&mut self, uuid: Uuid, asset: A) -> (Handle<A>, Option<A>) {
        if let Some(&index) = self.hash_map.get(&uuid) {
            let old = self.dense_storage.remove_still_alive(index);
            self.dense_storage.insert(index, asset).unwrap();
            let id = AssetId::<A>::new(Some(uuid), index);
            self.queued_events.push(AssetEvent::Modified { id });
            (
                Handle::Strong(
                    self.handle_provider
                        .get_handle(id.into(), false, None, None),
                ),
                old,
            )
        } else {
            let index = self.dense_storage.allocator.reserve();
            self.hash_map.insert(uuid, index);
            self.dense_storage.insert(index, asset).unwrap();
            let id = AssetId::<A>::new(Some(uuid), index);
            self.queued_events.push(AssetEvent::Added { id });
            (
                Handle::Strong(
                    self.handle_provider
                        .get_handle(id.into(), false, None, None),
                ),
                None,
            )
        }
    }

    pub(crate) fn insert_with_index(
        &mut self,
        index: AssetIndex,
        asset: A,
    ) -> Result<bool, InvalidGenerationError> {
        let replaced = self.dense_storage.insert(index, asset)?;
        if replaced {
            self.queued_events
                .push(AssetEvent::Modified { id: index.into() });
        } else {
            self.queued_events
                .push(AssetEvent::Added { id: index.into() });
        }
        Ok(replaced)
    }

    /// Adds the given `asset` and allocates a new strong [`Handle`] for it.
    #[inline]
    pub fn add(&mut self, asset: impl Into<A>) -> Handle<A> {
        let index = self.dense_storage.allocator.reserve();
        self.insert_with_index(index, asset.into()).unwrap();
        Handle::Strong(
            self.handle_provider
                .get_handle(index.into(), false, None, None),
        )
    }

    /// Upgrade an `AssetId` into a strong `Handle` that will prevent asset drop.
    ///
    /// Returns `None` if the provided `id` is not part of this `Assets` collection.
    /// For example, it may have been dropped earlier.
    #[inline]
    pub fn get_strong_handle(&mut self, id: AssetId<A>) -> Option<Handle<A>> {
        if !self.contains(id) {
            return None;
        }
        *self.duplicate_handles.entry(id).or_insert(0) += 1;
        Some(Handle::Strong(self.handle_provider.get_handle(
            InternalAssetId::from(id),
            false,
            None,
            None,
        )))
    }

    /// Retrieves a reference to the [`Asset`] with the given `id`, if it exists.
    /// Note that this supports anything that implements `Into<AssetId<A>>`, which includes [`Handle`] and [`AssetId`].
    #[inline]
    pub fn get(&self, id: impl Into<AssetId<A>>) -> Option<&A> {
        self.dense_storage.get(id.into().index())
    }

    #[inline]
    pub fn get_with_index(&self, index: AssetIndex) -> Option<&A> {
        self.dense_storage.get(index)
    }

    /// Retrieves a mutable reference to the [`Asset`] with the given `id`, if it exists.
    /// Note that this supports anything that implements `Into<AssetId<A>>`, which includes [`Handle`] and [`AssetId`].
    #[inline]
    pub fn get_mut(&mut self, id: impl Into<AssetId<A>>) -> Option<&mut A> {
        let id: AssetId<A> = id.into();
        let result = self.dense_storage.get_mut(id.index());
        if result.is_some() {
            self.queued_events.push(AssetEvent::Modified { id });
        }
        result
    }

    /// Removes (and returns) the [`Asset`] with the given `id`, if it exists.
    /// Note that this supports anything that implements `Into<AssetId<A>>`, which includes [`Handle`] and [`AssetId`].
    pub fn remove(&mut self, id: impl Into<AssetId<A>>) -> Option<A> {
        let id: AssetId<A> = id.into();
        let result = self.remove_untracked(id);
        if result.is_some() {
            self.queued_events.push(AssetEvent::Removed { id });
        }
        result
    }

    /// Removes (and returns) the [`Asset`] with the given `id`, if it exists. This skips emitting [`AssetEvent::Removed`].
    /// Note that this supports anything that implements `Into<AssetId<A>>`, which includes [`Handle`] and [`AssetId`].
    pub fn remove_untracked(&mut self, id: impl Into<AssetId<A>>) -> Option<A> {
        let id: AssetId<A> = id.into();
        self.duplicate_handles.remove(&id);
        self.dense_storage.remove_still_alive(id.index())
    }

    /// Removes the [`Asset`] with the given `id`.
    pub(crate) fn remove_dropped(&mut self, id: AssetId<A>) {
        match self.duplicate_handles.get_mut(&id) {
            None | Some(0) => {}
            Some(value) => {
                *value -= 1;
                return;
            }
        }
        let existed = self.dense_storage.remove_dropped(id.index()).is_some();
        if existed {
            self.queued_events.push(AssetEvent::Removed { id });
        }
    }

    /// Returns `true` if there are no assets in this collection.
    pub fn is_empty(&self) -> bool {
        self.dense_storage.is_empty()
    }

    /// Returns the number of assets currently stored in the collection.
    pub fn len(&self) -> usize {
        self.dense_storage.len()
    }

    /// Returns an iterator over the [`AssetId`] of every [`Asset`] stored in this collection.
    pub fn ids(&self) -> impl Iterator<Item = AssetId<A>> + '_ {
        self.dense_storage.ids()
    }

    /// Returns an iterator over the [`AssetId`] and [`Asset`] ref of every asset in this collection.
    // PERF: this could be accelerated if we implement a skip list. Consider the cost/benefits
    pub fn iter(&self) -> impl Iterator<Item = (AssetId<A>, &A)> {
        self.dense_storage
            .storage
            .iter()
            .enumerate()
            .filter_map(|(i, v)| match v {
                Entry::None => None,
                Entry::Some { value, generation } => value.as_ref().map(|v| {
                    (
                        AssetId::<A>::from(AssetIndex {
                            generation: *generation,
                            index: i as u32,
                        }),
                        v,
                    )
                }),
            })
    }

    /// Returns an iterator over the [`AssetId`] and mutable [`Asset`] ref of every asset in this collection.
    // PERF: this could be accelerated if we implement a skip list. Consider the cost/benefits
    pub fn iter_mut(&mut self) -> AssetsMutIterator<'_, A> {
        AssetsMutIterator {
            dense_storage: self.dense_storage.storage.iter_mut().enumerate(),
            queued_events: &mut self.queued_events,
        }
    }

    /// A system that synchronizes the state of assets in this collection with the [`AssetServer`]. This manages
    /// [`Handle`] drop events.
    pub fn track_assets(mut assets: ResMut<Self>, asset_server: Res<AssetServer>) {
        let assets = &mut *assets;
        // note that we must hold this lock for the entire duration of this function to ensure
        // that `asset_server.load` calls that occur during it block, which ensures that
        // re-loads are kicked off appropriately. This function must be "transactional" relative
        // to other asset info operations
        let mut infos = asset_server.data.infos.write();
        let mut not_ready = Vec::new();
        while let Ok(drop_event) = assets.handle_provider.drop_receiver.try_recv() {
            let id = drop_event.id.typed();

            if drop_event.asset_server_managed {
                let untyped_id = id.untyped();
                if let Some(info) = infos.get(untyped_id) {
                    if let LoadState::Loading | LoadState::NotLoaded = info.load_state {
                        not_ready.push(drop_event);
                        continue;
                    }
                }

                // the process_handle_drop call checks whether new handles have been created since the drop event was fired, before removing the asset
                if !infos.process_handle_drop(untyped_id) {
                    // a new handle has been created, or the asset doesn't exist
                    continue;
                }
            }

            assets.queued_events.push(AssetEvent::Unused { id });
            assets.remove_dropped(id);
        }

        // TODO: this is _extremely_ inefficient find a better fix
        // This will also loop failed assets indefinitely. Is that ok?
        for event in not_ready {
            assets.handle_provider.drop_sender.send(event).unwrap();
        }
    }

    /// A system that applies accumulated asset change events to the [`Events`] resource.
    ///
    /// [`Events`]: bevy_ecs::event::Events
    pub fn asset_events(mut assets: ResMut<Self>, mut events: EventWriter<AssetEvent<A>>) {
        events.send_batch(assets.queued_events.drain(..));
    }

    /// A run condition for [`asset_events`]. The system will not run if there are no events to
    /// flush.
    ///
    /// [`asset_events`]: Self::asset_events
    pub(crate) fn asset_events_condition(assets: Res<Self>) -> bool {
        !assets.queued_events.is_empty()
    }
}

/// A mutable iterator over [`Assets`].
pub struct AssetsMutIterator<'a, A: Asset> {
    queued_events: &'a mut Vec<AssetEvent<A>>,
    dense_storage: Enumerate<std::slice::IterMut<'a, Entry<A>>>,
}

impl<'a, A: Asset> Iterator for AssetsMutIterator<'a, A> {
    type Item = (AssetId<A>, &'a mut A);

    fn next(&mut self) -> Option<Self::Item> {
        for (i, entry) in &mut self.dense_storage {
            match entry {
                Entry::None => {
                    continue;
                }
                Entry::Some { value, generation } => {
                    let id = AssetId::from(AssetIndex {
                        generation: *generation,
                        index: i as u32,
                    });
                    self.queued_events.push(AssetEvent::Modified { id });
                    if let Some(value) = value {
                        return Some((id, value));
                    }
                }
            }
        }
        None
    }
}

#[derive(Error, Debug)]
#[error("AssetIndex {index:?} has an invalid generation. The current generation is: '{current_generation}'.")]
pub struct InvalidGenerationError {
    index: AssetIndex,
    current_generation: u32,
}

#[cfg(test)]
mod test {
    use crate::AssetIndex;

    #[test]
    fn asset_index_round_trip() {
        let asset_index = AssetIndex {
            generation: 42,
            index: 1337,
        };
        let roundtripped = AssetIndex::from_bits(asset_index.to_bits());
        assert_eq!(asset_index, roundtripped);
    }
}
