mod draw;
mod draw_state;

pub use draw::*;
pub use draw_state::*;

use bevy_ecs::prelude::{Component, Query};

use copyless::VecHelper;
use voracious_radix_sort::{RadixSort, Radixable};

/// A resource to collect and sort draw requests for specific [`PhaseItems`](PhaseItem).
#[derive(Component)]
pub struct RenderPhase<I: PhaseItem> {
    pub items: Vec<I>,
}

impl<I: PhaseItem> Default for RenderPhase<I> {
    fn default() -> Self {
        Self { items: Vec::new() }
    }
}

impl<I: PhaseItem + Radixable<<I as PhaseItem>::SortKey> + Copy> RenderPhase<I> {
    /// Adds a [`PhaseItem`] to this render phase.
    #[inline]
    pub fn add(&mut self, item: I) {
        self.items.alloc().init(item);
    }

    /// Sorts all of its [`PhaseItems`](PhaseItem).
    pub fn sort(&mut self) {
        self.items.voracious_stable_sort();
    }
}

impl<I: BatchedPhaseItem> RenderPhase<I> {
    /// Batches the compatible [`BatchedPhaseItem`]s of this render phase
    pub fn batch(&mut self) {
        // TODO: this could be done in-place
        let mut items = std::mem::take(&mut self.items);
        let mut items = items.drain(..);

        self.items.reserve(items.len());

        // Start the first batch from the first item
        if let Some(mut current_batch) = items.next() {
            // Batch following items until we find an incompatible item
            for next_item in items {
                if matches!(
                    current_batch.add_to_batch(&next_item),
                    BatchResult::IncompatibleItems
                ) {
                    // Store the completed batch, and start a new one from the incompatible item
                    self.items.push(current_batch);
                    current_batch = next_item;
                }
            }
            // Store the last batch
            self.items.push(current_batch);
        }
    }
}

/// This system sorts all [`RenderPhases`](RenderPhase) for the [`PhaseItem`] type.
pub fn sort_phase_system<I: PhaseItem + Radixable<<I as PhaseItem>::SortKey> + Copy>(
    mut render_phases: Query<&mut RenderPhase<I>>,
) {
    for mut phase in render_phases.iter_mut() {
        phase.sort();
    }
}

/// This system batches the [`PhaseItem`]s of all [`RenderPhase`]s of this type.
pub fn batch_phase_system<I: BatchedPhaseItem>(mut render_phases: Query<&mut RenderPhase<I>>) {
    for mut phase in render_phases.iter_mut() {
        phase.batch();
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use bevy_ecs::entity::Entity;

    use super::*;

    #[test]
    fn batching() {
        #[derive(Debug, Clone, Copy)]
        struct TestPhaseItem {
            entity: Entity,
            batch_range: Option<BatchRange>,
        }
        impl PartialOrd for TestPhaseItem {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.sort_key().partial_cmp(&other.sort_key())
            }
        }
        impl PartialEq for TestPhaseItem {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.sort_key() == other.sort_key()
            }
        }
        impl Radixable<<Self as PhaseItem>::SortKey> for TestPhaseItem {
            type Key = <Self as PhaseItem>::SortKey;

            #[inline]
            fn key(&self) -> Self::Key {
                self.sort_key()
            }
        }
        impl PhaseItem for TestPhaseItem {
            type SortKey = f32;

            #[inline]
            fn sort_key(&self) -> Self::SortKey {
                0.0f32
            }

            fn draw_function(&self) -> DrawFunctionId {
                unimplemented!();
            }
        }
        impl EntityPhaseItem for TestPhaseItem {
            fn entity(&self) -> bevy_ecs::entity::Entity {
                self.entity
            }
        }
        impl BatchedPhaseItem for TestPhaseItem {
            fn batch_range(&self) -> &Option<BatchRange> {
                &self.batch_range
            }

            fn batch_range_mut(&mut self) -> &mut Option<BatchRange> {
                &mut self.batch_range
            }
        }
        let mut render_phase = RenderPhase::<TestPhaseItem>::default();
        let items = [
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(0, 5)),
            },
            // This item should be batched
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(5, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(0, 5)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(5, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: None,
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(20, 25)),
            },
            // This item should be batched
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(25, 30)),
            },
            // This item should be batched
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(30, 35)),
            },
        ];
        for item in items {
            render_phase.add(item);
        }
        render_phase.batch();
        let items_batched = [
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(0, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(0, 5)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(0),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(5, 10)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: None,
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(10, 15)),
            },
            TestPhaseItem {
                entity: Entity::from_raw(1),
                batch_range: Some(BatchRange::new(20, 35)),
            },
        ];
        assert_eq!(&*render_phase.items, items_batched);
    }
}
