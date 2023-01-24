use std::fmt::Debug;
use std::hash::Hash;
use std::mem;

use crate as bevy_ecs;
use crate::scheduling::{ScheduleLabel, SystemSet};
use crate::system::Resource;
use crate::world::World;

/// Types that can define states in a finite-state machine.
///
/// The [`Default`] trait defines the starting state.
pub trait States:
    'static + Send + Sync + Clone + Copy + PartialEq + Eq + Hash + Debug + Default
{
    type Iter: Iterator<Item = Self>;

    /// Returns an iterator over all the state variants.
    fn variants() -> Self::Iter;
}

/// The label of a [`Schedule`](super::Schedule) that runs whenever [`State<S>`]
/// enters this state.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OnEnter<S: States>(pub S);

/// The label of a [`Schedule`](super::Schedule) that runs whenever [`State<S>`]
/// exits this state.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OnExit<S: States>(pub S);

/// A [`SystemSet`] that will run within `CoreSet::StateTransitions` when this state is active.
///
/// This is provided for convenience. A more general [`state_equals`](super::state_equals)
/// [condition](super::Condition) also exists for systems that need to run elsewhere.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OnUpdate<S: States>(pub S);

/// A finite-state machine whose transitions have associated schedules
/// ([`OnEnter(state)`] and [`OnExit(state)`]).
///
/// The current state value can be accessed through this resource. To *change* the state,
/// queue a transition in the [`NextState<S>`] resource, and it will be applied by the next
/// [`apply_state_transition::<S>`] system.
///
/// The starting state is defined via the [`Default`] implementation for `S`.
#[derive(Resource, Default)]
pub struct State<S: States>(pub S);

/// The next state of [`State<S>`].
///
/// To queue a transition, just set the contained value to `Some(next_state)`.
#[derive(Resource, Default)]
pub struct NextState<S: States>(pub Option<S>);

impl<S: States> NextState<S> {
    /// Queue the transition to a new `state`.
    pub fn queue(&mut self, state: S) {
        self.0 = Some(state);
    }
}

/// If a new state is queued in [`NextState<S>`], this system:
/// - Takes the new state value from [`NextState<S>`] and updates [`State<S>`].
/// - Runs the [`OnExit(exited_state)`] schedule.
/// - Runs the [`OnEnter(entered_state)`] schedule.
pub fn apply_state_transition<S: States>(world: &mut World) {
    if world.resource::<NextState<S>>().0.is_some() {
        let entered_state = world.resource_mut::<NextState<S>>().0.take().unwrap();
        let exited_state = mem::replace(&mut world.resource_mut::<State<S>>().0, entered_state);
        world.run_schedule(&OnExit(exited_state));
        world.run_schedule(&OnEnter(entered_state));
    }
}