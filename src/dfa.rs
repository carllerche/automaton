use core::{Action, Alphabet, Letter, State, Transitions};
use util;
use std::{cmp, hash, mem, ops};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;

pub struct Dfa<S> {
    states: HashSet<State>,
    alphabet: Alphabet,
    transitions: Transitions,
    start: State,
    terminal: HashMap<State, Vec<usize>>,
    actions: Vec<Action<S>>,
}

impl<S> Dfa<S> {
    pub fn empty() -> Dfa<S> {
        let mut terminal = HashMap::new();
        terminal.insert(0, vec![]);

        Dfa {
            states: set![0],
            alphabet: Alphabet::empty(),
            transitions: Transitions::empty(),
            start: 0,
            terminal: terminal,
            actions: vec![],
        }
    }

    pub fn token(letter: Letter) -> Dfa<S> {
        let mut terminal = HashMap::new();
        terminal.insert(1, vec![]);

        Dfa {
            states: set![0, 1],
            alphabet: Alphabet::single(letter.clone()),
            transitions: Transitions::of(0, 1, letter),
            start: 0,
            terminal: terminal,
            actions: vec![],
        }
    }

    pub fn sequence<I: Iterator<Item=Letter>>(letters: I) -> Dfa<S> {
        let start = 0;
        let mut states = set![start];
        let mut alphabet = Alphabet::empty();
        let mut terminal = HashMap::new();
        let mut transitions = Transitions::empty();
        let mut prev = start;

        for (i, letter) in letters.enumerate() {
            let state = i as State + 1;

            states.insert(state);
            alphabet.insert(letter.clone());
            transitions.letter(prev, state, letter, vec![]);
            prev = state;
        }

        terminal.insert(prev, vec![]);

        let mut ret = Dfa {
            states: states,
            alphabet: alphabet,
            transitions: transitions,
            start: start,
            terminal: terminal,
            actions: vec![],
        };

        // Optimize the automaton
        ret.optimize(&Context::new());

        ret
    }

    pub fn concat(mut self, mut other: Dfa<S>) -> Dfa<S> {
        // Ensure states don't overlap
        other.shift(self.next_state_id(), self.actions.len());

        // Add epsilon transitions from all final states to start of `other`.
        // Embed the exit actions
        for (exit, actions) in &self.terminal {
            // Dup other entry state
            let entry = other.dup_start_state();

            // If there are any actions, embed them
            if !actions.is_empty() {
                other.transitions.embed_actions_from(entry, actions);
            }

            // If the start state is also an exit, embed the action in the exit
            // as well
            if let Entry::Occupied(mut e) = other.terminal.entry(entry) {
                for action in actions {
                    insert_action(e.get_mut(), *action);
                }
            }

            self.transitions.epsilon(*exit, entry, vec![]);
        }

        // Add states from `other`
        self.states.extend(other.states);

        // Extend the alphabet
        self.alphabet.extend(&other.alphabet);

        // Add transitions from `other`
        self.transitions.extend(other.transitions);

        // Set terminal states to `other`
        self.terminal = other.terminal;

        self.actions.extend(other.actions);

        // Optimize the representation
        self.optimize(&Context::new());

        self
    }

    pub fn union(mut self, other: Dfa<S>) -> Dfa<S> {
        self.union_or_intersection(other, false)
    }

    pub fn intersection(mut self, other: Dfa<S>) -> Dfa<S> {
        self.union_or_intersection(other, true)
    }

    fn union_or_intersection(mut self, mut other: Dfa<S>, intersection: bool) -> Dfa<S> {
        // Ensure states don't overlap
        other.shift(self.next_state_id(), self.actions.len());

        // Optimization context
        let mut ctx = Context::new();

        if intersection {
            let a: HashSet<State> = self.terminal.keys().cloned().collect();
            let b: HashSet<State> = other.terminal.keys().cloned().collect();
            ctx.intersect = Some((a, b));
        }

        // Add the states from other
        self.states.extend(other.states);

        // Extend the alphabet
        self.alphabet.extend(&other.alphabet);

        // Merge transitions
        self.transitions.extend(other.transitions);

        debug!("  SELF TERM:  {:?}", self.terminal);
        debug!("  OTHER TERM: {:?}", other.terminal);

        // Merge terminal states
        self.terminal.extend(other.terminal);

        // Create a new start state
        let state = self.next_state_id();
        self.states.insert(state);

        // Create epsilon transitions to the start of the current automaton and
        // the other one that is being unioned
        self.transitions.epsilon(state, self.start, vec![]);
        self.transitions.epsilon(state, other.start, vec![]);

        // Update the automaton's start transition
        self.start = state;

        // Combine the actions
        self.actions.extend(other.actions);

        debug!("~~~~~~~~~~~~~~ OPTIMIZE UNION ~~~~~~~~~~~~~~");
        debug!("  States: {}", self.states.len());

        // Optimize the automaton, converting to a DFA and merging all similar
        // states
        self.optimize(&ctx);

        self
    }

    pub fn kleene(mut self) -> Dfa<S> {
        let new_start = self.next_state_id();
        self.states.insert(new_start);

        // TODO: Any terminal action must be embedded into the start state but
        // only after an iteration has occurred. In order for this to work, the
        // start state must be duped

        let noop = self.next_state_id();
        self.states.insert(noop);

        self.transitions.epsilon(new_start, self.start, vec![]);
        self.transitions.epsilon(new_start, noop, vec![]);

        // Any terminal action must also be embedded as an enter transition
        // when the machine returns to the start position, but the action must
        // not be called the first time. This is done by duplicating the start
        // state and pointing the terminal states to the duplicated start
        // state.

        let second_start = self.dup_start_state();

        for (exit, actions) in &self.terminal {
            self.transitions.epsilon(*exit, new_start, vec![]);
            self.transitions.embed_actions_from(new_start, actions);
        }

        self.start = new_start;
        self.terminal.insert(noop, vec![]);

        self.optimize(&Context::new());

        self
    }

    /*
     *
     * ===== Actions =====
     *
     */

    pub fn on_enter(&mut self, action: Action<S>) {
        let idx = self.push_action(action);

        // Isolate the start state to allow embedding actions only in the entry
        // transition
        self.isolate_start_state();

        // Embed the action in the enter transition
        self.transitions.embed_actions_from(self.start, &[idx]);
    }

    pub fn on_exit(&mut self, action: Action<S>) {
        let idx = self.push_action(action);

        for (_, v) in self.terminal.iter_mut() {
            insert_action(v, idx);
        }
    }

    // If any state transitions to the start state, duplicate the start state
    // and update the transitions to move to the newly duplicated state. This
    // allows actions to be embedded when "entering" into the machine without
    // having the actions executed each time execution returns to the start
    // state.
    fn isolate_start_state(&mut self) {
        let isolated = self.next_state_id();

        // Returns true if any transitions are updated
        if self.transitions.remap_dest(self.start, isolated) {
            // Copy all transitions originating from the start state
            self.transitions.dup_from(self.start, isolated);

            // If the start state is also a terminal state then the newly
            // created state is also a terminal state.
            if self.terminal.contains_key(&self.start) {
                let actions = self.terminal[&self.start].clone();
                self.terminal.insert(isolated, actions);
            }

            // Commit the newly committed state
            self.states.insert(isolated);
        }
    }

    // Duplicates and returns the start state
    fn dup_start_state(&mut self) -> State {
        let dup = self.next_state_id();

        // Commit the state
        self.states.insert(dup);

        // Duplicate the transitions
        self.transitions.dup_from(self.start, dup);

        // Make terminal if source state is also terminal
        if self.terminal.contains_key(&self.start) {
            let actions = self.terminal[&self.start].clone();
            self.terminal.insert(dup, actions);
        }

        dup
    }

    fn push_action(&mut self, action: Action<S>) -> usize {
        let ret = self.actions.len();
        self.actions.push(action);
        ret
    }

    /*
     *
     * ===== Optimize =====
     *
     */

    // Optimize the representation of the automaton
    fn optimize(&mut self, ctx: &Context) {
        // Ensure the alphabet tokens are disjoint
        self.refine_alphabet();

        debug!("BEFORE OPTIMIZE");
        debug!("   states: {:?}", self.states);
        debug!("   terminal: {:?}", self.terminal);
        debug!("   transitions:");
        self.transitions.each(|t| {
            debug!("      {:?}", t);
        });
        debug!("");

        // Convert to a DFA
        self.to_dfa(ctx);

        debug!("DFA CONVERSION");
        debug!("   states: {:?}", self.states);
        debug!("   terminal: {:?}", self.terminal);
        debug!("   transitions:");
        self.transitions.each(|t| {
            debug!("      {:?}", t);
        });
        debug!("");

        // Prune dead states
        self.prune();

        // Minimize the state machine
        self.minimize();

        // Verify that things are as expected
        self.verify();
    }

    pub fn states(&self) -> &HashSet<State> {
        &self.states
    }

    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    pub fn transitions(&self) -> &Transitions {
        &self.transitions
    }

    pub fn start(&self) -> &State {
        &self.start
    }

    pub fn terminal(&self) -> &HashMap<State, Vec<usize>> {
        &self.terminal
    }

    pub fn actions(self) -> Vec<Action<S>> {
        self.actions
    }

    /*
     *
     * ===== Alphabet refinement =====
     *
     */

    // At this point, the automaton could contain tokens that overlap with each
    // other. Find these tokens, split them to ensure a disjoint alphabet and
    // create new transitions to maintain the same semantics.
    //
    // It should not be possible at this point, given a state, to have a single
    // token transition to two possible final states.
    //
    // TODO: Optimize
    fn refine_alphabet<'a>(&'a mut self) {
        // First, reduce the alphabet to a set of fully disjoint tokens
        self.alphabet.refine();

        // Transitions to insert
        let mut additions = vec![];

        self.transitions.each(|transition| {
            let token = match transition.input() {
                Some(token) => token,
                None => return,
            };

            // Quick path
            if self.alphabet.contains(&token) {
                return;
            }

            for other in self.alphabet.iter() {
                if token.contains(&other) {
                    additions.push((transition.from(), transition.to(), other.clone(), transition.actions().to_vec()));
                }
            }
        });

        for (from, to, input, actions) in additions {
            self.transitions.insert(from, to, Some(input), actions);
        }

        let alphabet = &self.alphabet;

        self.transitions.retain(|t| {
            match t.input() {
                Some(token) => alphabet.contains(token),
                None => true,
            }
        });
    }

    /*
     *
     * ===== DFA minimization =====
     *
     */

    // Prunes dead-end states by starting from terminal states and walking
    // backwards through the state machine.
    fn prune(&mut self) {
        let mut remaining: HashSet<State> =
            self.terminal.keys().cloned().collect();

        // Start by setting the states to the set of terminal states
        self.states.clear();
        self.states.extend(remaining.iter().cloned());

        while !remaining.is_empty() {
            let mut new: HashSet<State> = HashSet::with_capacity(self.transitions.len());

            self.transitions.each(|transition| {
                if remaining.contains(&transition.to()) {
                    new.insert(transition.from());
                }
            });

            // Update the set of remaining states to process
            remaining.clear();
            remaining.extend(new.difference(&self.states).cloned());

            self.states.extend(new.iter().cloned());
        }

        assert!(self.states.contains(&self.start), "invalid automaton");

        // Finally, remove orphaned transitions
        self.prune_transitions();
    }

    fn prune_transitions(&mut self) {
        let states = &self.states;

        self.transitions.retain(|transition| {
            states.contains(&transition.from()) &&
                states.contains(&transition.to())
        });
    }

    fn minimize(&mut self) {
        let mut minimize = Minimize::new(
            self.terminal.keys().cloned().collect(),
            self.nonterminal());

        // Step 1) Refine the partitions
        self.refine(&mut minimize);

        // Step 2) Apply the refinement
        self.apply_refinement(&mut minimize);
    }

    // Refine the partitions. This is done by removing an (any) partition from
    // the set of remaining partitions. The reason why a set is used is to be
    // able to perform set ops on the remaining partitions.
    fn refine(&mut self, minimize: &mut Minimize) {
        while let Some(state) = util::pop(&mut minimize.remaining) {
            debug!("... iterating; curr={:?}", state);
            debug!("         partitions={:?}", minimize.partitions);
            for token in self.alphabet.iter() {
                let x = self.reached(&state, Some(&token));

                if x.is_empty() {
                    continue;
                }

                for x in self.partition_by_actions(x, &state, token) {
                    debug!("  set of states that can reach: {:?}", x);

                    if x.is_empty() {
                        continue;
                    }

                    for y in minimize.partitions.clone().into_iter() {
                        debug!("  comparing with {:?}", y);
                        let y1 = y.intersection(&x);

                        if y1.is_empty() {
                            continue;
                        }

                        let y2 = y.difference(&x);

                        if y2.is_empty() {
                            continue;
                        }

                        debug!("  match:");

                        // Refine the partition
                        assert!(minimize.partitions.remove(&y));
                        minimize.partitions.insert(y1.clone());
                        minimize.partitions.insert(y2.clone());

                        if minimize.remaining.remove(&y) {
                            debug!("    already contained, splitting");
                            minimize.remaining.insert(y1);
                            minimize.remaining.insert(y2);
                        } else {
                            if y1.len() <= y2.len() {
                                debug!("    not contained, adding intersection");
                                minimize.remaining.insert(y1);
                            } else {
                                debug!("    not contained, adding difference");
                                minimize.remaining.insert(y2);
                            }
                        }
                    }
                }
            }
        }
    }

    // Uses the computed refinements and applies them to the current DFA
    fn apply_refinement<'a>(&'a mut self, minimize: &'a mut Minimize) {
        let inc = self.next_state_id();

        // Map partitions to state IDs
        let target_states: HashMap<&'a Partition, State> = minimize.partitions.iter()
            .enumerate()
            .map(|(i, p)| (p, i as State + inc))
            .collect();

        debug!("STATES: {:?}", self.states);
        debug!("REFINEMENTS: {:?}", target_states);

        // Next step is to create a map from the original states of the DFA ->
        // new states. This is done by finding the partition that contains the
        // source state and then figuring out it's target state
        let state_map: HashMap<State, State> = self.states.iter()
            .map(|&s| {
                let partition = minimize.partitions.iter()
                    .find(|p| p.contains(s))
                    .expect("expected partitions to cover all states");

                (s, target_states[partition])
            })
            .collect();

        debug!("");
        debug!("State map:");
        for (from, to) in &state_map {
            debug!("  {} -> {}", from, to);
        }

        // Load the new state IDs
        self.states.clear();

        for state in target_states.values() {
            self.states.insert(*state);
        }

        self.transitions.remap(&state_map);

        // Update start state
        self.start = state_map[&self.start];

        let old = mem::replace(&mut self.terminal, HashMap::new());

        // Compute the new terminal states
        for partition in &minimize.partitions {
            let new_state = target_states[partition];

            for old_state in partition.iter() {
                if let Some(actions) = old.get(old_state) {
                    // The current partition is a terminal state.
                    match self.terminal.entry(new_state) {
                        Entry::Vacant(e) => {
                            e.insert(old[old_state].clone());
                        }
                        Entry::Occupied(e) => {
                            assert_eq!(&e.get()[..], &old[old_state][..]);
                        }
                    }
                }
            }
        }

        debug!("RESULT:");
        debug!("  states: {:?}", self.states);
        debug!("  start: {:?}", self.start);
        debug!("  terminal: {:?}", self.terminal);
        debug!("  transitions:");
        self.transitions.each(|t| {
            debug!("    {:?} -- ( {:?} ) --> {:?}", t.from(), t.input(), t.to());
        });
    }

    /*
     *
     * ===== NFA to DFA conversions
     *
     */

    // Convert the current (possibly NFA) automaton to a DFA.
    fn to_dfa(&mut self, ctx: &Context) {
        let mut actions: HashSet<usize> = HashSet::new();

        // Seed the start of the conversion
        let mut conv = Convert::new(self.epsilon_closure(&set![self.start], &mut actions));

        // Enter actions should be stored on on_enter
        assert!(actions.is_empty());

        while let Some(state) = conv.remaining.pop() {
            // Iterate through each possible alphabet entry
            for val in self.alphabet.iter().cloned() {
                actions.clear();

                // Find all reachable states from the current point with the
                // given input
                let reachable = self.reachable(&state, Some(&val), &mut actions);

                // Nothing more to do for this iteration if there are no
                // reachable states.
                if reachable.is_empty() {
                    assert!(actions.is_empty());
                    continue;
                }

                // Compute the epsilon closure for all reachable states, this
                // is used as the DFA state.
                let to = self.epsilon_closure(&reachable, &mut actions);

                // Create a transition from the original state to the newly
                // reachable state. If the destination state has never been
                // reached, this will also track it as unhandled so that it is
                // processed in a future loop iteration.
                conv.add_transition(&state, &to, val, actions.iter().cloned().collect());
            }
        }

        // == Update the automaton ==

        self.states.clear();
        self.states.extend(conv.states.values().cloned());

        self.start = conv.start;
        self.terminal = conv.convert_terminal_states(&self.terminal, ctx);

        self.transitions = conv.transitions;
    }

    // Return the set of states that are reachable from the given states via
    // any number of epsilon transitions
    fn epsilon_closure(&self, states: &HashSet<State>, actions: &mut HashSet<usize>) -> HashSet<State> {
        // The input states are always reachable via zero transitions, so start
        // building from that set.
        //
        // The basic strategy is a graph traversal
        let mut ret = states.clone();
        let mut rem: Vec<State> = states.iter().map(|s| *s).collect();

        // For each state, find all other states that are reachable via a
        // single epsilon transition. Add all states that have not already been
        // traversed to the list of remaining states to handle.
        while let Some(from) = rem.pop() {
            for dest in self.reachable(&from, None, actions) {
                if ret.insert(dest) {
                    rem.push(dest);
                }
            }
        }

        ret
    }

    /*
     *
     * ===== Utility =====
     *
     */

    // Return the set of states reachable from a given state when the given
    // input is applied
    fn reachable<A: StateSet>(&self, states: &A, input: Option<&Letter>, actions: &mut HashSet<usize>) -> HashSet<State> {
        let mut ret = HashSet::with_capacity(self.transitions.len());

        self.transitions.each(|transition| {
            if states.contains(transition.from()) && transition.input() == input {
                ret.insert(transition.to());
                actions.extend(transition.actions().iter().cloned());
            }
        });

        ret
    }

    // Return the set of states from which a transition on the given input will
    // lead to one of the given destination states
    fn reached<A: StateSet>(&self, dests: &A, input: Option<&Letter>) -> HashSet<State> {
        let mut ret = HashSet::with_capacity(self.transitions.len());

        self.transitions.each(|transition| {
            if dests.contains(transition.to()) && transition.input() == input {
                ret.insert(transition.from());
            }
        });

        ret
    }

    // Returns the set of states that are NOT terminal (not a member of the set
    // of terminal states)
    fn nonterminal(&self) -> HashSet<State> {
        self.states.iter()
            .filter(|s| !self.terminal.contains_key(s))
            .cloned()
            .collect()
    }

    fn partition_by_actions(&self, mut from: HashSet<State>, to: &Partition, token: &Letter) -> Vec<HashSet<State>> {
        // TODO: Optimize by doing nothing if the original set has consistent
        // actions
        let mut res: HashMap<&[usize], HashSet<State>> = HashMap::new();

        while let Some(f) = util::pop(&mut from) {
            for t in to.iter() {
                if let Some(a) = self.transitions.actions(f, *t, token) {
                    res.entry(&a).or_insert_with(|| HashSet::new())
                        .insert(f);
                }
            }
        }

        res.into_iter()
            .map(|(_, v)| v)
            .collect()
    }

    // Verifies that the DFA is sane
    fn verify(&self) {
        // Ensure that there are no epsilon transitions
        self.verify_is_dfa();

        // All transitions are on input contained by alphabet
        self.verify_alphabet();

        // Ensure that for each (State, Input) tuple there is only a single
        // possible target
        self.verify_deterministic();

        // Verify that the vec of actions to invoke are ordered by index.
        self.verify_actions_ordered();
    }

    fn verify_is_dfa(&self) {
        self.transitions.each(|transition| {
            assert!(transition.input().is_some(), "there should be no epsilon transitions at this point");
        });
    }

    fn verify_alphabet(&self) {
        self.transitions.each(|transition| {
            if let Some(token) = transition.input() {
                assert!(self.alphabet.contains(token), "transition input not contained by alphabet");
            }
        });
    }

    fn verify_deterministic(&self) {
        for token in self.alphabet.iter() {
            let mut states = set![];

            self.transitions.each(|transition| {
                if transition.input() == Some(token) {
                    assert!(states.insert(transition.from()), "invalid set of transitions");
                }
            });
        }
    }

    fn verify_actions_ordered(&self) {
        self.transitions.each(|t| {
            assert!(is_ordered(t.actions()));
        });

        for actions in self.terminal.values() {
            assert!(is_ordered(actions));
        }
    }

    // Increments all state IDs by `state_shift` and all action IDs by `action`
    //
    // Used to ensure that state IDs don't
    // overlap when combining two automata.
    fn shift(&mut self, state_shift: State, action_shift: usize) {
        let state_shift = cmp::max(self.next_state_id(), state_shift);

        let mut states = HashSet::with_capacity(self.states.len());

        for state in self.states.iter() {
            states.insert(state + state_shift);
        }

        let new = HashMap::with_capacity(self.terminal.len());
        let old = mem::replace(&mut self.terminal, new);

        for (state, mut actions) in old {
            for action in actions.iter_mut() {
                *action = *action + action_shift;
            }

            self.terminal.insert(state + state_shift, actions);
        }

        self.start += state_shift;
        self.transitions.shift(state_shift, action_shift);
        self.states = states;
    }

    // Next available state ID
    fn next_state_id(&self) -> State {
        1 + self.states.iter()
            .max()
            .map(|i| *i)
            .expect("the automaton has no states")
    }
}

/// Minimize a DFA using Hopcroft's algorithm
struct Minimize {
    remaining: HashSet<Partition>,
    partitions: HashSet<Partition>,
}

impl Minimize {
    fn new(terminal: HashSet<State>, nonterminal: HashSet<State>) -> Minimize {
        let terminal = Partition::new(terminal);

        debug!("MINIMIZE:");
        debug!("  terminal:     {:?}", terminal);
        debug!("  non-terminal: {:?}", nonterminal);

        Minimize {
            remaining: set![terminal.clone()],
            partitions: set![terminal, Partition::new(nonterminal)],
        }
    }
}

/// Convert an NFA -> DFA
struct Convert {
    states: HashMap<MultiState, State>,
    start: State,
    transitions: Transitions,
    remaining: Vec<MultiState>,
}

impl Convert {
    fn new(init: HashSet<State>) -> Convert {
        // The initial multistate
        let state = MultiState::new(&init);

        // Maps multistates (NFA states) to DFA states
        let mut states = HashMap::new();
        states.insert(state.clone(), 0);

        let ret = Convert {
            states: states,
            start: 0,
            transitions: Transitions::empty(),
            remaining: vec![state],
        };

        ret
    }

    fn add_transition(&mut self, from: &MultiState, to: &HashSet<State>, input: Letter, actions: Vec<usize>) {
        let multi = MultiState::new(to);
        let (to, first) = self.track_multistate(multi.clone());

        self.transitions.letter(self.states[from], to, input, actions);

        if first {
            self.remaining.push(multi);
        }
    }

    fn track_multistate(&mut self, multistate: MultiState) -> (State, bool) {
        let len = self.states.len();

        match self.states.entry(multistate) {
            Entry::Occupied(e) => (*e.get(), false),
            Entry::Vacant(e) => (*e.insert(len as u32), true),
        }
    }

    fn convert_terminal_states(&self, prev: &HashMap<State, Vec<usize>>, ctx: &Context) -> HashMap<State, Vec<usize>> {
        self.states.iter()
            .filter_map(|(k, v)| {
                match k.actions(prev) {
                    Some(actions) => {
                        match ctx.intersect {
                            // Currently handling an intersection. A terminal state
                            // must exist as a terminal state in both source
                            // automata.
                            Some((ref a, ref b)) => {
                                if k.is_disjoint(a) || k.is_disjoint(b) {
                                    None
                                } else {
                                    Some((*v, actions.to_vec()))
                                }
                            }
                            // Not handling an intersection, just return the new
                            // state
                            None => Some((*v, actions.to_vec()))
                        }
                    }
                    None => None,
                }
            })
            // .cloned()
            .collect()
    }
}

/// A partition of DFA states
#[derive(Eq, PartialEq, Clone, Debug)]
struct Partition {
    states: HashSet<State>,
}

impl Partition {
    fn new(states: HashSet<State>) -> Partition {
        Partition {
            states: states,
        }
    }

    fn intersection(&self, other: &HashSet<State>) -> Partition {
        Partition::new(
            self.states.intersection(other)
                .cloned()
                .collect())
    }

    fn difference(&self, other: &HashSet<State>) -> Partition {
        Partition::new(
            self.states.difference(other)
                .cloned()
                .collect())
    }

    fn is_disjoint_map<V>(&self, other: &HashMap<State, V>) -> bool {
        self.states.iter().all(|s| !other.contains_key(s))
    }
}

impl ops::Deref for Partition {
    type Target = HashSet<State>;

    fn deref(&self) -> &HashSet<State> {
        &self.states
    }
}

impl hash::Hash for Partition {
    fn hash<H>(&self, state: &mut H) where H: hash::Hasher {
        use std::hash::Hash;

        let sum = self.states.iter()
            .map(util::hash)
            .fold(0, |s, v| v.wrapping_add(s));

        self.states.len().hash(state);
        sum.hash(state);
    }
}

#[derive(Hash, Clone, Eq, PartialEq, Debug)]
struct MultiState {
    states: Vec<State>,
}

impl MultiState {
    fn new<'a, I>(states: I) -> MultiState
            where I: IntoIterator<Item=&'a State> {

        let mut states: Vec<State> = states.into_iter().map(|s| *s).collect();
        states.sort();

        MultiState {
            states: states,
        }
    }

    fn is_disjoint(&self, other: &HashSet<State>) -> bool {
        self.states.iter()
            .all(|s| !other.contains(s))
    }

    fn actions<'a>(&self, map: &'a HashMap<State, Vec<usize>>) -> Option<&'a [usize]> {
        let mut ret: Option<&'a [usize]> = None;

        for state in &self.states {
            if let Some(actions) = map.get(state) {
                match ret {
                    Some(r) => {
                        assert_eq!(&actions[..], r);
                    }
                    None => {
                        ret = Some(&actions[..]);
                    }
                }
            }
        }

        ret
    }
}

/// A value that represents multiple states
trait StateSet {
    fn contains(&self, state: State) -> bool;
}

impl StateSet for State {
    fn contains(&self, state: State) -> bool {
        *self == state
    }
}

impl StateSet for MultiState {
    fn contains(&self, state: State) -> bool {
        self.states.contains(&state)
    }
}

impl StateSet for Partition {
    fn contains(&self, state: State) -> bool {
        self.states.contains(&state)
    }
}

// Optimization context
struct Context {
    intersect: Option<(HashSet<State>, HashSet<State>)>
}

impl Context {
    fn new() -> Context {
        Context {
            intersect: None,
        }
    }
}

// TODO: Dedup from transitions
fn insert_action(dst: &mut Vec<usize>, idx: usize) {
    if dst.contains(&idx) {
        return;
    }

    dst.push(idx);
    dst.sort();
}

fn is_ordered(idxs: &[usize]) -> bool {
    let mut curr = None;

    for i in idxs {
        match curr {
            Some(j) => {
                if i < j {
                    return false;
                }

                curr = Some(i);
            }
            None => {
                curr = Some(i);
            }
        }
    }

    return true;
}
