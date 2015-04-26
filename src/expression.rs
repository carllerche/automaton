use {automaton, info, util, Action, Automaton, Token};
use std::{cmp, fmt, ops};
use std::hash::{self, Hash, Hasher};
use std::collections::{HashSet, HashMap};
use std::collections::hash_map::Entry;
use std::iter::{Iterator, IntoIterator};
use std::marker::PhantomData;

pub struct Expression<T: Token, S> {
    states: HashSet<State>,     // Set of states
    alphabet: Alphabet,         // The alphabet (valid input)
    transitions: Transitions,   // Set of transitions
    start: State,               // Initial state
    terminal: HashSet<State>,   // Set of terminal states
    actions: Vec<Action<S>>,    // Actions
    phantom: PhantomData<T>,
}

pub type State = u32;

impl<T: Token, S> Expression<T, S> {
    /// Returns an automaton that accepts the given token
    pub fn token(input: T) -> Expression<T, S> {
        let letter = Letter(input.as_range());

        Expression {
            states: set![0, 1],
            alphabet: Alphabet::single(letter.clone()),
            transitions: Transitions::of(0, 1, letter),
            start: 0,
            terminal: set![1],
            actions: vec![],
            phantom: PhantomData,
        }
    }

    pub fn sequence<I: Iterator<Item=T>>(tokens: I) -> Expression<T, S> {
        let start = 0;
        let mut states = set![start];
        let mut alphabet = Alphabet::empty();
        let mut transitions = Transitions::empty();
        let mut prev = start;

        for (i, token) in tokens.enumerate() {
            let state = i as State + 1;

            let letter = Letter(token.as_range());

            states.insert(state);
            alphabet.insert(letter.clone());
            transitions.letter(prev, state, letter, vec![]);
            prev = state;
        }

        let mut ret = Expression {
            states: states,
            alphabet: alphabet,
            transitions: transitions,
            start: start,
            terminal: set![prev],
            actions: vec![],
            phantom: PhantomData,
        };

        // Optimize the automaton
        ret.optimize(&Context::new());

        ret
    }

    /// Concatenate two automata
    pub fn concat(mut self, mut other: Expression<T, S>) -> Expression<T, S> {
        // Ensure states don't overlap
        other.shift(self.next_state_id());

        // Add epsilon transitions from all final states to start of `other`
        for state in &self.terminal {
            self.transitions.insert(*state, other.start, None, vec![]);
        }

        // Add states from `other`
        self.states.extend(other.states);

        // Extend the alphabet
        self.alphabet.extend(&other.alphabet);

        // Add transitions from `other`
        self.transitions.extend(other.transitions);

        // Set terminal states to `other`
        self.terminal = other.terminal;

        // Optimize the representation
        self.optimize(&Context::new());

        self
    }

    pub fn union(mut self, mut other: Expression<T, S>) -> Expression<T, S> {
        self.union_or_intersection(other, false)
    }

    pub fn intersection(mut self, mut other: Expression<T, S>) -> Expression<T, S> {
        self.union_or_intersection(other, true)
    }

    // Performs either a union or an intersection, depending on the flag
    fn union_or_intersection(mut self, mut other: Expression<T, S>, intersection: bool) -> Expression<T, S> {
        // Ensure states don't overlap
        other.shift(self.next_state_id());

        // Optimization context
        let mut ctx = Context::new();

        if intersection {
            ctx.intersect = Some((self.terminal.clone(), other.terminal.clone()));
        }

        // Add the states from other
        self.states.extend(other.states);

        // Extend the alphabet
        self.alphabet.extend(&other.alphabet);

        // Merge transitions
        self.transitions.extend(other.transitions);

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

        // Optimize the automaton, converting to a DFA and merging all similar
        // states
        self.optimize(&ctx);

        self
    }

    pub fn kleene(mut self) -> Expression<T, S> {
        let start = self.next_state_id();
        self.states.insert(start);

        let noop = self.next_state_id();
        self.states.insert(noop);

        self.transitions.epsilon(start, self.start, vec![]);
        self.transitions.epsilon(start, noop, vec![]);

        for s in &self.terminal {
            self.transitions.epsilon(*s, start, vec![]);
        }

        self.start = start;
        self.terminal.insert(noop);

        self.optimize(&Context::new());

        self
    }

    pub fn compile(self) -> Automaton<T, S> {
        self.into()
    }

    /*
     *
     * ===== Inspection helpers =====
     *
     */

    pub fn alphabet(&self) -> Vec<T> {
        self.alphabet.iter()
            .map(|r| <T as Token>::from_range(r))
            .collect()
    }

    // Optimize the representation of the automaton
    fn optimize(&mut self, ctx: &Context) {
        // Ensure the alphabet tokens are disjoint
        self.refine_alphabet();

        // Convert to a DFA
        self.to_dfa(ctx);

        // Prune dead states
        self.prune();

        // Minimize the state machine
        self.minimize();

        // Verify that things are as expected
        self.verify();
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
            let token = match transition.input {
                Some(ref token) => token,
                None => return,
            };

            // Quick path
            if self.alphabet.contains(&token) {
                return;
            }

            for other in self.alphabet.iter() {
                if contains(token, &other) {
                    additions.push((transition.from, transition.to, other.clone(), transition.actions.to_vec()));
                }
            }
        });

        for (from, to, input, actions) in additions {
            self.transitions.insert(from, to, Some(input), actions);
        }

        let alphabet = &self.alphabet;

        self.transitions.retain(|t| {
            match t.input.as_ref() {
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
        // Start by setting the states to the set of terminal states
        self.states.clear();
        self.states.extend(self.terminal.iter().cloned());

        let mut remaining = self.terminal.clone();

        while !remaining.is_empty() {
            let mut new: HashSet<State> = HashSet::with_capacity(self.transitions.len());

            self.transitions.each(|transition| {
                if remaining.contains(&transition.to) {
                    new.insert(transition.from);
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
            states.contains(&transition.from) &&
                states.contains(&transition.to)
        });
    }

    fn minimize(&mut self) {
        let mut minimize = Minimize::new(self.terminal.clone(), self.nonterminal());

        // Step 1) Refine the partitions
        self.refine(&mut minimize);
    }

    // Refine the partitions. This is done by removing an (any) partition from
    // the set of remaining partitions. The reason why a set is used is to be
    // able to perform set ops on the remaining partitions.
    fn refine(&mut self, minimize: &mut Minimize) {
        while let Some(state) = util::pop(&mut minimize.remaining) {
            for token in self.alphabet.iter() {
                let x = self.reached(&state, Some(&token));

                for y in minimize.partitions.clone().into_iter() {
                    let y1 = y.intersection(&x);

                    if y1.is_empty() {
                        continue;
                    }

                    let y2 = y.difference(&x);

                    if y2.is_empty() {
                        continue;
                    }

                    // Refine the partition
                    assert!(minimize.partitions.remove(&y));

                    if minimize.remaining.remove(&y) {
                        minimize.remaining.insert(y1);
                        minimize.remaining.insert(y2);
                    } else {
                        if y1.len() < y2.len() {
                            minimize.remaining.insert(y1);
                        } else {
                            minimize.remaining.insert(y2);
                        }
                    }
                }
            }
        }
    }

    // Uses the computed refinements and applies them to the current DFA
    fn apply_refinement<'a>(&'a mut self, minimize: &'a mut Minimize) {
        // Map partitions to state IDs
        let target_states: HashMap<&'a Partition, State> = minimize.partitions.iter()
            .enumerate()
            .map(|(i, p)| (p, i as State))
            .collect();

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

        // Load the new state IDs
        self.states.clear();

        for state in target_states.values() {
            self.states.insert(*state);
        }

        self.transitions.remap(&state_map);

        // Update start state
        self.start = state_map[&self.start];

        // Update terminal states
        self.terminal = minimize.partitions.iter()
            .filter(|p| !p.is_disjoint(&self.terminal))
            .map(|p| target_states[p])
            .collect();
    }

    /*
     *
     * ===== NFA to DFA conversions
     *
     */

    // Convert the current (possibly NFA) automaton to a DFA.
    fn to_dfa(&mut self, ctx: &Context) {
        // Seed the start of the conversion
        let mut conv = Convert::new(self.epsilon_closure(&set![self.start]));

        while let Some(state) = conv.remaining.pop() {
            // Iterate through each possible alphabet entry
            for val in self.alphabet.iter().cloned() {
                // Find all reachable states from the current point with the
                // given input
                let reachable = self.reachable(&state, Some(&val));

                // Nothing more to do for this iteration if there are no
                // reachable states.
                if reachable.is_empty() {
                    continue;
                }

                // Compute the epsilon closure for all reachable states, this
                // is used as the DFA state.
                let to = self.epsilon_closure(&reachable);

                // Create a transition from the original state to the newly
                // reachable state. If the destination state has never been
                // reached, this will also track it as unhandled so that it is
                // processed in a future loop iteration.
                conv.add_transition(&state, &to, val);
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
    fn epsilon_closure(&self, states: &HashSet<State>) -> HashSet<State> {
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
            for dest in self.reachable(&from, None) {
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
    fn reachable<A: StateSet>(&self, states: &A, input: Option<&Letter>) -> HashSet<State> {
        let mut ret = HashSet::with_capacity(self.transitions.len());

        self.transitions.each(|transition| {
            if states.contains(transition.from) && transition.input.as_ref() == input {
                ret.insert(transition.to);
            }
        });

        ret
    }

    // Return the set of states from which a transition on the given input will
    // lead to one of the given destination states
    fn reached<A: StateSet>(&self, dests: &A, input: Option<&Letter>) -> HashSet<State> {
        let mut ret = HashSet::with_capacity(self.transitions.len());

        self.transitions.each(|transition| {
            if dests.contains(transition.to) && transition.input.as_ref() == input {
                ret.insert(transition.from);
            }
        });

        ret
    }

    // Returns the set of states that are NOT terminal (not a member of the set
    // of terminal states)
    fn nonterminal(&self) -> HashSet<State> {
        self.states.difference(&self.terminal).map(|s| *s).collect()
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
    }

    fn verify_is_dfa(&self) {
        self.transitions.each(|transition| {
            assert!(transition.input.is_some(), "there should be no epsilon transitions at this point");
        });
    }

    fn verify_alphabet(&self) {
        self.transitions.each(|transition| {
            if let Some(token) = transition.input.as_ref() {
                assert!(self.alphabet.contains(token), "transition input not contained by alphabet");
            }
        });
    }

    fn verify_deterministic(&self) {
        for token in self.alphabet.iter() {
            let mut states = set![];

            self.transitions.each(|transition| {
                if transition.input.as_ref() == Some(token) {
                    assert!(states.insert(transition.from), "invalid set of transitions");
                }
            });
        }
    }

    // Increments all state IDs by `inc`. Used to ensure that state IDs don't
    // overlap when combining two automata.
    fn shift(&mut self, inc: State) {
        let inc = cmp::max(self.next_state_id(), inc);

        let mut states = HashSet::with_capacity(self.states.len());
        let mut terminal = HashSet::with_capacity(self.terminal.len());

        for state in self.states.iter() {
            states.insert(state + inc);
        }

        for state in self.terminal.iter() {
            terminal.insert(state + inc);
        }

        self.start += inc;
        self.transitions.shift(inc);
        self.states = states;
        self.terminal = terminal
    }

    // Next available state ID
    fn next_state_id(&self) -> State {
        1 + self.states.iter()
            .max()
            .map(|i| *i)
            .expect("the automaton has no states")
    }

    /*
     *
     * ===== DOT file generation =====
     *
     */

    pub fn dot(&self) -> String {
        format!("digraph finite_state_machine {{\n
                 rankdir=LR;size=\"8,5\"\n
                 node [shape = doublecircle];\n
                 {};\nnode [shape = circle];\n
                 {}
                 }}\n",
                self.dot_terminal(),
                self.dot_edges())
    }

    fn dot_terminal(&self) -> String {
        let mut ret = String::new();

        for state in &self.terminal {
            ret.push_str(&format!("  {}", state));
        }

        ret
    }

    fn dot_edges(&self) -> String {
        let mut ret = String::new();

        self.transitions.each(|t| {
            match t.input.as_ref() {
                Some(i) => {
                    ret.push_str(
                        &format!(
                            "{} -> {} [ label = \"{:?}\" ];\n",
                            t.from, t.to, i.to_token::<T>()));
                }
                None => {
                    ret.push_str(
                        &format!(
                            "{} -> {} [ label = \"ε\" ];\n",
                            t.from, t.to));
                }
            }
        });

        ret
    }
}

impl<T: Token, S> Into<Automaton<T, S>> for Expression<T, S> {
    fn into(self) -> Automaton<T, S> {
        let mut states = vec![];
        let mut map: HashMap<State, usize> = HashMap::with_capacity(self.states.len());

        for state in self.states {
            let idx = states.len();

            states.push(info::State {
                start: self.start == state,
                terminal: self.terminal.contains(&state),
                transitions: vec![],
            });

            map.insert(state, idx);
        }

        self.transitions.each(|transition| {
            states[map[&transition.from]].transitions.push(info::Transition {
                on: transition.input.unwrap().0,
                target: map[&transition.to],
            });
        });

        automaton::compile(states)
    }
}

impl<T: Token, S> fmt::Debug for Expression<T, S> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "Expression {{ ... }}")
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

    fn add_transition(&mut self, from: &MultiState, to: &HashSet<State>, input: Letter) {
        let multi = MultiState::new(to);
        let (to, first) = self.track_multistate(multi.clone());

        self.transitions.letter(self.states[from], to, input, vec![]);

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

    fn convert_terminal_states(&self, prev: &HashSet<State>, ctx: &Context) -> HashSet<State> {
        self.states.iter()
            .filter_map(|(k, v)| {
                if !k.is_disjoint(prev) {
                    match ctx.intersect {
                        // Currently handling an intersection. A terminal state
                        // must exist as a terminal state in both source
                        // automata.
                        Some((ref a, ref b)) => {
                            if k.is_disjoint(a) || k.is_disjoint(b) {
                                None
                            } else {
                                Some(v)
                            }
                        }
                        // Not handling an intersection, just return the new
                        // state
                        None => Some(v)
                    }
                } else {
                    None
                }
            })
            .cloned()
            .collect()
    }
}

#[derive(Clone, Debug)]
struct Transition<'a> {
    from: State,
    to: State,
    input: Option<Letter>, // None represents epsilon transition
    actions: &'a [usize],  // Offset to a transition
}

impl<'a> Transition<'a> {
    fn new(from: State, to: State, input: Letter, actions: &'a [usize]) -> Transition<'a> {
        Transition {
            from: from,
            to: to,
            input: Some(input),
            actions: actions,
        }
    }

    fn epsilon(from: State, to: State, actions: &'a [usize]) -> Transition<'a> {
        Transition {
            from: from,
            to: to,
            input: None,
            actions: actions,
        }
    }
}


#[derive(Clone, Debug)]
struct Alphabet {
    tokens: HashSet<Letter>,
}

impl Alphabet {
    fn empty() -> Alphabet {
        Alphabet {
            tokens: set![],
        }
    }

    fn single(letter: Letter) -> Alphabet {
        Alphabet {
            tokens: set![letter],
        }
    }

    fn extend(&mut self, other: &Alphabet) {
        for token in &other.tokens {
            self.insert(token.clone());
        }
    }

    fn refine(&mut self) {
        let mut tokens: Vec<Option<Letter>> = self.tokens.iter()
            .cloned()
            .map(|t| Some(t))
            .collect();

        let mut i = 0;

        while i < tokens.len() {
            if tokens[i].is_none() {
                i += 1;
                continue;
            }

            for j in i+1..tokens.len() {
                if tokens[i].is_none() || tokens[j].is_none() {
                    continue;
                }

                let new = disjoint(
                    tokens[i].as_ref().expect("token[i] is none"),
                    tokens[j].as_ref().expect("token[j] is none"));

                if let Some(new) = new {
                    tokens[i].take();
                    tokens[j].take();

                    tokens.extend(new.into_iter().map(|t| Some(t)));
                }
            }

            i += 1;
        }

        self.tokens.clear();
        self.tokens.extend(tokens.into_iter().filter_map(|t| t));
    }
}

impl ops::Deref for Alphabet {
    type Target = HashSet<Letter>;

    fn deref(&self) -> &HashSet<Letter> {
        &self.tokens
    }
}

impl ops::DerefMut for Alphabet {
    fn deref_mut(&mut self) -> &mut HashSet<Letter> {
        &mut self.tokens
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Letter(ops::Range<u32>);

impl Letter {
    fn contains(&self, val: u32) -> bool {
        self.start <= val && self.end > val
    }

    fn to_token<T: Token>(&self) -> T {
        <T as Token>::from_range(&self.0)
    }
}

impl ops::Deref for Letter {
    type Target = ops::Range<u32>;

    fn deref(&self) -> &ops::Range<u32> {
        &self.0
    }
}

impl Hash for Letter {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        self.start.hash(state);
        self.end.hash(state);
    }
}

#[derive(Debug, Clone)]
struct Transitions {
    // source state -> target state -> input -> actions
    transitions: HashMap<State, HashMap<State, HashMap<Option<Letter>, Vec<usize>>>>,
    len: usize,
}

impl Transitions {
    fn empty() -> Transitions {
        Transitions {
            transitions: HashMap::new(),
            len: 0,
        }
    }

    fn of(from: State, to: State, letter: Letter) -> Transitions {
        let mut ret = Transitions::empty();
        ret.letter(from, to, letter, vec![]);
        ret
    }

    fn epsilon(&mut self, from: State, to: State, actions: Vec<usize>) {
        self.insert(from, to, None, actions);
    }

    fn letter(&mut self, from: State, to: State, letter: Letter, actions: Vec<usize>) {
        self.insert(from, to, Some(letter), actions);
    }

    fn insert(&mut self, from: State, to: State, letter: Option<Letter>, actions: Vec<usize>) {
        let tokens = self.transitions
            .entry(from).or_insert_with(|| HashMap::new())
            .entry(to).or_insert_with(|| HashMap::new());

        match tokens.entry(letter) {
            Entry::Occupied(mut e) => {
                e.get_mut().extend(actions);
            }
            Entry::Vacant(e) => {
                self.len += 1;
                e.insert(actions);
            }
        }
    }

    fn extend(&mut self, other: Transitions) {
        let mut inserted = 0;

        for (from, dests) in other.transitions {
            match self.transitions.entry(from) {
                // If the origin state does not exist, just move the data into
                // it, otherwise, do a merge
                Entry::Occupied(mut e) => {
                    if e.get().is_empty() {
                        inserted += dests.values()
                            .map(|v| v.len())
                            .fold(0, |s, n| s + n);

                        e.insert(dests);

                        continue;
                    }

                    for (to, tokens) in dests {
                        match e.get_mut().entry(to) {
                            Entry::Occupied(mut e) => {
                                if e.get().is_empty() {
                                    inserted += tokens.len();
                                    e.insert(tokens);
                                    continue;
                                }

                                for (token, actions) in tokens {
                                    match e.get_mut().entry(token) {
                                        Entry::Occupied(mut e) => {
                                            e.get_mut().extend(actions);
                                        }
                                        Entry::Vacant(e) => {
                                            inserted += 1;
                                            e.insert(actions);
                                        }
                                    }
                                }
                            }
                            Entry::Vacant(e) => {
                                inserted += tokens.len();
                                e.insert(tokens);
                            }
                        }
                    }
                }
                Entry::Vacant(e) => {
                    inserted += dests.values()
                        .map(|v| v.len())
                        .fold(0, |s, n| s + n);

                    e.insert(dests);
                }
            }
        }

        self.len += inserted;
    }

    fn shift(&mut self, count: State) {
        let mut to_states = vec![];
        let from_states: Vec<State> =
            self.transitions.keys().cloned().collect();

        for from in from_states {
            assert!(from < count, "invalid state shift");

            let mut dests = self.transitions.remove(&from).unwrap();

            to_states.clear();
            to_states.extend(dests.keys().cloned());

            for &to in &to_states {
                let tokens = dests.remove(&to).unwrap();
                dests.insert(to + count, tokens);
            }

            self.transitions.insert(from + count, dests);
        }
    }

    fn retain<F>(&mut self, predicate: F) where F: Fn(&Transition) -> bool {
        for (from, dests) in self.transitions.iter_mut() {
            for (to, tokens) in dests.iter_mut() {
                let mut t: Vec<Option<Letter>> = tokens.keys().cloned().collect();

                t.retain(|token| {
                    let actions = &tokens[token];

                    !predicate(&Transition {
                        from: *from,
                        to: *to,
                        input: token.clone(),
                        actions: actions,
                    })
                });

                for token in t {
                    tokens.remove(&token);
                }
            }
        }
    }

    fn remap(&mut self, map: &HashMap<State, State>) {
        for (orig, new) in map {
            if let Some(mut dests) = self.transitions.remove(&orig) {
                for (orig, new) in map {
                    if let Some(tokens) = dests.remove(&orig) {
                        dests.insert(*new, tokens);
                    }
                }

                self.transitions.insert(*new, dests);
            }
        }
    }

    fn destination(&self, from: State, input: u32) -> Option<State> {
        for (f, d) in &self.transitions {
            if *f == from {
                for (d, i) in d {
                    if i.keys().any(|i| i.as_ref().unwrap().contains(input)) {
                        return Some(*d);
                    }
                }
            }
        }

        None
    }

    fn len(&self) -> usize {
        self.len
    }

    fn each<'a, F>(&'a self, mut action: F) where F: FnMut(Transition<'a>) {
        for (from, dests) in &self.transitions {
            for (to, tokens) in dests {
                for (token, actions) in tokens {
                    action(Transition {
                        from: *from,
                        to: *to,
                        input: token.clone(),
                        actions: actions,
                    });
                }
            }
        }
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



fn disjoint(a: &Letter, b: &Letter) -> Option<Vec<Letter>> {
    // If the tokens don't overlap, nothing more to do
    if a.0.end <= b.0.start || b.0.end <= a.0.start {
        return None;
    }

    let mut points = [
        a.0.start,
        a.0.end,
        b.0.start,
        b.0.end,
    ];

    points.sort();

    fn push(dst: &mut Vec<Letter>, range: ops::Range<u32>) {
        if range.end > range.start {
            dst.push(Letter(range));
        }
    }

    let mut ret = Vec::with_capacity(3);

    push(&mut ret, ops::Range { start: points[0], end: points[1] });
    push(&mut ret, ops::Range { start: points[1], end: points[2] });
    push(&mut ret, ops::Range { start: points[2], end: points[3] });

    Some(ret)
}

fn contains(a: &Letter, b: &Letter) -> bool {
    a.0.start <= b.0.start && a.0.end >= b.0.end
}
