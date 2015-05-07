use core::{Letter, State};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;

#[derive(Clone, Debug)]
pub struct Transition<'a> {
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

    pub fn from(&self) -> State {
        self.from
    }

    pub fn to(&self) -> State {
        self.to
    }

    pub fn input(&self) -> Option<&Letter> {
        self.input.as_ref()
    }

    pub fn actions(&self) -> &[usize] {
        self.actions
    }
}


#[derive(Debug, Clone)]
pub struct Transitions {
    // source state -> target state -> input -> actions
    transitions: HashMap<State, HashMap<State, HashMap<Option<Letter>, Vec<usize>>>>,
    len: usize,
}

impl Transitions {
    pub fn empty() -> Transitions {
        Transitions {
            transitions: HashMap::new(),
            len: 0,
        }
    }

    pub fn of(from: State, to: State, letter: Letter) -> Transitions {
        let mut ret = Transitions::empty();
        ret.letter(from, to, letter, vec![]);
        ret
    }

    pub fn epsilon(&mut self, from: State, to: State, actions: Vec<usize>) {
        self.insert(from, to, None, actions);
    }

    pub fn letter(&mut self, from: State, to: State, letter: Letter, actions: Vec<usize>) {
        self.insert(from, to, Some(letter), actions);
    }

    pub fn insert(&mut self, from: State, to: State, letter: Option<Letter>, actions: Vec<usize>) {
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

    pub fn extend(&mut self, other: Transitions) {
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

    pub fn shift(&mut self, state_shift: State, action_shift: usize) {
        let mut to_states = vec![];
        let from_states: Vec<State> =
            self.transitions.keys().cloned().collect();

        for from in from_states {
            assert!(from < state_shift, "invalid state shift");

            let mut dests = self.transitions.remove(&from).unwrap();

            to_states.clear();
            to_states.extend(dests.keys().cloned());

            for &to in &to_states {
                let mut tokens = dests.remove(&to).unwrap();

                for (_, mut actions) in tokens.iter_mut() {
                    for action in actions.iter_mut() {
                        *action = *action + action_shift;
                    }
                }

                dests.insert(to + state_shift, tokens);
            }

            self.transitions.insert(from + state_shift, dests);
        }
    }

    pub fn retain<F>(&mut self, predicate: F) where F: Fn(&Transition) -> bool {
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

    // Updates any transitions that ended on `orig` to end on `new`. Returns
    // true if any transitions were modified
    pub fn remap_dest(&mut self, orig: State, new: State) -> bool {
        let mut any = false;

        for (_, dests) in self.transitions.iter_mut() {
            if let Some(tokens) = dests.remove(&orig) {
                dests.insert(new, tokens);
                any = true;
            }
        }

        any
    }

    pub fn remap(&mut self, map: &HashMap<State, State>) {
        debug!("Remapping transitions");
        debug!("  map: {:?}", map);
        debug!("  transitions: {:?}", self.transitions);

        let mut uniq = HashSet::new();

        for (from_orig, from_new) in map {
            if let Some(mut dests) = self.transitions.remove(&from_orig) {
                for (to_orig, to_new) in map {
                    if let Some(tokens) = dests.remove(&to_orig) {
                        debug!("    to {} -> {}", to_orig, to_new);
                        match dests.entry(*to_new) {
                            Entry::Vacant(e) => {
                                e.insert(tokens);
                            }
                            Entry::Occupied(mut e) => {
                                merge_tokens(e.get_mut(), tokens, &mut uniq);
                            }
                        }
                    }
                }

                match self.transitions.entry(*from_new) {
                    Entry::Vacant(e) => {
                        e.insert(dests);
                    }
                    Entry::Occupied(mut e) => {
                        for (k, v) in dests {
                            match e.get_mut().entry(k) {
                                Entry::Vacant(mut e) => {
                                    e.insert(v);
                                }
                                Entry::Occupied(mut e) => {
                                    merge_tokens(e.get_mut(), v, &mut uniq);
                                }
                            }
                        }
                    }
                }

                debug!("    from {} -> {}", from_orig, from_new);
            }
        }
        debug!("\nAFTER:");
        debug!("  transitions: {:?}", self.transitions);
    }

    pub fn dup_from(&mut self, orig: State, new: State) {
        let dests = self.transitions[&orig].clone();
        assert!(self.transitions.insert(new, dests).is_none());
    }

    pub fn destination(&self, from: State, input: u32) -> Option<State> {
        for (f, d) in &self.transitions {
            if *f == from {
                for (d, i) in d {
                    if i.keys().any(|i| i.as_ref().unwrap().contains_val(input)) {
                        return Some(*d);
                    }
                }
            }
        }

        None
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn each<'a, F>(&'a self, mut action: F) where F: FnMut(Transition<'a>) {
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

    pub fn each_from<'a, F>(&'a self, from: State, mut action: F) where F: FnMut(Transition<'a>) {
        if let Some(dests) = self.transitions.get(&from) {
            for (to, tokens) in dests {
                for (token, actions) in tokens {
                    action(Transition {
                        from: from,
                        to: *to,
                        input: token.clone(),
                        actions: actions,
                    });
                }
            }
        }
    }

    pub fn actions<'a>(&'a self, from: State, to: State, letter: &Letter) -> Option<&'a [usize]> {
        let letter = Some(letter.clone());
        if let Some(dests) = self.transitions.get(&from) {
            if let Some(tokens) = dests.get(&to) {
                if let Some(actions) = tokens.get(&letter) {
                    return Some(actions);
                }
            }
        }

        None
    }

    pub fn embed_actions_from(&mut self, from: State, src: &[usize]) {
        if let Some(dests) = self.transitions.get_mut(&from) {
            for (_, tokens) in dests.iter_mut() {
                for (_, dest) in tokens.iter_mut() {
                    for action in src {
                        insert_action(dest, *action);
                    }
                }
            }
        }
    }
}

// TODO: Dedup from dfa
fn insert_action(dst: &mut Vec<usize>, idx: usize) {
    if dst.contains(&idx) {
        return;
    }

    dst.push(idx);
    dst.sort();
}

// Merge actions for a specific transition
fn merge_actions(mut e: Entry<Option<Letter>, Vec<usize>>,
                 actions: Vec<usize>,
                 uniq: &mut HashSet<usize>) {
    match e {
        Entry::Vacant(mut e) => {
            e.insert(actions);
        }
        Entry::Occupied(mut e) => {
            uniq.clear();
            uniq.extend(e.get().iter().cloned());
            uniq.extend(actions);
            e.get_mut().clear();
            e.get_mut().extend(uniq.iter().cloned());
        }
    }
}

// Merge transitions from a given state to another given state
fn merge_tokens(a: &mut HashMap<Option<Letter>, Vec<usize>>,
                b: HashMap<Option<Letter>, Vec<usize>>,
                uniq: &mut HashSet<usize>) {

    // Loop over all the token / action pairs and merge them into `a`
    for (token, actions) in b {
        merge_actions(a.entry(token), actions, uniq);
    }
}
