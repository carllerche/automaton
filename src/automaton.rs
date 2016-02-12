// use {info, Dfa, Token, Input};
use core::{Action, Dfa, Token, Transition, Letter, Input, State};
use std::{cmp, fmt, iter, usize};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::marker::PhantomData;

// Max size to make a lookup table
const MAX_LOOKUP_TABLE: u32 = 255;
const INVALID: usize = usize::MAX;

pub struct Automaton<T, S> {
    ops: Vec<Op>,
    enter: usize,
    actions: Vec<Action<S>>,
    token: PhantomData<T>,
}

impl<T: Token, S> From<Dfa<S>> for Automaton<T, S> {
    fn from(dfa: Dfa<S>) -> Automaton<T, S> {
        assert!(!dfa.states().is_empty());

        // Map state IDs -> vec offsets
        let mut compile = Compile::new(&dfa);

        // Create a list of transitions to handle
        dfa.transitions().each(|t| {
            compile.compile_transition(&t, &dfa);
        });

        compile.compile_entry(&dfa);
        compile.compile_terminal(&dfa);
        compile.to_automaton(dfa.actions())
    }
}

impl<T: Token, S> Automaton<T, S> {
    pub fn eval<I, J>(&self, s: &mut S, input: I) -> bool
            where I: Iterator<Item=J>,
                  J: Input<T>
    {
        self.try_eval(s, input.map(Ok)).expect("can't be err")
    }

    pub fn try_eval<I, J>(&self, s: &mut S, input: I) -> Result<bool, ::std::io::CharsError>
        where I: Iterator<Item=Result<J, ::std::io::CharsError>>,
              J: Input<T>
    {
        let mut state = self.enter;

        debug!("EVAL; state={}", state);

        for val in input {
            let val = try!(val);
            debug!("  input={}", val.as_u32());

            loop {
                match self.ops[state] {
                    Op::Lookup(ref table, _) => {
                        match table.lookup(val.as_u32()) {
                            Some(dest) => {
                                debug!("  matching input; success; state={}; jump={}", state, dest);
                                state = dest;
                                break;
                            }
                            None => {
                                debug!("  matching input; state={}; FAIL", state);
                                return Ok(false);
                            }
                        }
                    }
                    Op::Jump(dst) => {
                        debug!("  state={}; jump={}", state, dst);
                        state = dst;
                    }
                    Op::Invoke(ref actions) => {
                        debug!("  invoking actions; state={}", state);
                        for &action in actions {
                            self.actions[action](s);
                        }
                        state += 1;
                    }
                    Op::Terminal => {
                        debug!("  unexpected terminal; state={}", state);
                        // Not expecting a terminal
                        return Ok(false);
                    }
                    _ => {
                        panic!("oh noes, we haz a bug: unexpected op in automaton");
                    }
                }
            }
        }

        // Handle exit
        loop {
            match self.ops[state] {
                Op::Lookup(_, term) => {
                    if term == INVALID {
                        debug!("  invalid termination state; state={}", state);
                        return Ok(false);
                    }

                    debug!("");

                    state = term;
                }
                Op::Jump(dst) => {
                    debug!("  state={}; jump={}", state, dst);
                    state = dst;
                }
                Op::Invoke(ref actions) => {
                    debug!("  invoking actions; state={}", state);
                    for &action in actions {
                        self.actions[action](s);
                    }
                    state += 1;
                }
                Op::Terminal => {
                    debug!("  terminal -- ending; state={}", state);
                    return Ok(true);
                }
                _ => {
                    panic!("oh noes, we haz a bug: unexpected op in automaton");
                }
            }
        }
    }
}

impl<T: Token, S> fmt::Debug for Automaton<T, S> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "Automaton {{\n"));

        try!(write!(fmt, "      enter: {}\n", self.enter));

        for (i, op) in self.ops.iter().enumerate() {
            match *op {
                Op::Lookup(ref table, term) => {
                    if term == INVALID {
                        try!(write!(fmt, "      {}: Lookup({:?})\n", i, table));
                    } else {
                        try!(write!(fmt, "      {}: Lookup({:?}, TERM: {})\n", i, table, term));
                    }
                }
                Op::Invoke(_) => {
                    try!(write!(fmt, "      {}: Invoke()\n", i));
                }
                Op::Jump(dst) => {
                    try!(write!(fmt, "      {}: Jump({})\n", i, dst));
                }
                Op::Terminal => {
                    try!(write!(fmt, "      {}: Terminal\n", i));
                }
                Op::Noop => {
                    try!(write!(fmt, "      {}: Noop\n", i));
                }
            }
        }

        try!(write!(fmt, "}}"));

        Ok(())
    }
}

struct Compile<S> {
    ops: Vec<Op>,
    map: HashMap<State, usize>,
    enter: Option<usize>,
    actions: Vec<Action<S>>,
}

impl<S> Compile<S> {
    fn new(dfa: &Dfa<S>) -> Compile<S> {
        Compile {
            ops: Vec::with_capacity(2 * dfa.states().len()),
            map: HashMap::with_capacity(dfa.states().len()),
            enter: None,
            actions: vec![],
        }
    }

    fn to_automaton<T: Token>(self, actions: Vec<Action<S>>) -> Automaton<T, S> {
        Automaton {
            ops: self.ops,
            enter: self.enter.expect("automaton start state not determined"),
            actions: actions,
            token: PhantomData,
        }
    }

    fn compile_transition(&mut self, t: &Transition, dfa: &Dfa<S>) {
        // Ensure there are entries for the from & to states in the state
        // to op map
        self.ensure_states_preped(&t, &dfa);

        // Offset to jump to
        let mut dest = self.map[&t.to()];

        if !t.actions().is_empty() {
            dest = self.transition_action(dest, t);
        }

        self.add_input(t.from(), dest, t.input().unwrap());
    }

    fn compile_entry(&mut self, dfa: &Dfa<S>) {
        self.enter = Some(self.map[dfa.start()]);
    }

    fn compile_terminal(&mut self, dfa: &Dfa<S>) {
        // TODO: Probably can improve this
        for (terminal, actions) in dfa.terminal() {
            let dest = self.ops.len();

            if !actions.is_empty() {
                self.ops.push(Op::invoke(actions));
            }

            self.ops.push(Op::Terminal);

            match self.ops[self.map[terminal]] {
                Op::Lookup(_, ref mut t) => {
                    *t = dest;
                }
                _ => panic!("expected state op"),
            }
        }
    }

    // Compiles an action invocation
    fn transition_action(&mut self, dest: usize, t: &Transition) -> usize {
        // Attempt to store the action OP immediately before the target state
        let mut action = dest - 1;

        if self.ops[action].is_noop() {
            self.ops[action] = Op::invoke(t.actions());
            action
        } else {
            // The slot immediately before the target is already consumed.
            // Create a new spot for it.
            self.ops.push(Op::invoke(t.actions()));
            self.ops.push(Op::jump(dest));
            self.ops.len() - 2
        }
    }

    fn add_input(&mut self, from: State, dest: usize, on: &Letter) {
        match self.ops[self.map[&from]] {
            Op::Lookup(ref mut table, _) => {
                table.update(dest, on);
            }
            _ => panic!("expected state transition op"),
        }
    }

    fn ensure_states_preped(&mut self, transition: &Transition, dfa: &Dfa<S>) {
        self.ensure_state_preped(transition.from(), dfa);
        self.ensure_state_preped(transition.to(), dfa);
    }

    fn ensure_state_preped(&mut self, state: State, dfa: &Dfa<S>) {
        if let Entry::Vacant(e) = self.map.entry(state) {
            // Reserve a state for any "enter" transitions
            self.ops.push(Op::Noop);
            // TODO: DOn't generate a table if there are no transitions from
            // the state (ie, a final state)
            self.ops.push(Op::lookup(Table::build(&state, dfa)));
            e.insert(self.ops.len() - 1);
        }
    }
}

enum Op {
    Lookup(Table, usize), // Match a character by looking it up in a table
    Jump(usize),          // Jump to a new op position
    Invoke(Vec<usize>),   // Invoke actions
    Terminal,             // Terminal position, execution done
    Noop,                 // Placeholder. Should not be encountered during execution
}

impl Op {
    fn lookup(table: Table) -> Op {
        Op::Lookup(table, INVALID)
    }

    fn invoke(actions: &[usize]) -> Op {
        Op::Invoke(actions.iter().cloned().collect())
    }

    fn jump(to: usize) -> Op {
        Op::Jump(to)
    }

    fn is_noop(&self) -> bool {
        match *self {
            Op::Noop => true,
            _ => false,
        }
    }
}

// TODO: Implement bounds directly vs. repeating bound checks in Vec as well
struct Table {
    dests: Vec<usize>,
}

impl Table {
    #[inline]
    fn lookup(&self, input: u32) -> Option<usize> {
        let input = input as usize;

        if input >= self.dests.len() {
            return None;
        }

        let ret = self.dests[input];

        if ret == INVALID {
            return None;
        }

        Some(ret)
    }

    fn update(&mut self, dest: usize, input: &Letter) {
        for i in (&**input).clone() {
            self.dests[i as usize] = dest;
        }
    }
}

impl fmt::Debug for Table {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut one = false;

        for (i, &dest) in self.dests.iter().enumerate() {
            if dest != INVALID {
                if !one {
                    one = true;
                } else {
                    try!(write!(fmt, ", "));
                }

                try!(write!(fmt, "{}: {}", i, dest));
            }
        }

        Ok(())
    }
}

impl Table {
    fn build<S>(state: &State, dfa: &Dfa<S>) -> Table {
        let mut upper = 0;

        dfa.transitions().each_from(*state, |transition| {
            let letter = transition.input().unwrap();
            upper = cmp::max(upper, letter.end);
        });

        // TODO: Have assert once terminal states don't have associated tables
        // assert!(upper > 0 && upper <= MAX_LOOKUP_TABLE, "invalid input upper bound {}", upper);

        let mut table: Vec<usize> = iter::repeat(usize::MAX)
            .take(upper as usize)
            .collect();

        Table { dests: table }
    }
}
