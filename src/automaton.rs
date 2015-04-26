use {info, Token, Input};
use std::{iter, usize};
use std::marker::PhantomData;

// Max size to make a lookup table
const MAX_LOOKUP_TABLE: u32 = 255;
const INVALID: usize = usize::MAX;

pub struct Automaton<T, S> {
    states: Vec<State>,
    start: usize,
    token: PhantomData<T>,
    state: PhantomData<S>,
}

pub fn compile<T: Token, S>(info: Vec<info::State>) -> Automaton<T, S> {
    let states: Vec<State> = info.iter()
        .map(|state| {
            State {
                op: Op::Lookup(Table::build(state)),
                terminal: state.terminal,
            }
        })
        .collect();

    let start = info.iter().enumerate()
        .find(|&(_, state)| state.start)
        .map(|(i, _)| i)
        .expect("expression had no start state");

    Automaton {
        states: states,
        start: start,
        token: PhantomData,
        state: PhantomData,
    }
}

impl<T: Token, S> Automaton<T, S> {
    pub fn eval<I, J>(&self, input: I) -> bool
            where I: Iterator<Item=J>,
                  J: Input<T> {

        let mut state = self.start;

        for val in input {
            match self.states[state].op {
                Op::Lookup(ref table) => {
                    match table.lookup(val.as_u32()) {
                        Some(nxt) => {
                            state = nxt;
                        }
                        None => return false,
                    }
                }
            }
        }

        self.states[state].terminal
    }
}

struct State {
    op: Op,
    terminal: bool,
}

enum Op {
    Lookup(Table),
}

// TODO: Implement bounds directly vs. repeating bound checks in Vec as well
struct Table {
    dests: Vec<usize>,
}

impl Table {
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
}

impl Table {
    fn build(state: &info::State) -> Table {
        let upper = char_upper_bound(&state.transitions);

        // TODO: Implement better
        assert!(upper <= MAX_LOOKUP_TABLE);

        let mut table: Vec<usize> = iter::repeat(usize::MAX)
            .take(upper as usize)
            .collect();

        for t in &state.transitions {
            for i in t.on.clone() {
                table[i as usize] = t.target
            }
        }

        Table { dests: table }
    }
}

fn char_upper_bound(transitions: &[info::Transition]) -> u32 {
    transitions.iter().map(|t| t.on.end).max().unwrap_or(0)
}
