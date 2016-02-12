#![allow(dead_code, unused_variables, unused_mut, unreachable_code)]
#![deny(warnings)]

#[macro_use]
extern crate log;

macro_rules! set {
    () => ({ HashSet::new() });
    ($($elem:expr),*) => ({
        let mut s = HashSet::new();
        $(s.insert($elem);)*
        s
    })
}

mod alphabet;
mod automaton;
mod dfa;
mod expression;
mod transition;
mod util;

pub mod encoding;
pub use automaton::Automaton;
pub use expression::Expression;

use std::fmt;
use std::ops::Range;

/// A set of input values for a state machine
pub trait Token : fmt::Debug {
    /// Convert the range to a token
    fn from_range(range: &Range<u32>) -> Self;

    /// Return a representation of the token as a range
    fn as_range(&self) -> Range<u32>;
}

pub trait Input<T: Token> {
    fn as_u32(&self) -> u32;
}

mod core {
    pub use {Input, Token};
    pub use alphabet::{Alphabet, Letter};
    pub use automaton::Automaton;
    pub use dfa::Dfa;
    pub use transition::{Transition, Transitions};

    /// A state in an NFA or DFA
    pub type State = u32;

    /// A user defined action
    pub type Action<S> = Box<Fn(&mut S)>;
}

/// Used for compiling an expression
mod info {
    use std::ops;

    pub struct State {
        pub start: bool,
        pub terminal: bool,
        pub transitions: Vec<Transition>,
    }

    pub struct Transition {
        pub on: ops::Range<u32>,
        pub target: usize,
    }
}
