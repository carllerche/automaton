use core::{Automaton, Dfa, Letter, Token};
use std::fmt;
use std::iter::Iterator;
use std::marker::PhantomData;

pub struct Expression<T: Token, S> {
    dfa: Dfa<S>,
    phantom: PhantomData<T>,
}

impl<T: Token, S> Expression<T, S> {
    fn empty() -> Expression<T, S> {
        Expression {
            dfa: Dfa::empty(),
            phantom: PhantomData,
        }
    }

    /// Returns an automaton that accepts the given token
    pub fn token(input: T) -> Expression<T, S> {
        Expression {
            dfa: Dfa::token(Letter(input.as_range())),
            phantom: PhantomData,
        }
    }

    pub fn sequence<I: Iterator<Item=T>>(tokens: I) -> Expression<T, S> {
        Expression {
            dfa: Dfa::sequence(tokens.map(|t| Letter(t.as_range()))),
            phantom: PhantomData,
        }
    }

    /// Concatenate two automata
    pub fn concat(mut self, mut other: Expression<T, S>) -> Expression<T, S> {
        Expression {
            dfa: self.dfa.concat(other.dfa),
            phantom: PhantomData,
        }
    }

    pub fn union(mut self, mut other: Expression<T, S>) -> Expression<T, S> {
        Expression {
            dfa: self.dfa.union(other.dfa),
            phantom: PhantomData,
        }
    }

    pub fn intersection(mut self, mut other: Expression<T, S>) -> Expression<T, S> {
        Expression {
            dfa: self.dfa.intersection(other.dfa),
            phantom: PhantomData,
        }
    }

    pub fn kleene(mut self) -> Expression<T, S> {
        Expression {
            dfa: self.dfa.kleene(),
            phantom: PhantomData,
        }
    }

    pub fn optional(mut self) -> Expression<T, S> {
        self.union(Expression::empty())
    }

    /// Specify an action to be invoked when evaluation enters the machine
    pub fn on_enter<F: Fn(&mut S) + 'static>(mut self, action: F) -> Expression<T, S> {
        self.dfa.on_enter(Box::new(action));
        self
    }

    /// Specify an action to be invoked when evaluation leaves the machine
    pub fn on_exit<F: Fn(&mut S) + 'static>(mut self, action: F) -> Expression<T, S> {
        self.dfa.on_exit(Box::new(action));
        self
    }

    /// Compile the expression for evaluation
    pub fn compile(self) -> Automaton<T, S> {
        From::from(self.dfa)
    }

    /*
     *
     * ===== Inspection helpers =====
     *
     */

    pub fn alphabet(&self) -> Vec<T> {
        self.dfa.alphabet().iter()
            .map(|r| <T as Token>::from_range(r))
            .collect()
    }

    /*
     *
     * ===== DOT file generation =====
     *
     */

    pub fn dot(&self) -> String {
        format!("digraph finite_state_machine {{\n
                 rankdir=LR;size=\"8,5\"\n
                 node [shape = point];\n
                   start\n
                 node [shape = doublecircle];\n
                 {};\nnode [shape = circle];\n
                 {}
                 }}\n",
                self.dot_terminal(),
                self.dot_edges())
    }

    fn dot_terminal(&self) -> String {
        let mut ret = String::new();

        for (state, _) in self.dfa.terminal() {
            ret.push_str(&format!("  {}", state));
        }

        ret
    }

    fn dot_edges(&self) -> String {
        let mut ret = String::new();

        self.dfa.transitions().each(|t| {
            let mut actions = "";

            if !t.actions().is_empty() {
                actions = "!";
            }

            match t.input() {
                Some(i) => {
                    ret.push_str(
                        &format!(
                            "{} -> {} [ label = \"{:?}{}\" ];\n",
                            t.from(), t.to(), i.to_token::<T>(), actions));
                }
                None => {
                    ret.push_str(
                        &format!(
                            "{} -> {} [ label = \"ε{}\" ];\n",
                            t.from(), t.to(), actions));
                }
            }
        });

        ret.push_str(
            &format!(
                "start -> {} [ ];\n",
                self.dfa.start()));

        ret
    }
}

impl<T: Token, S> fmt::Debug for Expression<T, S> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "Expression {{ ... }}")
    }
}
