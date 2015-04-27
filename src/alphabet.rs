use {Token};
use std::ops;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct Alphabet {
    tokens: HashSet<Letter>,
}

impl Alphabet {
    pub fn empty() -> Alphabet {
        Alphabet {
            tokens: set![],
        }
    }

    pub fn single(letter: Letter) -> Alphabet {
        Alphabet {
            tokens: set![letter],
        }
    }

    pub fn extend(&mut self, other: &Alphabet) {
        for token in &other.tokens {
            self.insert(token.clone());
        }
    }

    pub fn refine(&mut self) {
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

                let new = Letter::disjoint(
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
pub struct Letter(pub ops::Range<u32>);

impl Letter {
    pub fn disjoint(a: &Letter, b: &Letter) -> Option<Vec<Letter>> {
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

    pub fn contains(&self, other: &Letter) -> bool {
        self.0.start <= other.0.start && self.0.end >= other.0.end
    }

    pub fn contains_val(&self, val: u32) -> bool {
        self.start <= val && self.end > val
    }

    pub fn to_token<T: Token>(&self) -> T {
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
