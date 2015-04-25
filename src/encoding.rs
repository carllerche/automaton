use super::{Automaton, Expression, Token, Input};
use std::fmt;
use std::ascii::AsciiExt;
use std::hash::{Hash, Hasher};
use std::ops::Range;

#[derive(Clone, Eq, PartialEq)]
pub struct Ascii {
    bytes: Range<u8>,
}

impl Ascii {
    pub fn of<T: Into<Ascii>>(v: T) -> Expression<Ascii> {
        Expression::token(v.into())
    }

    fn new(range: Range<u8>) -> Ascii {
        assert!(range.start < range.end && range.end <= 128, "invalid ASCII range");
        Ascii { bytes: range }
    }

    /// Returns an automaton that matches the given string. Panics if the
    /// string is not valid ASCII.
    pub fn exact(s: &str) -> Expression<Ascii> {
        Expression::sequence(
            s.chars().map(|c| {
                assert!(c.is_ascii(), "invalid ASCII character");
                c.into()
            }))
    }

    pub fn any() -> Expression<Ascii> {
        Expression::token(Ascii::new(0..128))
    }
}

impl Token for Ascii {
    fn from_range(range: &Range<u32>) -> Ascii {
        Ascii {
            bytes: Range {
                start: range.start as u8,
                end: range.end as u8,
            },
        }
    }

    fn as_range(&self) -> Range<u32> {
        Range {
            start: self.bytes.start as u32,
            end: self.bytes.end as u32,
        }
    }
}

impl Hash for Ascii {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        self.bytes.start.hash(state);
        self.bytes.end.hash(state);
    }
}

impl fmt::Debug for Ascii {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use std::str;

        fn debug(byte: &[u8]) -> String {
            if byte == [9] {
                "\\t".to_string()
            } else if byte[0] < 32 || byte == [127] {
                format!("\\{}", byte[0])
            } else {
                str::from_utf8(byte).unwrap_or("?").to_string()
            }
        }

        let start = [self.bytes.start];
        let end = [self.bytes.end];

        if self.bytes.len() == 1 {
            write!(fmt, "{}", debug(&start))
        } else {
            write!(fmt, "{}..{}", debug(&start), debug(&end))
        }
    }
}

impl Input<Ascii> for u8 {
    fn as_u32(&self) -> u32 {
        *self as u32
    }
}

impl Input<Ascii> for char {
    fn as_u32(&self) -> u32 {
        *self as u32
    }
}

impl Automaton<Ascii> {
    pub fn parse(&self, s: &str) -> bool {
        self.eval(s.chars())
    }
}

impl Into<Ascii> for Range<u8> {
    fn into(self) -> Ascii {
        Ascii::new(self)
    }
}

impl Into<Ascii> for Range<char> {
    fn into(self) -> Ascii {
        assert!(self.start.is_ascii() && self.end.is_ascii());

        Ascii::new(Range {
            start: self.start as u8,
            end: self.end as u8,
        })
    }
}

impl Into<Ascii> for u8 {
    fn into(self) -> Ascii {
        Ascii::new(Range {
            start: self,
            end: self + 1,
        })
    }
}

impl Into<Ascii> for char {
    fn into(self) -> Ascii {
        assert!(self.is_ascii(), "invalid ASCII character");
        Ascii::new(Range {
            start: self as u8,
            end: self as u8 + 1,
        })
    }
}
