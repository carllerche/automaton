use automaton::*;
use automaton::encoding::Ascii;

const STR1: &'static str = "abcdefghijkl";
const STR2: &'static str = "zxywijslyla";

#[test]
pub fn test_simple_union() {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("a")
            .union(Ascii::exact("c"))
            .compile();

    for s in ["a", "c"].iter() {
        assert!(machine.parse(s));
    }

    for s in ["ab", "abcdefghijklzxywijslyla"].iter() {
        assert!(!machine.parse(s));
    }
}

#[test]
pub fn test_larger_union() {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("a")
            .union(Ascii::exact("b"))
            .union(Ascii::exact("c"))
            .union(Ascii::exact("d"))
            .union(Ascii::exact(STR1))
            .union(Ascii::exact(STR2))
            .compile();

    for s in ["a", STR1, STR2].iter() {
        assert!(machine.parse(s));
    }

    for s in ["e", "abcdefghijklzxywijslyla"].iter() {
        assert!(!machine.parse(s));
    }
}
