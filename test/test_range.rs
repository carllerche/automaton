use automaton::*;
use automaton::encoding::Ascii;

#[test]
pub fn test_basic_range() {
    let machine: Automaton<Ascii, ()> =
        Ascii::of('a'..'e')
            .compile();

    for s in ["a", "b", "c", "d"].iter() {
        assert!(machine.parse(&mut (), s), "failed to parse `{}`", s);
    }

    for s in ["e", "aa", "", "ab", "f", "fa"].iter() {
        assert!(!machine.parse(&mut (), s), "parsed `{}`", s);
    }
}

#[test]
pub fn test_union_distinct_ranges() {
    let machine: Automaton<Ascii, ()> =
        Ascii::of('a'..'d')
            .union(Ascii::of('f'..'h'))
            .compile();

    for s in ["a", "b", "c", "f", "g"].iter() {
        assert!(machine.parse(&mut (), s), "failed to parse `{}`", s);
    }

    for s in ["d", "e", "h"].iter() {
        assert!(!machine.parse(&mut (), s));
    }
}

#[test]
pub fn test_union_sequential_ranges() {
    let machine: Automaton<Ascii, ()> =
        Ascii::of('a'..'c')
            .union(Ascii::of('c'..'e'))
            .compile();

    for s in ["a", "b", "c", "d"].iter() {
        assert!(machine.parse(&mut (), s), "failed to parse `{}`", s);
    }

    assert!(!machine.parse(&mut (), "e"));
}

#[test]
pub fn test_union_subset_ranges() {
    let machine: Automaton<Ascii, ()> =
        Ascii::of('a'..'d')
            .union(Ascii::of('a'..'c'))
            .compile();

    for s in ["a", "b", "c"].iter() {
        assert!(machine.parse(&mut (), s), "failed to parse `{}`", s);
    }
}

#[test]
pub fn test_union_overlapping_ranges() {
    let machine: Automaton<Ascii, ()> =
        Ascii::of('a'..'e')
            .union(Ascii::of('c'..'h'))
            .compile();

    for s in ["a", "b", "c", "d", "e", "f", "g"].iter() {
        assert!(machine.parse(&mut (), s), "failed to parse `{}`", s);
    }

    for s in ["h", "ab", "ia"].iter() {
        assert!(!machine.parse(&mut (), s));
    }
}
