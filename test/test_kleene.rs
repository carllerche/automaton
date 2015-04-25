use automaton::encoding::Ascii;

#[test]
pub fn test_simple_kleene_star() {
    let machine = Ascii::exact("ab")
        .kleene()
        .compile();

    for s in ["", "ab", "abab", "ababab"].iter() {
        assert!(machine.parse(s), "failed to parse `{}`", s);
    }

    for s in ["a", "c", "ac", "aba", "abc", "ababc"].iter() {
        assert!(!machine.parse(s), "parsed `{}`", s);
    }
}

#[test]
pub fn test_simple_non_determinism() {
    let machine = Ascii::exact("ab")
        .kleene()
        .concat(Ascii::exact("ac"))
        .compile();

    for s in ["ac", "abac", "ababac"].iter() {
        assert!(machine.parse(s), "failed to parse `{}`", s);
    }
}

#[test]
pub fn test_any_non_determinism() {
    let machine = Ascii::any()
        .kleene()
        .concat(Ascii::exact("FIN"))
        .compile();
}
