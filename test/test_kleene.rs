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
