use automaton::encoding::Ascii;

#[test]
pub fn test_intersection_with_kleene() {
    let a = Ascii::exact("a").kleene();
    let b = Ascii::exact("b");
    let c = Ascii::exact("c").kleene();

    let machine = a.concat(b.clone())
        .intersection(b.concat(c))
        .compile();

    assert!(machine.parse("b"));

    for s in ["ab", "bc"].iter() {
        assert!(!machine.parse(s), "parsed `{}`", s);
    }
}
