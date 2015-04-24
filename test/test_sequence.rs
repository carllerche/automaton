use automaton::encoding::Ascii;

#[test]
pub fn test_single_char_automaton() {
    let machine = Ascii::exact("a").compile();

    assert!(machine.parse("a"));
    assert!(!machine.parse("b"));
    assert!(!machine.parse("ab"));
}

#[test]
pub fn test_concat_two_chars() {
    let machine = Ascii::exact("ab").compile();

    assert!(machine.parse("ab"));

    for s in ["a", "b", "c", "d", "ac", "abc"].iter() {
        assert!(!machine.parse(s));
    }
}
