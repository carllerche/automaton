use automaton::*;
use automaton::encoding::Ascii;

#[test]
pub fn test_intersection_with_kleene() {
    let machine: Automaton<Ascii, ()> = Ascii::exact("a").kleene()
        .concat(Ascii::exact("b"))
        .intersection(
            Ascii::exact("b")
                .concat(Ascii::exact("c").kleene()))
        .compile();

    assert!(machine.parse(&mut (), "b"));

    for s in ["ab", "bc"].iter() {
        assert!(!machine.parse(&mut (), s), "parsed `{}`", s);
    }
}
