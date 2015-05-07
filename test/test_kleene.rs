use automaton::*;
use automaton::encoding::Ascii;

#[test]
pub fn test_simple_kleene_star() {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("ab")
            .kleene()
            .compile();

    for s in ["", "ab", "abab", "ababab"].iter() {
        assert!(machine.parse(&mut (), s), "failed to parse `{}`", s);
    }

    for s in ["a", "c", "ac", "aba", "abc", "ababc"].iter() {
        assert!(!machine.parse(&mut (), s), "parsed `{}`", s);
    }
}

#[test]
pub fn test_simple_non_determinism() {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("ab")
            .kleene()
            .concat(Ascii::exact("ac"))
            .compile();

    for s in ["ac", "abac", "ababac"].iter() {
        assert!(machine.parse(&mut (), s), "failed to parse `{}`", s);
    }
}

#[test]
pub fn test_any_non_determinism() {
    let _: Automaton<Ascii, ()> =
        Ascii::any()
            .kleene()
            .concat(Ascii::exact("FIN"))
            .compile();
}

#[test]
pub fn test_embed_enter_action_in_kleen() {
    let mut invoked = 0;

    let machine = Ascii::exact::<i32>("foo")
        .kleene()
        .on_enter(|i| {
            *i += 1;
        })
        .compile();

    assert!(machine.parse(&mut invoked, ""));
    assert_eq!(invoked, 0);

    assert!(machine.parse(&mut invoked, "foo"));
    assert_eq!(invoked, 1);

    invoked = 0;

    assert!(machine.parse(&mut invoked, "foofoo"));
    assert_eq!(invoked, 1);
}

#[test]
pub fn test_embed_enter_action_before_kleen() {
    let mut invoked = 0;

    let machine = Ascii::exact::<i32>("foo")
        .on_enter(|i| {
            *i += 1;
        })
        .kleene()
        .compile();

    assert!(machine.parse(&mut invoked, ""));
    assert_eq!(invoked, 0);

    assert!(machine.parse(&mut invoked, "foo"));
    assert_eq!(invoked, 1);

    invoked = 0;

    assert!(machine.parse(&mut invoked, "foofoo"));
    assert_eq!(invoked, 2);
}
