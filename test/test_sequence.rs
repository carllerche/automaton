use automaton::*;
use automaton::encoding::Ascii;

#[test]
pub fn test_single_char_automaton() {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("a")
            .compile();

    assert!(machine.parse(&mut (), "a"));
    assert!(!machine.parse(&mut (), "b"));
    assert!(!machine.parse(&mut (), "ab"));
}

#[test]
pub fn test_concat_two_chars() {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("ab")
            .compile();

    assert!(machine.parse(&mut (), "ab"));

    for s in ["a", "b", "c", "d", "ac", "abc"].iter() {
        assert!(!machine.parse(&mut (), s));
    }
}

#[test]
pub fn test_enter_action() {
    let machine = Ascii::exact("foo")
        .on_enter(|i| {
            *i = true;
            println!("ZOMG");
        })
        .compile();

    let mut invoked = false;
    assert!(machine.parse(&mut invoked, "foo"));
    assert!(invoked);
}

#[test]
pub fn test_embedding_action_in_middle() {
    let machine = Ascii::exact("foo")
        .concat(Ascii::exact("bar").on_enter(|i| *i = true))
        .compile();

    let mut invoked = false;

    assert!(machine.parse(&mut invoked, "foobar"));
    assert!(invoked);

    invoked = false;
    assert!(!machine.parse(&mut invoked, "foobaz"));
    assert!(invoked);

    invoked = false;
    assert!(!machine.parse(&mut invoked, "forbar"));
    assert!(!invoked);

    for s in ["forbar", "foo", "foogoo"].iter() {
        assert!(!machine.parse(&mut invoked, s), "parsed {}", s);
        assert!(!invoked, "invoked set {}", s);
    }
}
