use std::io::Read;
use automaton::*;
use automaton::encoding::Ascii;

#[test]
pub fn test_stream() {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("a")
            .kleene()
            .concat(Ascii::exact("b"))
            .compile();

    let stream = "aaaaaab".as_bytes();
    assert!(machine.try_eval(&mut (), stream.chars()).is_ok());
}
