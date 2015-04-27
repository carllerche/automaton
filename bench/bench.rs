#![cfg(feature = "bench")]
#![feature(test)]

extern crate test;
extern crate regex;
extern crate automaton;

use automaton::*;
use automaton::encoding::Ascii;
use regex::Regex;
use test::Bencher;

const STR1: &'static str = "abcdefghijkl";
const STR2: &'static str = "zxywijslyla";

#[bench]
pub fn bench_automaton_sequence(b: &mut Bencher) {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact(STR1)
            .compile();

    b.iter(|| {
        machine.parse(STR1);
    })
}

#[bench]
pub fn bench_regex_sequence(b: &mut Bencher) {
    let re = Regex::new(STR1).unwrap();

    b.iter(|| {
        re.is_match(STR1);
    })
}

#[bench]
pub fn bench_automaton_union(b: &mut Bencher) {
    let machine: Automaton<Ascii, ()> =
        Ascii::exact("a")
            .union(Ascii::exact("b"))
            .union(Ascii::exact("c"))
            .union(Ascii::exact("d"))
            .union(Ascii::exact(STR1))
            .union(Ascii::exact(STR2))
            .compile();

    b.iter(|| {
        assert!(machine.parse(STR1));
        assert!(machine.parse(STR2));
    })
}

#[bench]
pub fn bench_regex_union(b: &mut Bencher) {
    let re = Regex::new(&format!("a|b|c|d|{}|{}", STR1, STR2)).unwrap();

    b.iter(|| {
        assert!(re.is_match(STR1));
        assert!(re.is_match(STR2));
    })
}
