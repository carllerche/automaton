use std::{cmp, hash};
use std::collections::HashSet;

pub fn pop<T: cmp::Eq + hash::Hash + Clone>(set: &mut HashSet<T>) -> Option<T> {
    let val = set.iter().next().cloned();

    if let Some(val) = val.as_ref() {
        assert!(set.remove(val));
    }

    val
}

pub fn hash<T: hash::Hash>(val: &T) -> u64 {
    use std::hash::Hasher;

    let mut state = hash::SipHasher::new();
    val.hash(&mut state);
    state.finish()
}
