use std::collections::{HashSet, HashMap};
use std::iter::FromIterator;
use super::sentences::Sentences;

use ndarray::Array2;

pub(crate) struct Sentence<'a>(pub(crate) Vec<&'a str>);

pub(crate) struct Processed<'a> {
    pub(crate) split_corpus: Vec<Sentence<'a>>,
    pub(crate) vocab: Vec<&'a str>,
    pub(crate) onehots: Array2<f32>,
}

impl<'a> Processed<'a> {
    pub fn new(sentences: &Sentences, min_occurence: usize) -> Processed {
        let flattened_words: Vec<&str> = sentences
            .data
            .iter()
            .flat_map(|s| {
                let v: Vec<_> = s.iter().map(|w| w.as_ref()).collect();
                v
            })
            .collect();
        let counted = to_counted_word_map(&flattened_words);
        let vocab: Vec<&str> = counted
            .iter()
            .filter(|&(_, o)| *o >= min_occurence)
            .map(|(k, _)| *k)
            .collect();
        let vocab_set: HashSet<&str> = HashSet::from_iter(vocab.iter().map(|s| *s));
        let trimmed_split_corpus: Vec<_> = sentences
            .data
            .iter()
            .filter_map(|s| {
                let vocab_only: Vec<_> = s.iter()
                    .filter_map(|w| if vocab_set.contains(&w.as_ref()) {
                        Some(w.as_ref())
                    } else {
                        None
                    })
                    .collect();
                if vocab_only.len() > 0 {
                    Some(Sentence(vocab_only))
                } else {
                    None
                }
            })
            .collect();
        let onehots = Array2::eye(vocab.len());
        Processed {
            split_corpus: trimmed_split_corpus,
            vocab: vocab,
            onehots: onehots,
        }
    }
}

fn to_counted_word_map<'a>(flattened_words: &Vec<&'a str>) -> HashMap<&'a str, usize> {
    let mut h = HashMap::new();
    for w in flattened_words {
        *h.entry(*w).or_insert(0usize) += 1;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sentences::Sentences;

    #[test]
    fn new_processed_test() {
        let s = Sentences::new("hello there. my name is lloyd. what is your name?");
        let p = Processed::new(&s, 0);
        // First and last sentences are made of all stop words so should not be included
        assert_eq!(p.split_corpus.len(), 1);
    }
}
