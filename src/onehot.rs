use std::collections::HashMap;
use text_processing::vocab::Processed;
use serialisables::SerializableMatrix;
use ndarray::{Array2, ArrayView1};

/// Wraps a Row. 1 * V matrix into a newtype
#[derive(Clone, Copy)]
pub struct OneHot<'a>(ArrayView1<'a, f32>, usize);

impl<'a> OneHot<'a> {
    /// Returns the row contained within this one hot newtype
    pub fn row(&self) -> ArrayView1<'a, f32> {
        self.0
    }

    /// Returns the column index of "1" inside the one-hot vector
    pub fn index(&self) -> usize {
        self.1
    }
}


#[derive(Serialize, Deserialize)]
pub struct OneHotLookup {
    matrix: SerializableMatrix,
    string_to_row: HashMap<String, usize>,
    row_to_string: HashMap<usize, String>,
}

impl OneHotLookup {
    pub(crate) fn from_internal(internal: OneHotLookupInternal) -> OneHotLookup {
        let mut string_to_row = HashMap::with_capacity(internal.str_to_row.len());
        for (k, v) in internal.str_to_row.iter() {
            string_to_row.insert(k.to_string(), *v);
        }
        let mut row_to_string = HashMap::with_capacity(internal.row_to_str.len());
        for (k, v) in internal.row_to_str.iter() {
            row_to_string.insert(*k, v.to_string());
        }
        OneHotLookup {
            matrix: SerializableMatrix(internal.matrix),
            string_to_row,
            row_to_string,
        }
    }

    /// Looks up the one-hot vector for a given word, assuming it is in the
    /// lookup struct
    pub fn one_hot_for(&self, s: &str) -> Option<OneHot> {
        match self.string_to_row.get(&*s) {
            Some(i) => Some(OneHot(self.matrix.0.row(*i), *i)),
            None => None,
        }
    }

    /// Looks up the string for a given one-hot vector assuming it is in the
    /// lookup struct
    pub fn str_for(&self, r: &OneHot) -> Option<&str> {
        self.row_to_string.get(&r.index()).map(|r| r.as_ref())
    }
    /// Looks up the string for a given one-hot vector assuming it is in the
    /// lookup struct
    pub fn str_for_idx(&self, i: usize) -> Option<&str> {
        self.row_to_string.get(&i).map(|r| r.as_ref())
    }
}



/// A life-cycle-limited lookup. Makes it easy for us to use str instead of String everywhere
pub(crate) struct OneHotLookupInternal<'a> {
    /// Our corpus encoded as OneHot vectors
    pub(crate) corpus: Vec<Vec<OneHot<'a>>>,
    /// A Matrix that holds the rows that represent each word in our vocabulary as a one hot vector
    pub(crate) matrix: Array2<f32>,
    /// Map from a word to a row index in our one-hot vectors matrix
    pub(crate) str_to_row: HashMap<&'a str, usize>,
    /// Map from a row index to a word in our one-hot vectors matrix
    pub(crate) row_to_str: HashMap<usize, &'a str>,
}


impl<'a> OneHotLookupInternal<'a> {
    pub(crate) fn new(processed: &'a Processed) -> OneHotLookupInternal<'a> {
        let mut s_to_row_h = HashMap::new();
        let mut row_to_s_h = HashMap::new();
        for (i, s) in processed.vocab.iter().enumerate() {
            s_to_row_h.insert(*s, i);
            row_to_s_h.insert(i, *s);
        }
        let corpus_one_hots: Vec<Vec<OneHot>> = processed
            .split_corpus
            .iter()
            .map(|s| {
                s.0
                    .iter()
                    .filter_map(|w| {
                        s_to_row_h.get(w).map(
                            |i| OneHot(processed.onehots.row(*i), *i),
                        )
                    })
                    .collect()
            })
            .collect();
        OneHotLookupInternal {
            corpus: corpus_one_hots,
            matrix: processed.onehots.clone(),
            str_to_row: s_to_row_h,
            row_to_str: row_to_s_h,
        }

    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.matrix.rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::text_processing::sentences::Sentences;

    fn new_lookup() -> OneHotLookup {
        let sentences = Sentences::new(
            "the quick fox jumped over the lazy dog as he lay in his bed sleeping",
        );
        let p = Processed::new(&sentences, 0);
        OneHotLookup::from_internal(OneHotLookupInternal::new(&p))
    }

    #[test]
    fn test_new() {
        let s = Sentences::new("hello world everyone");
        let p = Processed::new(&s, 0);
        let _ = OneHotLookup::from_internal(OneHotLookupInternal::new(&p));
    }

    #[test]
    fn test_round_trip() {
        let lookup = new_lookup();
        let one_h = lookup.one_hot_for("fox").unwrap();
        let n = lookup.one_hot_for("whoa");
        assert!(n.is_none());
        assert_eq!(lookup.str_for(&one_h), Some("fox"));
    }

    #[test]
    fn test_one_hot_index() {
        let s = Sentences::new("quick fox");
        let p = Processed::new(&s, 0);
        let o_lookup = OneHotLookupInternal::new(&p);
        let o1 = one_hot_for(&o_lookup, "quick").unwrap();
        let o2 = one_hot_for(&o_lookup, "fox").unwrap();
        // We don't know or care which ones come first in our hashmap,
        // but we are sure one of them will have index 0 and the other, 1
        assert_eq!(o1.index() + o2.index(), 1);
    }

    /// Looks up the one-hot vector for a given word, assuming it is in the
    /// lookup struct
    fn one_hot_for<'a>(one_hot: &'a OneHotLookupInternal, s: &'a str) -> Option<OneHot<'a>> {
        match one_hot.str_to_row.get(&*s) {
            Some(i) => Some(OneHot(one_hot.matrix.row(*i), *i)),
            None => None,
        }
    }

}
