use rand::random;
use ndarray::Array;
use std::f32::consts::E;
use std::f32;
use super::onehot::{OneHotLookupInternal, OneHotLookup, OneHot};
use text_processing::sentences::Sentences;
use text_processing::vocab::Processed;
use std::cmp::Ordering;
use serde_json;
use std::fs::OpenOptions;
use std::path::Path;
use std::io::{Write, Read};
use errors::Error;
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;

use serialisables::SerializableMatrix;
use ndarray::{ArrayView1, Array2, Array1, ArrayView2};

/// Holds state that represents the trained result of Word2Vec
///
/// To get a hold of one of these, use Trainer::train(s, training_params)
#[derive(Serialize, Deserialize)]
pub struct Trained {
    #[doc(hidden)]
    one_hot_lookup: OneHotLookup,
    // A V x N matrix where V is vocab size and N is # of dimensions
    encodings: SerializableMatrix,
}


impl Trained {
    /// Returns the encoding for a given string, if it exists
    pub fn encoding_of(&self, s: &str) -> Option<ArrayView1<f32>> {
        self.one_hot_lookup.one_hot_for(s).map(|o| {
            self.encodings.0.row(o.index())
        })
    }

    /// Returns a list of terms and their similarity to the given word
    pub fn similar_to(&self, s: &str) -> Vec<(&str, f32)> {
        let opt_encoding = self.encoding_of(s);
        opt_encoding
            .map(|encoding| {
                let mut with_dots: Vec<_> = self.encodings
                    .0
                    .outer_iter()
                    .enumerate()
                    .map(|(i, e)| (i, e, Trained::similarity(&encoding, &e)))
                    .filter_map(|(i, _, s)| {
                        self.one_hot_lookup.str_for_idx(i).map(|w| (w, s))
                    })
                    .filter(|&(w, _)| w != s)
                    .collect();
                with_dots.sort_by(|&(_, s1), &(_, s2)| {
                    s1.partial_cmp(&s2).unwrap_or(Ordering::Equal).reverse()
                });
                with_dots
            })
            .unwrap_or(vec![])
    }


    pub fn add_subtract(
        &self,
        adds: &Vec<&str>,
        subtracts: &Vec<&str>,
    ) -> Result<Vec<(&str, f32)>, Error> {
        let add_encodings: Vec<_> = adds.iter().filter_map(|s| self.encoding_of(s)).collect();
        let subtract_encodings: Vec<_> = subtracts
            .iter()
            .filter_map(|s| self.encoding_of(s))
            .collect();

        if add_encodings.len() != adds.len() || subtract_encodings.len() != subtracts.len() {
            Err(Error::AddSubtractEncodings)
        } else {
            let sum = {
                let add_sum = add_encodings.iter().fold(
                    Array1::zeros(self.encodings.0.cols()),
                    |acc, row| acc + row,
                );
                subtract_encodings.iter().fold(
                    add_sum,
                    |acc, row| acc - row,
                )
            };
            let mut with_dots: Vec<_> = self.encodings
                .0
                .outer_iter()
                .enumerate()
                .map(|(i, e)| (i, e, Self::similarity(&sum.view(), &e)))
                .filter_map(|(i, _, s)| {
                    self.one_hot_lookup.str_for_idx(i).map(|w| (w, s))
                })
                .filter(|&(w, _)| !(adds.contains(&w) || subtracts.contains(&w)))
                .collect();
            with_dots.sort_by(|&(_, s1), &(_, s2)| {
                s1.partial_cmp(&s2).unwrap_or(Ordering::Equal).reverse()
            });
            Ok(with_dots)
        }
    }

    fn similarity(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
        let dotted = x.dot(y);
        let x_mag = x.iter().fold(0f32, |acc, n| acc + n * n).sqrt();
        let y_mag = y.iter().fold(0f32, |acc, n| acc + n * n).sqrt();
        dotted / (x_mag * y_mag)
    }

    /// Saves the trained data to a path
    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let mut file = OpenOptions::new().write(true).create(true).open(path)?;
        let serialised = serde_json::to_string(self)?;
        let mut e = GzEncoder::new(Vec::new(), Compression::Best);
        e.write_all(serialised.as_bytes())?;
        let compressed_bytes = e.finish()?;
        Ok(file.write_all(&compressed_bytes)?)
    }

    /// Loads trained stuff from a path
    pub fn load_from<P: AsRef<Path>>(path: P) -> Result<Trained, Error> {
        let mut file = OpenOptions::new().read(true).open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut gz = GzDecoder::new(&bytes[..])?;
        let mut s = String::new();
        gz.read_to_string(&mut s)?;
        Ok(serde_json::from_str(&s)?)
    }
}

/// Holds hyper parameters for our trainer
pub struct TrainingParams {
    pub max_iterations: usize,
    pub learning_rate: f32,
    pub encoding_dimensions: usize,
    pub context_radius: usize,
    pub min_occurences: usize,
    pub avg_err_improvement_min: f32,
    pub acceptable_err: f32,
}

/// Our scratchpad that holds the state of our training
pub struct Trainer<'a> {
    /// Allows us to look up one hot vectors for a given str and back.
    one_hot_lookup: OneHotLookupInternal<'a>,
    /// V x N where V is vocab size, N is the number of dimensions in the word encoding
    input_weights: Array2<f32>,
    /// N x V where V is vocab size, N is the number of dimensions in the word encoding
    output_weights: Array2<f32>,
    /// The average of all the error magnitudes (across all centre words) in a given corpus training iteration
    last_avg_err: Option<f32>,
    /// Hyper parameters for our trainer
    hyper_params: TrainingParams,
}

struct ForwardPropLayers {
    hidden: Array1<f32>,
    output_as_softmax: Array1<f32>,
}

impl<'a> Trainer<'a> {
    pub fn train<F>(
        s: &str,
        training_params: TrainingParams,
        per_iteration_callback: Option<F>,
    ) -> Trained
    where
        F: FnMut(usize, f32) -> (),
    {
        let sentences = Sentences::new(s);
        let processed = Processed::new(&sentences, training_params.min_occurences);
        let one_hot_internal = OneHotLookupInternal::new(&processed);
        let mut trainer = Trainer::new(one_hot_internal, training_params);
        trainer.run(per_iteration_callback);

        let one_hot_lookup = OneHotLookup::from_internal(trainer.one_hot_lookup);
        Trained {
            encodings: SerializableMatrix(trainer.input_weights),
            one_hot_lookup,
        }
    }

    fn new(
        one_hot_lookup: OneHotLookupInternal<'a>,
        training_params: TrainingParams,
    ) -> Trainer<'a> {
        let vocab_size = one_hot_lookup.vocab_size();
        let input_weights = Array::from_shape_fn(
            (vocab_size, training_params.encoding_dimensions),
            |_| random(),
        );
        // Not sure if we can just transpose input_weights. Just use a different one just in case.
        let output_weights = Array::from_shape_fn(
            (training_params.encoding_dimensions, vocab_size),
            |_| random(),
        );
        Trainer {
            one_hot_lookup,
            input_weights,
            output_weights,
            hyper_params: training_params,
            last_avg_err: None,
        }
    }

    fn run<F: FnMut(usize, f32) -> ()>(&mut self, mut per_iteration_callback: Option<F>) -> () {
        let max_iterations = self.hyper_params.max_iterations;
        let sentences_count = self.sentence_onehots().len();
        for training_iteration in 0..max_iterations {
            info!("Training iteration {}", training_iteration);
            /*
             *  This may look annoyingly complex, and it is, but all the index-usage and weird
             *  binding placements are to satisfy the borrow checker so that we can keep track of
             *  the per-sentence avg error magnitude *and* update the weights for every single word,
             *  which should lead to faster learning overall.
             */
            let total_err_magnitude =
                (0..sentences_count).fold(0f32, |avg_err_sum, sentence_index| {
                    let sentence_avg_err = {
                        let words_in_sentence = self.sentence_onehots()[sentence_index].len();
                        let total_sentence_error = (0..words_in_sentence).fold(0f32, |inner_acc,
                         word_index| {
                            let ret = {
                                let sentence = &self.sentence_onehots()[sentence_index];
                                let (real_context, centre_word) =
                                    self.context_at_index(sentence, word_index);
                                let forward_propagation = self.forward_propagation(&centre_word);
                                let error = self.error_vector(
                                    &forward_propagation.output_as_softmax,
                                    &(&real_context, centre_word),
                                );
                                let err_magnitude =
                                    error.iter().fold(0f32, |acc, n| acc + n * n).sqrt();
                                (
                                    centre_word.index(),
                                    forward_propagation,
                                    error,
                                    err_magnitude,
                                )
                            };
                            let (one_hot_index, ref forward_propagation, ref error, err_magnitude) =
                            ret;
                            self.update_input_weights(one_hot_index, error);
                            self.update_output_weights(forward_propagation, error);
                            inner_acc + err_magnitude
                        });
                        total_sentence_error / (words_in_sentence as f32)
                    };
                    avg_err_sum + sentence_avg_err
                });
            let current_avg_err = total_err_magnitude / (sentences_count as f32);
            info!("current_err_avg {:?}", current_avg_err);
            match self.last_avg_err {
                Some(last_avg_err)
                    if (last_avg_err - current_avg_err) <
                           self.hyper_params.avg_err_improvement_min => {
                    info!(
                        "Avg error improvement is under {}, stopping at {} iterations",
                        self.hyper_params.avg_err_improvement_min,
                        training_iteration
                    );
                    break;
                }
                _ if current_avg_err <= self.hyper_params.acceptable_err => {
                    info!(
                        "Avg err is under acceptable error threshold of {}, stopping at {} iterations",
                        self.hyper_params.acceptable_err,
                        training_iteration
                    );
                    break;
                }
                _ => {
                    self.last_avg_err = {
                        if let Some(ref mut cb) = per_iteration_callback {
                            cb(training_iteration + 1, current_avg_err);
                        }
                        Some(current_avg_err)
                    }
                }
            }
        }
    }

    // Shortcut to the corpus
    fn sentence_onehots(&self) -> &Vec<Vec<OneHot<'a>>> {
        &self.one_hot_lookup.corpus
    }

    // For a given sentence (made of OneHots) and a word index, context one hots and
    // the one hot for the index.alloc
    // Note: does not do any bound checking for getting the index itself.
    fn context_at_index(
        &self,
        sentence_onehots: &Vec<OneHot<'a>>,
        index: usize,
    ) -> (Vec<OneHot<'a>>, OneHot<'a>) {
        let indices_raw = index as isize - self.hyper_params.context_radius as isize..
            index as isize + self.hyper_params.context_radius as isize;
        let indices = indices_raw
            .filter(|i| *i >= 0 && *i < sentence_onehots.len() as isize)
            .map(|i| i as usize);
        let pairs = indices.map(|u| sentence_onehots[u]).collect();
        (pairs, sentence_onehots[index])
    }

    fn forward_propagation(&self, centre_word_one_hot: &OneHot) -> ForwardPropLayers {
        let h = self.to_hidden_layer(centre_word_one_hot);
        let mut output = self.to_output_layer(&h);
        to_softmax_ndarray(&mut output);
        ForwardPropLayers {
            hidden: h.to_owned(),
            output_as_softmax: output.to_owned(),
        }
    }

    /// Takes a word and returns a 1 x N Matrix
    /// Not that instead of actually computing the dot-product between the vector and our input
    /// weight matrix, we simply take the row, because it is faster.
    fn to_hidden_layer(&self, one_hot: &OneHot) -> ArrayView1<f32> {
        let r = self.input_weights.row(one_hot.index());
        debug!("Hidden layer: {}", r);
        r
    }

    /// Given softmax_output and a truth context with centre, returns the 1 x V
    /// error matrix
    fn error_vector(
        &self,
        softmax_output: &Array1<f32>,
        truth_context_with_centre: &(&Vec<OneHot>, OneHot),
    ) -> Array1<f32> {
        let one_hot_context = truth_context_with_centre
            .0
            .iter()
            /*
                This is a bit messy, but in essence, we want to sum up the context
                from individual one-hots into a single vector with size 1 x V
            */
            .fold(Array1::zeros(softmax_output.len()),
                  |acc, n| acc + n.row());
        debug!("one_hot_context: {}", one_hot_context);
        debug!(
            "softmax_output: {}",
            softmax_output * (truth_context_with_centre.0.len() as f32)
        );
        softmax_output * (truth_context_with_centre.0.len() as f32) - one_hot_context
    }

    fn update_input_weights(
        &mut self,
        word_index: usize,
        // a 1 x V error matrix
        error: &Array1<f32>,
    ) -> () {
        // println!("output_weights dot error_t");
        // (N x V) * (1 x V)' => N x 1
        let e_w_matrix = self.output_weights.dot(error);
        debug!("EI * W: {}", e_w_matrix);
        // N x 1 * n -> N x 1
        let times_learning_rate = e_w_matrix * self.hyper_params.learning_rate;
        debug!("EI * W * learning_rate: {}", times_learning_rate);
        {
            // Take the entire row of the V x N input matrix that corresponds to the current one_not
            // word.
            let idx = word_index as isize;
            let mut mutable_slice = self.input_weights.slice_mut(s![idx..idx + 1, ..]);
            // 1 x N - (N x 1)' => 1 x N
            mutable_slice -= &times_learning_rate.t();
        }
    }

    fn update_output_weights(
        &mut self,
        f_prop: &ForwardPropLayers,
        // a 1 x V error matrix
        error: &Array1<f32>,
    ) -> () {
        // 1 x V
        let error_as_m = broadcast_to_2d(error);
        debug!("error_as_m: {}", error_as_m);
        // V x 1
        let error_t = error_as_m.t();
        debug!("error_t: {}", error_t);
        // 1 x N
        let f_prop_as_m = broadcast_to_2d(&f_prop.hidden);
        debug!("forward-propagaged hidden layer: {}", f_prop_as_m);
        // V x 1 times 1 x N -> V x N
        let e_times_h = error_t.dot(&f_prop_as_m); // V x N
        debug!("Eerror * hidden: {}", e_times_h);
        let e_times_h_t = e_times_h.t(); // transposed -> N x V
        debug!("(E * h)': {}", e_times_h_t);
        let with_learn = e_times_h_t.to_owned() * self.hyper_params.learning_rate;
        debug!("(E * h)' * learning rate: {}", with_learn);
        let mut outputs_as_slice_mut = self.output_weights.slice_mut(s![.., ..]);
        outputs_as_slice_mut -= &with_learn
    }

    /// Takes the hiddne layer, a 1 x N matrix and returns the output layer, a 1 x V matrix
    fn to_output_layer(&self, hidden_layer: &ArrayView1<f32>) -> Array1<f32> {
        // ((N x V)' * (1 X N)')' => ((V x N) * (N x 1))' => (V x 1)' => (1 x V)
        let r = self.output_weights
            .t()
            .dot(&hidden_layer.t())
            .t()
            .to_owned();
        debug!("output layer: {}", r);
        r
    }
}

fn broadcast_to_2d(a: &Array1<f32>) -> ArrayView2<f32> {
    let c = a.len();
    a.broadcast((1, c))
        .and_then(|m| m.into_shape((1, c)).ok())
        .expect("Broadcast to 2D row matrix should not fail")
}

/// Does element-wise conversion to softmax
fn to_softmax_ndarray(m: &mut Array1<f32>) -> () {
    let max: f32 = *m.iter()
        .max_by(|s1, s2| s1.partial_cmp(&s2).unwrap_or(Ordering::Equal))
        .unwrap_or(&0f32);
    let summed: f32 = m.iter()
        .map(|v| {
            let r = E.powf(*v - max);
            if r.is_nan() {
                panic!("Fatal error: e^{} is not a number", v)
            }
            r
        })
        .sum();
    for v in m.into_iter() {
        let r = E.powf(*v - max) / summed;
        if r.is_nan() {
            panic!("Fatal error: Softmax is not a number: e^{} / {}", v, summed)
        }
        *v = r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    extern crate env_logger;

    fn trained() -> Trained {
        let training_params = TrainingParams {
            encoding_dimensions: 100,
            max_iterations: 50,
            context_radius: 5,
            min_occurences: 2,
            avg_err_improvement_min: 0.0001,
            acceptable_err: 0.01,
            learning_rate: 0.001,
        };
        Trainer::train(
            include_str!("data/nk-missiles"),
            training_params,
            Some(|_, _| ()),
        )
    }

    #[test]
    fn test_train() {
        let trained = trained();
        let hello_encoding = trained.encoding_of("missiles").unwrap();
        assert_eq!(hello_encoding.len(), 100);
    }

    #[test]
    fn test_save() {
        let _ = env_logger::init();
        let trained = trained();
        for w in ["Trump", "missiles", "Guam"].iter() {
            let similar_to = trained.similar_to(w);
            info!("similar_to \"{}\": {:?}", w, similar_to);
            assert!(similar_to.len() > 0);
        }
        trained.save_to("./models/nk-missiles-model.gzm").unwrap();
    }

    #[test]
    fn test_save_load() {
        let _ = env_logger::init();
        let trained = Trained::load_from("./models/nk-missiles-loading.gzm").unwrap();
        for w in ["Trump", "missiles", "Guam"].iter() {
            let similar_to = trained.similar_to(w);
            info!("similar_to \"{}\": {:?}", w, similar_to);
            assert!(similar_to.len() > 0);
        }
    }

    #[test]
    fn test_add_subtract() {
        let _ = env_logger::init();
        let trained = Trained::load_from("./models/nk-missiles-loading.gzm").unwrap();
        let adds = vec!["missiles", "American"];
        let subtracts = vec!["Korean"];
        let results = trained.add_subtract(&adds, &subtracts).unwrap();
        info!("Add: {:?}, Subtract: {:?}", adds, subtracts);
        for (w, similarity) in results {
            info!("Term: {} Similarity: {}", w, similarity)
        }
    }

    #[test]
    fn test_ndarray_mut_slice() {
        let mut a = arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]);
        {
            let mut row_2 = a.slice_mut(s![1..2, ..]);
            let to_minus = Array2::from_elem((1, 4), -1f32);
            row_2 += &to_minus;
        }
        assert_eq!(
            a,
            arr2(&[[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]])
        )
    }

}
