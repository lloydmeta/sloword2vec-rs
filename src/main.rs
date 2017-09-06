extern crate sloword2vec;
extern crate clap;
extern crate indicatif;

use clap::{App, SubCommand, Arg};
use sloword2vec::training::*;
use sloword2vec::errors::Error;
use std::process::exit;
use std::error::Error as StdError;
use std::str::FromStr;
use std::path::Path;
use std::fmt::Display;
use std::fs::OpenOptions;
use std::io::Read;
use indicatif::{ProgressBar, ProgressStyle};

const CORPUS_PATH_KEY: &'static str = "corpus";
const MODEL_PATH_KEY: &'static str = "path";
const MAX_ITERATIONS_KEY: &'static str = "iterations";
const CONTEXT_RADIUS_KEY: &'static str = "context-radius";
const LEARNING_RATE: &'static str = "learning-rate";
const ENCODING_DIMENSIONS_KEY: &'static str = "dimensions";
const MIN_WORD_OCCURENCES_KEY: &'static str = "min-word-occurences";
const MIN_ERROR_IMPROVEMENT_KEY: &'static str = "min-error-improvement";
const ACCEPTABLE_ERROR_KEY: &'static str = "acceptable-error";

const SIMILAR_TO_KEY: &'static str = "word";
const SIMILAR_LIMIT_KEY: &'static str = "limit";

const ADD_KEY: &'static str = "add";
const SUBTRACT_KEY: &'static str = "subtract";

const GREET: &'static str = r#"
**************************** SloWord2Vec ****************************

Please be patient. Training is slooooooow.
"#;

fn main() {
    exit(match inner_main() {
        Ok(_) => 0,
        Err(e) => {
            println!("Something horrible happened:\n{}", e);
            1
        }
    })
}

fn inner_main() -> Result<(), Box<StdError>> {

    let v = version();
    let app = App::new("SloWord2Vec")
        .version(v.as_str())
        .author("Lloyd (github.com/lloydmeta)")
        .about("A naive Word2Vec implementation")
        .subcommand(
            SubCommand::with_name("train")
                .about("Given a corpus and a path to save a trained model, trains Word2Vec encodings for the vocabulary in the corpus and saves it.")
                .arg(
                    Arg::with_name(CORPUS_PATH_KEY)
                        .short("C")
                        .long(CORPUS_PATH_KEY)
                        .number_of_values(1)
                        .takes_value(true)
                        .help("Where the corpus file is.")
                        .required(true),
                )
                .arg(
                    Arg::with_name(MODEL_PATH_KEY)
                        .short("P")
                        .long(MODEL_PATH_KEY)
                        .number_of_values(1)
                        .takes_value(true)
                        .help("Where to store the model when training is done.")
                        .required(true),
                )
                .arg(
                    Arg::with_name(MAX_ITERATIONS_KEY)
                        .short("I")
                        .long(MAX_ITERATIONS_KEY)
                        .takes_value(true)
                        .number_of_values(1)
                        .help("Max number of training iterations.")
                        .default_value("500")
                        .validator(|s| {
                            let r: Result<usize, String> = coerce_validate(
                                &s,
                                MAX_ITERATIONS_KEY,
                                Some(("should be greater than 0", |u| u > 0)));
                            r.map(|_| ())
                        })
                        .required(false),
                )
                .arg(
                    Arg::with_name(CONTEXT_RADIUS_KEY)
                        .short("R")
                        .long(CONTEXT_RADIUS_KEY)
                        .takes_value(true)
                        .number_of_values(1)
                        .default_value("5")
                        .validator(|s| {
                            let r: Result<usize, String> = coerce_validate(
                                &s,
                                CONTEXT_RADIUS_KEY,
                                Some(("should be greater than 0", |u| u > 0)));
                            r.map(|_| ())
                        })
                        .help("The context radius (how many word surrounding a centre word to take into account per training sample).")
                        .required(false),
                )
                .arg(
                    Arg::with_name(LEARNING_RATE)
                        .short("L")
                        .long(LEARNING_RATE)
                        .takes_value(true)
                        .number_of_values(1)
                        .default_value("0.001")
                        .validator(|s| {
                            let r: Result<f32, String> = coerce_validate(
                                &s,
                                LEARNING_RATE,
                                Some(("should be greater than 0", |u| u > 0f32)));
                            r.map(|_| ())
                        })
                        .help("Learning rate.")
                        .required(false),
                )
                .arg(
                    Arg::with_name(ENCODING_DIMENSIONS_KEY)
                        .short("D")
                        .long(ENCODING_DIMENSIONS_KEY)
                        .takes_value(true)
                        .number_of_values(1)
                        .default_value("100")
                        .validator(|s| {
                            let r: Result<usize, String> = coerce_validate(
                                &s,
                                ENCODING_DIMENSIONS_KEY,
                                Some(("should be greater than 0", |u| u > 0)));
                            r.map(|_| ())
                        })
                        .help("Number of dimensions to use for encoding a word as a vector.")
                        .required(false),
                )
                .arg(
                    Arg::with_name(MIN_WORD_OCCURENCES_KEY)
                        .short("O")
                        .long(MIN_WORD_OCCURENCES_KEY)
                        .takes_value(true)
                        .number_of_values(1)
                        .default_value("2")
                        .validator(|s| {
                            let r: Result<usize, String> = coerce_validate(
                                &s,
                                MIN_WORD_OCCURENCES_KEY,
                                Some(("should be greater than 0", |u| u > 0)));
                            r.map(|_| ())
                        })
                        .help("Minimum number of occurences in the corpus a word needs to have in order to be included in the trained vocabulary.")
                        .required(false),
                )
                .arg(
                    Arg::with_name(MIN_ERROR_IMPROVEMENT_KEY)
                        .short("M")
                        .long(MIN_ERROR_IMPROVEMENT_KEY)
                        .takes_value(true)
                        .number_of_values(1)
                        .default_value("0.0000001")
                        .validator(|s| {
                            let r: Result<f32, String> = coerce_validate(
                                &s,
                                MIN_ERROR_IMPROVEMENT_KEY,
                                Some(("should be greater than 0", |u| u > 0f32)));
                            r.map(|_| ())
                        })
                        .help("Minimum improvement in average error magnitude in a single training iteration (over all words) to keep on training")
                        .required(false),
                )
                .arg(
                    Arg::with_name(ACCEPTABLE_ERROR_KEY)
                        .short("A")
                        .long(ACCEPTABLE_ERROR_KEY)
                        .takes_value(true)
                        .number_of_values(1)
                        .default_value("0.1")
                        .validator(|s| {
                            let r: Result<f32, String> = coerce_validate(
                                &s,
                                ACCEPTABLE_ERROR_KEY,
                                Some(("should be greater than 0", |u| u > 0f32)));
                            r.map(|_| ())
                        })
                        .help("Acceptable error threshold under which training will end.")
                        .required(false),
                )
        )
        .subcommand(
            SubCommand::with_name("similar")
                .about("Given a path to a saved Word2Vec model and a target word, finds words in the model's vocab that are similar.")
                .arg(
                    Arg::with_name(MODEL_PATH_KEY)
                        .short("P")
                        .long(MODEL_PATH_KEY)
                        .number_of_values(1)
                        .takes_value(true)
                        .help("Where to store the model when training is done.")
                        .required(true),
                )
                .arg(
                    Arg::with_name(SIMILAR_TO_KEY)
                        .short("W")
                        .long(SIMILAR_TO_KEY)
                        .number_of_values(1)
                        .takes_value(true)
                        .help("Word to find similar terms for.")
                        .required(true),
                )
                .arg(
                    Arg::with_name(SIMILAR_LIMIT_KEY)
                        .short("L")
                        .long(SIMILAR_LIMIT_KEY)
                        .number_of_values(1)
                        .takes_value(true)
                        .help("Max number of similar entries to show.")
                        .default_value("20")
                        .validator(|s| {
                            let r: Result<usize, String> = coerce_validate(
                                &s,
                                ACCEPTABLE_ERROR_KEY,
                                Some(("should be greater than 0", |u| u > 0)));
                            r.map(|_| ())
                        })
                        .required(true),
                )
        ).subcommand(
            SubCommand::with_name("add-subtract")
                .about("Given a number of words to add and to subtract, returns a list of words in that area.")
                .arg(
                    Arg::with_name(MODEL_PATH_KEY)
                        .short("P")
                        .long(MODEL_PATH_KEY)
                        .number_of_values(1)
                        .takes_value(true)
                        .help("Where to store the model when training is done.")
                        .required(true),
                )
                .arg(
                    Arg::with_name(ADD_KEY)
                        .short("A")
                        .long(ADD_KEY)
                        .multiple(true)
                        .takes_value(true)
                        .help("Words to add encodings for")
                        .required(true),
                )
                .arg(
                    Arg::with_name(SUBTRACT_KEY)
                        .short("S")
                        .long(SUBTRACT_KEY)
                        .multiple(true)
                        .takes_value(true)
                        .help("Words to subtract encodings for")
                        .required(true),
                )
                .arg(
                    Arg::with_name(SIMILAR_LIMIT_KEY)
                        .short("L")
                        .long(SIMILAR_LIMIT_KEY)
                        .number_of_values(1)
                        .takes_value(true)
                        .help("Max number of similar entries to show.")
                        .default_value("20")
                        .validator(|s| {
                            let r: Result<usize, String> = coerce_validate(
                                &s,
                                ACCEPTABLE_ERROR_KEY,
                                Some(("should be greater than 0", |u| u > 0)));
                            r.map(|_| ())
                        })
                        .required(true),
                )
        );

    let mut app_clone = app.clone();

    let matches = app.get_matches();

    if let Some(training_matches) = matches.subcommand_matches("train") {
        let match_tuple = (
            training_matches.value_of(MODEL_PATH_KEY),
            training_matches.value_of(CORPUS_PATH_KEY),
            training_matches.value_of(MAX_ITERATIONS_KEY),
            training_matches.value_of(CONTEXT_RADIUS_KEY),
            training_matches.value_of(LEARNING_RATE),
            training_matches.value_of(ENCODING_DIMENSIONS_KEY),
            training_matches.value_of(MIN_WORD_OCCURENCES_KEY),
            training_matches.value_of(MIN_ERROR_IMPROVEMENT_KEY),
            training_matches.value_of(ACCEPTABLE_ERROR_KEY),
        );
        match match_tuple {
            (Some(model_path_str),
             Some(corpus_path_str),
             Some(max_iterations_str),
             Some(context_radius_str),
             Some(learning_rate_str),
             Some(enc_dims_str),
             Some(min_word_occ_str),
             Some(min_err_impr_str),
             Some(acpt_err_str)) => {
                let max_iterations = usize::from_str(max_iterations_str)?;
                let context_radius = usize::from_str(context_radius_str)?;
                let learning_rate = f32::from_str(learning_rate_str)?;
                let encoding_dims = usize::from_str(enc_dims_str)?;
                let min_word_occ = usize::from_str(min_word_occ_str)?;
                let min_err_impr = f32::from_str(min_err_impr_str)?;
                let acpt_err = f32::from_str(acpt_err_str)?;
                let training_params = TrainingParams {
                    max_iterations: max_iterations,
                    learning_rate: learning_rate,
                    encoding_dimensions: encoding_dims,
                    context_radius: context_radius,
                    min_occurences: min_word_occ,
                    avg_err_improvement_min: min_err_impr,
                    acceptable_err: acpt_err,
                };
                Ok(train_and_save(
                    corpus_path_str,
                    model_path_str,
                    training_params,
                )?)
            }
            _ => Ok(app_clone.print_long_help()?),
        }
    } else if let Some(similar_matches) = matches.subcommand_matches("similar") {
        let match_tuple = (
            similar_matches.value_of(MODEL_PATH_KEY),
            similar_matches.value_of(SIMILAR_TO_KEY),
            similar_matches.value_of(SIMILAR_LIMIT_KEY),
        );
        match match_tuple {
            (Some(model_path_str), Some(similar_to_target), Some(similar_limit_str)) => {
                let similar_limit = usize::from_str(similar_limit_str)?;
                Ok(load_and_find_similar(
                    model_path_str,
                    similar_to_target,
                    similar_limit,
                )?)
            }
            _ => Ok(app_clone.print_long_help()?),
        }
    } else if let Some(add_subtract_matches) = matches.subcommand_matches("add-subtract") {
        let match_tuple = (
            add_subtract_matches.value_of(MODEL_PATH_KEY),
            add_subtract_matches.values_of(ADD_KEY),
            add_subtract_matches.values_of(SUBTRACT_KEY),
            add_subtract_matches.value_of(SIMILAR_LIMIT_KEY),
        );
        match match_tuple {
            (Some(model_path_str),
             Some(add_strs),
             Some(subtract_strs),
             Some(similar_limit_str)) => {
                let similar_limit = usize::from_str(similar_limit_str)?;
                Ok(load_and_do_maths(
                    model_path_str,
                    &add_strs.collect(),
                    &subtract_strs.collect(),
                    similar_limit,
                )?)
            }
            _ => Ok(app_clone.print_long_help()?),
        }
    } else {
        Ok(app_clone.print_long_help()?)
    }

}

fn train_and_save<P: AsRef<Path>>(
    corpus_path: P,
    model_path: P,
    training_params: TrainingParams,
) -> Result<(), Error> {
    println!("{}", GREET);
    let mut corpus_file = OpenOptions::new().read(true).open(corpus_path)?;
    let mut contents = String::new();
    corpus_file.read_to_string(&mut contents)?;
    let pb = ProgressBar::new(training_params.max_iterations as u64);
    pb.set_position(1);
    pb.enable_steady_tick(50);
    pb.set_message("Working ...");
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>5}/{len:5} [ETA: {eta}] {msg}")
        .progress_chars("#>-"));
    let trained = Trainer::train(
        &contents,
        training_params,
        Some(|_, avg_err| {
            let msg = format!("Avg err: {:.4}", avg_err);
            pb.inc(1);
            pb.set_message(&msg);
        }),
    );
    pb.finish_with_message("Training finished. Saving now.");
    trained.save_to(model_path)
}

fn load_and_find_similar<P: AsRef<Path>>(
    model_path: P,
    similar_to: &str,
    limit: usize,
) -> Result<(), Error> {
    println!("Loading the model ...");
    let trained = Trained::load_from(model_path)?;
    println!("Finding similar terms ...");
    let found = trained.similar_to(similar_to);
    if found.len() > 0 {
        println!("Similar to \"{}\"", similar_to);
        let limited: Vec<_> = found.iter().take(limit).collect();
        let longest_str_len = limited.iter().fold(0, |acc, &&(s, _)| {
            std::cmp::max(s.chars().count(), acc)
        });
        let spacing = longest_str_len + 4;
        for &&(term, similarity) in limited.iter() {
            println!(
                "Term: {:width$} Similarity: {similarity}",
                term,
                width = spacing,
                similarity = similarity
            )
        }
    } else {
        println!(
            "No similar terms found for \"{}\". It likely isn't in the vocab!",
            similar_to
        )
    }
    Ok(())
}

fn load_and_do_maths<P: AsRef<Path>>(
    model_path: P,
    adds: &Vec<&str>,
    subtracts: &Vec<&str>,
    limit: usize,
) -> Result<(), Error> {
    println!("Loading the model ...");
    let trained = Trained::load_from(model_path)?;
    println!("Finding similar terms ...");
    let found = trained.add_subtract(adds, subtracts)?;
    if found.len() > 0 {
        println!("{:?} - {:?}", adds, subtracts);
        let limited: Vec<_> = found.iter().take(limit).collect();
        let longest_str_len = limited.iter().fold(0, |acc, &&(s, _)| {
            std::cmp::max(s.chars().count(), acc)
        });
        let spacing = longest_str_len + 4;
        for &&(term, similarity) in limited.iter() {
            println!(
                "Term: {:width$} Similarity: {similarity}",
                term,
                width = spacing,
                similarity = similarity
            )
        }
    }
    Ok(())
}

fn coerce_validate<'a, T, F>(
    s: &'a str,
    field: &'a str,
    validation: Option<(&'a str, F)>,
) -> Result<T, String>
where
    T: FromStr + Copy,
    <T as FromStr>::Err: Display,
    F: Fn(T) -> bool,
{
    let as_t = s.parse().map_err(
        |e| format!("Could not parse {}: {}", field, e),
    );
    let validated = as_t.and_then(|t| if let Some((validation_err, ref validator)) =
        validation
    {
        if validator(t) {
            Ok(t)
        } else {
            Err(validation_err.to_string())
        }
    } else {
        Ok(t)
    });
    match validated {
        Ok(t) => Ok(t),
        Err(err) => Err(err),
    }
}


/// Return the current crate version
fn version() -> String {
    let (maj, min, pat) = (
        option_env!("CARGO_PKG_VERSION_MAJOR"),
        option_env!("CARGO_PKG_VERSION_MINOR"),
        option_env!("CARGO_PKG_VERSION_PATCH"),
    );
    match (maj, min, pat) {
        (Some(maj), Some(min), Some(pat)) => format!("{}.{}.{}", maj, min, pat),
        _ => "".to_owned(),
    }
}
