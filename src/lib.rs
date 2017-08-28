#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate rand;
extern crate rayon;

#[macro_use]
extern crate log;

#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;

extern crate flate2;

#[macro_use]
extern crate ndarray;

pub mod onehot;
pub mod text_processing;
pub mod training;
pub mod errors;
mod serialisables;
