use serde_json;
use std::io;

/// Project-specfic error enum.
#[derive(Debug)]
pub enum Error {
    JsonError(serde_json::Error),
    IOError(io::Error),
    AddSubtractEncodings,
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::JsonError(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::IOError(e)
    }
}
