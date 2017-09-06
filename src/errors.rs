use serde_json;
use std::io;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::error::Error as StdError;

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

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::Error::*;
        match self {
            &JsonError(ref e) => e.fmt(f),
            &IOError(ref e) => e.fmt(f),
            &AddSubtractEncodings => write!(f, "Not all the words supplied had encodings"),
        }
    }
}

impl StdError for Error {
    fn description(&self) -> &str {
        use self::Error::*;
        match self {
            &JsonError(ref e) => e.description(),
            &IOError(ref e) => e.description(),
            &AddSubtractEncodings => "Encodings not found error",
        }
    }

    fn cause(&self) -> Option<&StdError> {
        use self::Error::*;
        match self {
            &JsonError(ref e) => Some(e),
            &IOError(ref e) => Some(e),
            &AddSubtractEncodings => None,
        }
    }
}
