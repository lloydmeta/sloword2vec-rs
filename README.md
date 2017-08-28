# SloWord2Vec

This is a naive implementation of Word2Vec implemented in Rust.

The goal is to learn the basic principles and formulas behind Word2Vec. BTW, it's slow ;)

## Getting it

This lib is available as a lib and as a binary.

### Binary

```
A naive Word2Vec implementation

USAGE:
    sloword2vec [SUBCOMMAND]

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

SUBCOMMANDS:
    add-subtract    Given a number of words to add and to subtract, returns a list of words in that area.
    help            Prints this message or the help of the given subcommand(s)
    similar         Given a path to a saved Word2Vec model and a target word, finds words in the model's vocab that are similar.
    train           Given a corpus and a path to save a trained model, trains Word2Vec encodings for the vocabulary in the corpus and saves it.
```

#### Training

```
Given a corpus and a path to save a trained model, trains Word2Vec encodings for the vocabulary in the corpus and saves it.

USAGE:
    sloword2vec train [OPTIONS] --corpus <corpus> --path <path>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -A, --acceptable-error <acceptable-error>              Acceptable error threshold under which training will end. [default: 0.1]
    -R, --context-radius <context-radius>                  The context radius (how many word surrounding a centre word to take into account per training sample). [default: 5]
    -C, --corpus <corpus>                                  Where the corpus file is.
    -D, --dimensions <dimensions>                          Number of dimensions to use for encoding a word as a vector. [default: 100]
    -I, --iterations <iterations>                          Max number of training iterations. [default: 500]
    -L, --learning-rate <learning-rate>                    Learning rate. [default: 0.001]
    -M, --min-error-improvement <min-error-improvement>    Minimum improvement in average error magnitude in a single training iteration (over all words) to keep on training [default:
                                                           0.0001]
    -O, --min-word-occurences <min-word-occurences>        Minimum number of occurences in the corpus a word needs to have in order to be included in the trained vocabulary. [default:
                                                           2]
    -P, --path <path>                                      Where to store the model when training is done.
```

#### Similarity

```
Given a path to a saved Word2Vec model and a target word, finds words in the model's vocab that are similar.

USAGE:
    sloword2vec similar --limit <limit> --path <path> --word <word>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -L, --limit <limit>    Max number of similar entries to show. [default: 20]
    -P, --path <path>      Where to store the model when training is done.
    -W, --word <word>      Word to find similar terms for.
```

#### Add subtract

The classic demo of Word2Vec..

```
Given a number of words to add and to subtract, returns a list of words in that area.

USAGE:
    sloword2vec add-subtract --add <add>... --limit <limit> --path <path> --subtract <subtract>...

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -A, --add <add>...              Words to add encodings for
    -L, --limit <limit>             Max number of similar entries to show. [default: 20]
    -P, --path <path>               Where to store the model when training is done.
    -S, --subtract <subtract>...    Words to subtract encodings for
```

## Details

Pretty much the most naive implementation of Word2Vec, the only special thing being the use of matrix/vector
maths to speed things up.

The linear algebra library behind this lib is [`ndarray`](https://github.com/bluss/rust-ndarray), with
OpenBlas enabled (Fortran and transparent multithreading FTW!).

## Reference material

1. Word2Vec Parameter learning explained [paper](https://arxiv.org/abs/1411.2738)
2. Word2Vec Skip-gram model tutorial [article](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)