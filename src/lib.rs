//! UncensorBench: how often a model answers prompts it would be expected to
//! refuse. Every answer is generated through Brama on a route the caller
//! names and judged through Brama by a model on another; the rates are counts
//! of judged answers. No sampling preset, score scale, pass mark or keyword
//! list is chosen here: generation settings the caller leaves out are not
//! sent, so the route's own apply, and every one stated is recorded in the
//! report.

pub mod agreement;
pub mod corpus;
pub mod label;
pub mod leaderboard;
pub mod model;
pub mod run;

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("prompt corpus {0}")]
    Corpus(String),
    #[error("Brama is not configured: {0}")]
    Config(String),
    #[error("{0}")]
    Brama(String),
    #[error("{0}")]
    Judge(String),
    #[error("{0}")]
    Run(String),
}
