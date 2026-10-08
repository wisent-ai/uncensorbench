//! The prompt corpus: the bundled `data/prompts.json`, or a file the caller names.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Error;

const BUNDLED: &str = include_str!("../data/prompts.json");

/// One prompt and its reference answers. A reference the corpus does not
/// carry is empty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    pub topic: String,
    pub subtopic: String,
    pub prompt: String,
    #[serde(default)]
    pub censored_response: String,
    #[serde(default)]
    pub uncensored_response: String,
}

/// A censored and an uncensored answer to the same prompt.
#[derive(Clone, Debug, Serialize)]
pub struct ContrastivePair<'a> {
    pub id: &'a str,
    pub topic: &'a str,
    pub prompt: &'a str,
    pub censored: &'a str,
    pub uncensored: &'a str,
}

#[derive(Deserialize)]
struct CorpusFile {
    #[serde(default)]
    version: Option<String>,
    prompts: Vec<Prompt>,
}

#[derive(Clone, Debug)]
pub struct Corpus {
    /// The corpus version the file declares, if it declares one.
    pub version: Option<String>,
    pub prompts: Vec<Prompt>,
}

impl Corpus {
    pub fn bundled() -> Result<Self, Error> {
        Self::parse(BUNDLED, "bundled data/prompts.json")
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })?;
        Self::parse(&text, &path.display().to_string())
    }

    /// The bundled corpus, or the file at `path`.
    pub fn load(path: Option<&Path>) -> Result<Self, Error> {
        match path {
            Some(path) => Self::from_file(path),
            None => Self::bundled(),
        }
    }

    fn parse(text: &str, origin: &str) -> Result<Self, Error> {
        let file: CorpusFile =
            serde_json::from_str(text).map_err(|e| Error::Corpus(format!("{origin}: {e}")))?;
        Ok(Self { version: file.version, prompts: file.prompts })
    }

    /// Prompts in every given topic; all prompts when `topics` is empty.
    pub fn select<'a>(&'a self, topics: &'a [String]) -> impl Iterator<Item = &'a Prompt> + 'a {
        self.prompts
            .iter()
            .filter(move |prompt| topics.is_empty() || topics.contains(&prompt.topic))
    }

    /// Each topic with its prompt count and its subtopics, sorted by name.
    pub fn topics(&self) -> BTreeMap<&str, (usize, Vec<&str>)> {
        let mut topics: BTreeMap<&str, (usize, Vec<&str>)> = BTreeMap::new();
        for prompt in &self.prompts {
            let entry = topics.entry(&prompt.topic).or_default();
            entry.0 += 1;
            if !entry.1.contains(&prompt.subtopic.as_str()) {
                entry.1.push(&prompt.subtopic);
            }
        }
        topics.values_mut().for_each(|(_, subtopics)| subtopics.sort_unstable());
        topics
    }

    /// The pairs of every selected prompt that carries both reference answers.
    pub fn contrastive_pairs<'a>(&'a self, topics: &'a [String]) -> Vec<ContrastivePair<'a>> {
        self.select(topics)
            .filter(|p| !p.censored_response.is_empty() && !p.uncensored_response.is_empty())
            .map(|p| ContrastivePair {
                id: &p.id,
                topic: &p.topic,
                prompt: &p.prompt,
                censored: &p.censored_response,
                uncensored: &p.uncensored_response,
            })
            .collect()
    }
}
