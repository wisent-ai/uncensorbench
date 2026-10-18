//! The words after a command, and how an answer is written or printed.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

use serde_json::Value;
use uncensorbench::corpus::Corpus;
use uncensorbench::run::Report;

/// The words after the command: positionals, `--name value` pairs, and
/// switches. An option followed by nothing or by another option is a switch.
pub(crate) struct Words {
    pub(crate) positionals: Vec<String>,
    values: BTreeMap<String, Vec<String>>,
    switches: BTreeSet<String>,
}

pub(crate) fn words(args: &[String]) -> Words {
    let mut parsed = Words {
        positionals: Vec::new(),
        values: BTreeMap::new(),
        switches: BTreeSet::new(),
    };
    let mut rest = args.iter().peekable();
    while let Some(word) = rest.next() {
        let Some(name) = word.strip_prefix("--") else {
            parsed.positionals.push(word.clone());
            continue;
        };
        match rest.next_if(|next| !next.starts_with("--")) {
            Some(value) => parsed
                .values
                .entry(name.to_owned())
                .or_default()
                .push(value.clone()),
            None => {
                parsed.switches.insert(name.to_owned());
            }
        }
    }
    parsed
}

impl Words {
    pub(crate) fn one(&self, name: &str) -> Result<Option<&str>, String> {
        if self.switches.contains(name) {
            return Err(format!("--{name} needs a value"));
        }
        match self.values.get(name).map(Vec::as_slice) {
            None => Ok(None),
            Some([value]) => Ok(Some(value.as_str())),
            Some(_) => Err(format!("--{name} is given more than once")),
        }
    }
    pub(crate) fn required(&self, name: &str) -> Result<&str, String> {
        self.one(name)?
            .ok_or_else(|| format!("--{name} is required"))
    }
    pub(crate) fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        self.one(name)?
            .map(|raw| {
                raw.parse()
                    .map_err(|_| format!("--{name} takes a number, not {raw:?}"))
            })
            .transpose()
    }
    /// A switch, refused when it was given a value.
    pub(crate) fn switch(&self, name: &str) -> Result<bool, String> {
        if self.values.contains_key(name) {
            return Err(format!("--{name} takes no value"));
        }
        Ok(self.switches.contains(name))
    }
    /// Every value of a repeatable option; none given selects nothing by it.
    pub(crate) fn many(&self, name: &str) -> Vec<String> {
        match self.values.get(name) {
            Some(values) => values.clone(),
            None => Vec::new(),
        }
    }
    pub(crate) fn corpus(&self) -> Result<Corpus, String> {
        Corpus::load(self.one("prompts")?.map(std::path::Path::new))
            .map_err(|error| error.to_string())
    }
}

pub(crate) fn hub_token() -> Result<String, String> {
    let mut token = String::new();
    std::io::stdin()
        .read_to_string(&mut token)
        .map_err(|error| format!("the Hub token could not be read from standard input: {error}"))?;
    let token = token.trim().to_owned();
    if token.is_empty() {
        return Err(
            "the Hub token is read from standard input, and standard input was empty".into(),
        );
    }
    Ok(token)
}

pub(crate) fn write_json(path: &str, value: &impl serde::Serialize) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    std::fs::write(path, text + "\n").map_err(|error| format!("{path}: {error}"))
}

/// A run report read from `path`.
pub(crate) fn report(path: &str) -> Result<Report, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("{path} is not a run report: {error}"))
}

/// `key: value` lines for a person; nested values as compact JSON.
pub(crate) fn print_text(value: &Value) {
    match value {
        Value::Object(fields) => fields
            .iter()
            .for_each(|(key, field)| println!("{key}: {field}")),
        Value::Array(items) => items.iter().for_each(|item| println!("{item}")),
        other => println!("{other}"),
    }
}
