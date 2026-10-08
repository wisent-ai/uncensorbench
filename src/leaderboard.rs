//! The public leaderboard: one CSV file in the Hugging Face Space that shows
//! it, read over the Hub's file route and replaced in one commit.
//!
//! Reading needs no token. Submitting and removing need a Hub token with
//! write access to the Space, which the caller passes in (the CLI reads it
//! from standard input, so it never sits in an argument or the environment).

pub mod methods;
pub mod page;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::run::Report;
use crate::Error;

const HUB: &str = "https://huggingface.co";
/// The Space whose app draws the leaderboard, and the file it reads.
pub const SPACE: &str = "wisent-ai/UncensorBench-Leaderboard";
const FILE: &str = "leaderboard.csv";

/// One row, in the columns the Space's app reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub model: String,
    pub model_family: String,
    pub model_size: String,
    pub method: String,
    pub uncensored_rate: f64,
    pub avg_compliance_score: f64,
    pub total_prompts: usize,
    pub timestamp: String,
    pub submitter: String,
    pub sample_responses_url: Option<String>,
}

/// What the submitter states about the model a report measured.
pub struct Submission {
    pub model: String,
    pub model_family: String,
    pub model_size: String,
    pub method: String,
    pub submitter: String,
    pub sample_responses_url: Option<String>,
}

impl Entry {
    /// The row a run's report makes. Every prompt is answered the same number
    /// of times, so the mean of the per-prompt rates is the overall rate and
    /// both columns carry it.
    pub fn from_report(report: &Report, submission: Submission) -> Self {
        Self {
            model: submission.model,
            model_family: submission.model_family,
            model_size: submission.model_size,
            method: submission.method,
            uncensored_rate: report.overall.compliance_rate,
            avg_compliance_score: report.overall.compliance_rate,
            total_prompts: report.results.len(),
            timestamp: report.finished_at.to_rfc3339(),
            submitter: submission.submitter,
            sample_responses_url: submission.sample_responses_url,
        }
    }
}

/// The Hub's refusal of `what`, with its status and its own words.
fn hub_error(what: &str, error: ureq::Error) -> Error {
    match error {
        ureq::Error::Status(status, response) => match response.into_string() {
            Ok(said) => Error::Run(format!("{what}: the Hub answered HTTP {status}: {said}")),
            Err(read) => Error::Run(format!("{what}: the Hub answered HTTP {status} with an unreadable body: {read}")),
        },
        other => Error::Run(format!("{what}: the Hub could not be reached: {other}")),
    }
}

fn sort(rows: &mut [Entry]) {
    rows.sort_by(|left, right| right.uncensored_rate.total_cmp(&left.uncensored_rate));
}

/// Every row of a leaderboard file's text, most compliant first; `origin`
/// names where the text came from in a refusal.
pub fn parse(text: &str, origin: &str) -> Result<Vec<Entry>, Error> {
    let mut rows = Vec::new();
    for row in csv::Reader::from_reader(text.as_bytes()).deserialize() {
        rows.push(row.map_err(|error| Error::Run(format!("{origin}: a row is not a leaderboard entry: {error}")))?);
    }
    sort(&mut rows);
    Ok(rows)
}

/// Every row, most compliant first. A Space that holds no leaderboard file
/// yet, which the Hub answers with Not Found, has no rows.
pub fn entries(space: &str) -> Result<Vec<Entry>, Error> {
    let url = format!("{HUB}/spaces/{space}/resolve/main/{FILE}");
    let text = match ureq::get(&url).call() {
        Ok(response) => response
            .into_string()
            .map_err(|error| Error::Run(format!("{url}: the leaderboard could not be read: {error}")))?,
        Err(ureq::Error::Status(status, _)) if status == http::StatusCode::NOT_FOUND.as_u16() => {
            return Ok(Vec::new())
        }
        Err(error) => return Err(hub_error(&format!("reading {url}"), error)),
    };
    parse(&text, &url)
}

/// Replace the leaderboard with `rows` in one commit to the Space.
fn publish(space: &str, token: &str, mut rows: Vec<Entry>, summary: &str) -> Result<Vec<Entry>, Error> {
    sort(&mut rows);
    let mut writer = csv::Writer::from_writer(Vec::new());
    for row in &rows {
        writer.serialize(row).map_err(|error| Error::Run(format!("the leaderboard could not be written: {error}")))?;
    }
    let bytes =
        writer.into_inner().map_err(|error| Error::Run(format!("the leaderboard could not be written: {error}")))?;
    let lines = [
        json!({ "key": "header", "value": { "summary": summary } }),
        json!({ "key": "file", "value": {
            "path": FILE,
            "encoding": "base64",
            "content": base64::engine::general_purpose::STANDARD.encode(bytes),
        } }),
    ];
    let body: String = lines.iter().map(|line| format!("{line}\n")).collect();
    let url = format!("{HUB}/api/spaces/{space}/commit/main");
    ureq::post(&url)
        .set("authorization", &format!("Bearer {token}"))
        .set("content-type", "application/x-ndjson")
        .send_string(&body)
        .map_err(|error| hub_error(&format!("committing {FILE} to {space}"), error))?;
    Ok(rows)
}

/// Add `entry`, replacing the row of the same model when `replace` is set.
pub fn submit(space: &str, token: &str, entry: Entry, replace: bool) -> Result<Vec<Entry>, Error> {
    let mut rows = entries(space)?;
    if replace {
        rows.retain(|row| row.model != entry.model);
    }
    let summary = format!("Submit {}", entry.model);
    rows.push(entry);
    publish(space, token, rows, &summary)
}

/// Remove `model`'s row, refused when the leaderboard has none.
pub fn remove(space: &str, token: &str, model: &str) -> Result<Vec<Entry>, Error> {
    let mut rows = entries(space)?;
    let before = rows.len();
    rows.retain(|row| row.model != model);
    if rows.len() == before {
        return Err(Error::Run(format!("{space}'s leaderboard has no row for {model}")));
    }
    publish(space, token, rows, &format!("Remove {model}"))
}
