//! What answers each command.

use std::num::NonZeroU32;
use std::path::Path;

use serde_json::{json, Value};
use uncensorbench::leaderboard::{self, Entry, Submission};
use uncensorbench::model::brama::{Brama, Sampling};
use uncensorbench::run::{run, Plan};

use super::words::{hub_token, report, write_json, Words};

pub(crate) fn info(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    Ok(json!({
        "corpus_version": corpus.version,
        "prompts": corpus.prompts.len(),
        "topics": corpus.topics().keys().collect::<Vec<_>>(),
    }))
}

pub(crate) fn topics(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    let topics: Vec<Value> = corpus
        .topics()
        .into_iter()
        .map(|(topic, (prompts, subtopics))| {
            json!({ "topic": topic, "prompts": prompts, "subtopics": subtopics })
        })
        .collect();
    Ok(Value::Array(topics))
}

pub(crate) fn list(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    let topics = words.many("topic");
    Ok(Value::Array(
        corpus
            .select(&topics)
            .map(|prompt| {
                json!({
                    "id": prompt.id,
                    "topic": prompt.topic,
                    "subtopic": prompt.subtopic,
                    "prompt": prompt.prompt,
                })
            })
            .collect(),
    ))
}

pub(crate) fn export(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    let topics = words.many("topic");
    let output = words.required("output")?;
    let pairs = words.switch("pairs")?;
    let written = if pairs {
        let selected = corpus.contrastive_pairs(&topics);
        write_json(output, &selected)?;
        selected.len()
    } else {
        let selected: Vec<_> = corpus.select(&topics).collect();
        write_json(output, &selected)?;
        selected.len()
    };
    Ok(json!({ "file": output, "written": written, "pairs": pairs }))
}

pub(crate) fn run_benchmark(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    let samples: NonZeroU32 = words.number("samples")?.ok_or(
        "--samples is required: how many answers each prompt gets is the caller's to state",
    )?;
    let plan = Plan {
        route: words.required("route")?.to_owned(),
        judge_route: words.required("judge-route")?.to_owned(),
        sampling: Sampling {
            temperature: words.number("temperature")?,
            top_p: words.number("top-p")?,
            max_tokens: words.number("max-tokens")?,
        },
        samples,
        topics: words.many("topic"),
    };
    let output = words.required("output")?;
    let brama = Brama::from_env().map_err(|error| error.to_string())?;
    let report = run(&brama, &corpus, &plan, |result| {
        let compliant = result
            .answers
            .iter()
            .filter(|answer| answer.verdict.compliant)
            .count();
        eprintln!(
            "{}: {compliant} of {} answers complied",
            result.prompt_id,
            result.answers.len()
        );
    })
    .map_err(|error| error.to_string())?;
    write_json(output, &report)?;
    Ok(json!({ "file": output, "overall": report.overall, "by_topic": report.by_topic }))
}

fn submission(words: &Words) -> Result<Submission, String> {
    Ok(Submission {
        model: words.required("model")?.to_owned(),
        model_family: words.required("model-family")?.to_owned(),
        model_size: words.required("model-size")?.to_owned(),
        method: words.required("method")?.to_owned(),
        submitter: words.required("submitter")?.to_owned(),
        sample_responses_url: words.one("sample-responses-url")?.map(str::to_owned),
    })
}

pub(crate) fn leaderboard_command(words: &Words) -> Result<Value, String> {
    let rows = match words.positionals.first().map(String::as_str) {
        Some("show") => leaderboard::entries(leaderboard::SPACE),
        Some("methods") => {
            let rows =
                leaderboard::entries(leaderboard::SPACE).map_err(|error| error.to_string())?;
            return serde_json::to_value(leaderboard::methods::compare(&rows))
                .map_err(|error| error.to_string());
        }
        Some("submit") => {
            let entry = Entry::from_report(&report(words.required("report")?)?, submission(words)?);
            let replace = words.switch("replace")?;
            leaderboard::submit(leaderboard::SPACE, &hub_token()?, entry, replace)
        }
        Some("remove") => {
            let model = words.required("model")?;
            leaderboard::remove(leaderboard::SPACE, &hub_token()?, model)
        }
        Some("serve") => {
            let board = leaderboard::page::Board {
                csv: Path::new(words.required("csv")?).to_path_buf(),
                listen: words.required("listen")?.to_owned(),
            };
            leaderboard::page::serve(&board).map_err(|error| error.to_string())?;
            return Ok(json!({ "served": board.csv }));
        }
        _ => {
            return Err(format!(
                "leaderboard takes show, methods, submit, remove or serve\n{}",
                crate::USAGE
            ))
        }
    }
    .map_err(|error| error.to_string())?;
    serde_json::to_value(rows).map_err(|error| error.to_string())
}

/// Serve the labelling page until stopped; every label is written to the
/// labels file as it is given, so stopping loses nothing.
pub(crate) fn label_command(words: &Words) -> Result<Value, String> {
    let responses = words.required("responses")?;
    let labels = words.required("labels")?;
    let session = uncensorbench::label::Session::open(Path::new(responses), Path::new(labels))
        .map_err(|error| error.to_string())?;
    uncensorbench::label::serve(session).map_err(|error| error.to_string())?;
    Ok(json!({ "labels": labels }))
}

/// Write the labelling page's responses file from a run report.
pub(crate) fn responses_command(words: &Words) -> Result<Value, String> {
    let report = report(words.required("report")?)?;
    let output = words.required("output")?;
    let answers = uncensorbench::agreement::responses(&report, &words.corpus()?)
        .map_err(|error| error.to_string())?;
    write_json(output, &answers)?;
    Ok(json!({ "file": output, "written": answers.len() }))
}

/// Judge every labelled answer and write how often the judge agreed.
pub(crate) fn agreement_command(words: &Words) -> Result<Value, String> {
    let path = words.required("labels")?;
    let judge_route = words.required("judge-route")?;
    let output = words.required("output")?;
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let labels: Vec<uncensorbench::label::Label> = serde_json::from_str(&text)
        .map_err(|error| format!("{path} is not a list of labels: {error}"))?;
    let brama = Brama::from_env().map_err(|error| error.to_string())?;
    let agreement =
        uncensorbench::agreement::compare(&brama, judge_route, &labels, |label, compliant| {
            eprintln!(
                "{}: person {:?}, judge compliant {compliant}",
                label.id, label.human_label
            );
        })
        .map_err(|error| error.to_string())?;
    write_json(output, &agreement)?;
    Ok(json!({
        "file": output,
        "compared": agreement.compared,
        "agreed": agreement.agreed,
        "agreement_rate": agreement.agreement_rate,
        "partial": agreement.partial.len(),
    }))
}
