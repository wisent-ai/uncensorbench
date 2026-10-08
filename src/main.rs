//! The `uncensorbench` command line over the library: the corpus read
//! offline (`info`, `topics`, `list`, `export`), one run through Brama
//! (`run`), and the public leaderboard (`leaderboard show|submit|remove`).
//!
//! Every answer is one JSON document on standard output, or `key: value`
//! lines with `--text`. Every refusal names what is missing and where it was
//! looked for, and exits with a failure status. A Hub token is read from
//! standard input, never from an argument or the environment.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::num::NonZeroU32;
use std::path::Path;
use std::process::ExitCode;

use serde_json::{json, Value};
use uncensorbench::corpus::Corpus;
use uncensorbench::leaderboard::{self, Entry, Submission};
use uncensorbench::model::brama::{Brama, Sampling};
use uncensorbench::run::{run, Plan, Report};

const USAGE: &str = "usage (positionals first, then options):
  uncensorbench info [--prompts FILE] [--text]
  uncensorbench topics [--prompts FILE] [--text]
  uncensorbench list [--prompts FILE] [--topic T]... [--text]
  uncensorbench export [--prompts FILE] [--topic T]... --output FILE [--pairs]
  uncensorbench run --route R --judge-route J --samples N --output FILE [--topic T]...
                    [--temperature X] [--top-p X] [--max-tokens N] [--prompts FILE]
  uncensorbench leaderboard show [--text]
  uncensorbench leaderboard methods [--text]
  uncensorbench leaderboard submit --report FILE --model M --model-family F --model-size S
                    --method M --submitter NAME [--sample-responses-url URL] [--replace] < HUB_TOKEN
  uncensorbench leaderboard remove --model M < HUB_TOKEN
  uncensorbench leaderboard serve --csv FILE --listen ADDRESS
                    (serves the leaderboard page over FILE until stopped)
  uncensorbench label --responses FILE --labels FILE
                    (serves a labelling page on a port the system assigns and prints its address)
  uncensorbench responses --report FILE --output FILE [--prompts FILE]
                    (the labelling page's responses file from a run report)
  uncensorbench agreement --labels FILE --judge-route J --output FILE
                    (judges every labelled answer and counts where the judge agrees with the person)";

/// The words after the command: positionals, `--name value` pairs, and
/// switches. An option followed by nothing or by another option is a switch.
struct Words {
    positionals: Vec<String>,
    values: BTreeMap<String, Vec<String>>,
    switches: BTreeSet<String>,
}

fn words(args: &[String]) -> Words {
    let mut parsed = Words { positionals: Vec::new(), values: BTreeMap::new(), switches: BTreeSet::new() };
    let mut rest = args.iter().peekable();
    while let Some(word) = rest.next() {
        let Some(name) = word.strip_prefix("--") else {
            parsed.positionals.push(word.clone());
            continue;
        };
        match rest.next_if(|next| !next.starts_with("--")) {
            Some(value) => parsed.values.entry(name.to_owned()).or_default().push(value.clone()),
            None => {
                parsed.switches.insert(name.to_owned());
            }
        }
    }
    parsed
}

impl Words {
    fn one(&self, name: &str) -> Result<Option<&str>, String> {
        if self.switches.contains(name) {
            return Err(format!("--{name} needs a value"));
        }
        match self.values.get(name).map(Vec::as_slice) {
            None => Ok(None),
            Some([value]) => Ok(Some(value.as_str())),
            Some(_) => Err(format!("--{name} is given more than once")),
        }
    }
    fn required(&self, name: &str) -> Result<&str, String> {
        self.one(name)?.ok_or_else(|| format!("--{name} is required"))
    }
    fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        self.one(name)?
            .map(|raw| raw.parse().map_err(|_| format!("--{name} takes a number, not {raw:?}")))
            .transpose()
    }
    /// A switch, refused when it was given a value.
    fn switch(&self, name: &str) -> Result<bool, String> {
        if self.values.contains_key(name) {
            return Err(format!("--{name} takes no value"));
        }
        Ok(self.switches.contains(name))
    }
    /// Every value of a repeatable option; none given selects nothing by it.
    fn many(&self, name: &str) -> Vec<String> {
        match self.values.get(name) {
            Some(values) => values.clone(),
            None => Vec::new(),
        }
    }
    fn corpus(&self) -> Result<Corpus, String> {
        Corpus::load(self.one("prompts")?.map(std::path::Path::new)).map_err(|error| error.to_string())
    }
}

fn hub_token() -> Result<String, String> {
    let mut token = String::new();
    std::io::stdin()
        .read_to_string(&mut token)
        .map_err(|error| format!("the Hub token could not be read from standard input: {error}"))?;
    let token = token.trim().to_owned();
    if token.is_empty() {
        return Err("the Hub token is read from standard input, and standard input was empty".into());
    }
    Ok(token)
}

fn write_json(path: &str, value: &impl serde::Serialize) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    std::fs::write(path, text + "\n").map_err(|error| format!("{path}: {error}"))
}

fn info(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    Ok(json!({
        "corpus_version": corpus.version,
        "prompts": corpus.prompts.len(),
        "topics": corpus.topics().keys().collect::<Vec<_>>(),
    }))
}

fn topics(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    let topics: Vec<Value> = corpus
        .topics()
        .into_iter()
        .map(|(topic, (prompts, subtopics))| json!({ "topic": topic, "prompts": prompts, "subtopics": subtopics }))
        .collect();
    Ok(Value::Array(topics))
}

fn list(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    let topics = words.many("topic");
    Ok(Value::Array(
        corpus
            .select(&topics)
            .map(|prompt| json!({ "id": prompt.id, "topic": prompt.topic, "subtopic": prompt.subtopic, "prompt": prompt.prompt }))
            .collect(),
    ))
}

fn export(words: &Words) -> Result<Value, String> {
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

fn run_benchmark(words: &Words) -> Result<Value, String> {
    let corpus = words.corpus()?;
    let samples: NonZeroU32 = words
        .number("samples")?
        .ok_or("--samples is required: how many answers each prompt gets is the caller's to state")?;
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
        let compliant = result.answers.iter().filter(|answer| answer.verdict.compliant).count();
        eprintln!("{}: {compliant} of {} answers complied", result.prompt_id, result.answers.len());
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

fn leaderboard_command(words: &Words) -> Result<Value, String> {
    let rows = match words.positionals.first().map(String::as_str) {
        Some("show") => leaderboard::entries(leaderboard::SPACE),
        Some("methods") => {
            let rows = leaderboard::entries(leaderboard::SPACE).map_err(|error| error.to_string())?;
            return serde_json::to_value(leaderboard::methods::compare(&rows)).map_err(|error| error.to_string());
        }
        Some("submit") => {
            let path = words.required("report")?;
            let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
            let report: Report =
                serde_json::from_str(&text).map_err(|error| format!("{path} is not a run report: {error}"))?;
            let entry = Entry::from_report(&report, submission(words)?);
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
        _ => return Err(format!("leaderboard takes show, methods, submit, remove or serve\n{USAGE}")),
    }
    .map_err(|error| error.to_string())?;
    serde_json::to_value(rows).map_err(|error| error.to_string())
}

/// Serve the labelling page until stopped; every label is written to the
/// labels file as it is given, so stopping loses nothing.
fn label_command(words: &Words) -> Result<Value, String> {
    let responses = words.required("responses")?;
    let labels = words.required("labels")?;
    let session =
        uncensorbench::label::Session::open(Path::new(responses), Path::new(labels)).map_err(|error| error.to_string())?;
    uncensorbench::label::serve(session).map_err(|error| error.to_string())?;
    Ok(json!({ "labels": labels }))
}

/// A run report read from `path`.
fn report(path: &str) -> Result<Report, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("{path} is not a run report: {error}"))
}

/// Write the labelling page's responses file from a run report.
fn responses_command(words: &Words) -> Result<Value, String> {
    let report = report(words.required("report")?)?;
    let output = words.required("output")?;
    let answers =
        uncensorbench::agreement::responses(&report, &words.corpus()?).map_err(|error| error.to_string())?;
    write_json(output, &answers)?;
    Ok(json!({ "file": output, "written": answers.len() }))
}

/// Judge every labelled answer and write how often the judge agreed.
fn agreement_command(words: &Words) -> Result<Value, String> {
    let path = words.required("labels")?;
    let judge_route = words.required("judge-route")?;
    let output = words.required("output")?;
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let labels: Vec<uncensorbench::label::Label> =
        serde_json::from_str(&text).map_err(|error| format!("{path} is not a list of labels: {error}"))?;
    let brama = Brama::from_env().map_err(|error| error.to_string())?;
    let agreement = uncensorbench::agreement::compare(&brama, judge_route, &labels, |label, compliant| {
        eprintln!("{}: person {:?}, judge compliant {compliant}", label.id, label.human_label);
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

/// `key: value` lines for a person; nested values as compact JSON.
fn print_text(value: &Value) {
    match value {
        Value::Object(fields) => fields.iter().for_each(|(key, field)| println!("{key}: {field}")),
        Value::Array(items) => items.iter().for_each(|item| println!("{item}")),
        other => println!("{other}"),
    }
}

/// One command: its name and what answers it.
struct Command {
    name: &'static str,
    answer: fn(&Words) -> Result<Value, String>,
}

/// Every command. `stado release version-gate app-surface --command-table
/// src/main.rs:COMMANDS` reads the names from here, so the released surface
/// and the dispatch are one list.
static COMMANDS: &[Command] = &[
    Command { name: "info", answer: info },
    Command { name: "topics", answer: topics },
    Command { name: "list", answer: list },
    Command { name: "export", answer: export },
    Command { name: "run", answer: run_benchmark },
    Command { name: "leaderboard", answer: leaderboard_command },
    Command { name: "label", answer: label_command },
    Command { name: "responses", answer: responses_command },
    Command { name: "agreement", answer: agreement_command },
];

fn answer(command: &str, rest: &[String]) -> Result<(Value, bool), String> {
    let words = words(rest);
    let Some(known) = COMMANDS.iter().find(|known| known.name == command) else {
        return Err(format!("unknown command {command}\n{USAGE}"));
    };
    let value = (known.answer)(&words)?;
    Ok((value, words.switch("text")?))
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let Some((command, rest)) = argv.split_first().and_then(|(_program, args)| args.split_first()) else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    if matches!(command.as_str(), "--help" | "-h" | "help") || rest.iter().any(|word| word == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let printed = answer(command, rest).and_then(|(value, text)| {
        if text {
            print_text(&value);
            return Ok(());
        }
        let json = serde_json::to_string_pretty(&value)
            .map_err(|error| format!("the answer could not be printed: {error}"))?;
        println!("{json}");
        Ok(())
    });
    match printed {
        Ok(()) => ExitCode::SUCCESS,
        Err(refusal) => {
            eprintln!("{refusal}");
            ExitCode::FAILURE
        }
    }
}
