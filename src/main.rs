//! The `uncensorbench` command line over the library: the corpus read
//! offline (`info`, `topics`, `list`, `export`), one run through Brama
//! (`run`), and the public leaderboard (`leaderboard show|submit|remove`).
//!
//! Every answer is one JSON document on standard output, or `key: value`
//! lines with `--text`. Every refusal names what is missing and where it was
//! looked for, and exits with a failure status. A Hub token is read from
//! standard input, never from an argument or the environment.

mod cli;

use std::process::ExitCode;

use cli::commands::{
    agreement_command, export, info, label_command, leaderboard_command, list, responses_command,
    run_benchmark, topics,
};
use cli::words::{print_text, words, Words};
use serde_json::Value;

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

/// One command: its name and what answers it.
struct Command {
    name: &'static str,
    answer: fn(&Words) -> Result<Value, String>,
}

/// Every command. `stado release version-gate app-surface --command-table
/// src/main.rs:COMMANDS` reads the names from here, so the released surface
/// and the dispatch are one list.
static COMMANDS: &[Command] = &[
    Command {
        name: "info",
        answer: info,
    },
    Command {
        name: "topics",
        answer: topics,
    },
    Command {
        name: "list",
        answer: list,
    },
    Command {
        name: "export",
        answer: export,
    },
    Command {
        name: "run",
        answer: run_benchmark,
    },
    Command {
        name: "leaderboard",
        answer: leaderboard_command,
    },
    Command {
        name: "label",
        answer: label_command,
    },
    Command {
        name: "responses",
        answer: responses_command,
    },
    Command {
        name: "agreement",
        answer: agreement_command,
    },
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
    let Some((command, rest)) = argv
        .split_first()
        .and_then(|(_program, args)| args.split_first())
    else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    if matches!(command.as_str(), "--help" | "-h" | "help")
        || rest.iter().any(|word| word == "--help")
    {
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
