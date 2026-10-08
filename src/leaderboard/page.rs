//! The page the leaderboard's Hugging Face Space shows: every row of the
//! Space's leaderboard file, most compliant first, and each method's paired
//! standing ([`super::methods`]). The file is read again for every request,
//! so the page always shows what the file holds; a file that cannot be read
//! or parsed is answered with the reason instead of a stale table.
//!
//! The address to listen on is the caller's: the Space's container states the
//! one its host routes to.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use http::StatusCode;

use super::methods::{compare, Standing};
use super::{parse, Entry};
use crate::Error;

const PAGE: &str = include_str!("page.html");

/// The leaderboard file and the address its page is served on.
pub struct Board {
    pub csv: PathBuf,
    pub listen: String,
}

fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#x27;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Every row of the file, refused with the file's name when it cannot be read.
pub fn read(csv: &Path) -> Result<Vec<Entry>, Error> {
    let text = std::fs::read_to_string(csv).map_err(|source| Error::Io { path: csv.to_path_buf(), source })?;
    parse(&text, &csv.display().to_string())
}

fn cell(text: &str) -> String {
    format!("<td>{}</td>", escape(text))
}

fn model_row(row: &Entry) -> String {
    let sample = match &row.sample_responses_url {
        Some(url) => format!("<td><a href=\"{}\">responses</a></td>", escape(url)),
        None => "<td></td>".to_owned(),
    };
    format!(
        "<tr><td class=\"rank\"></td>{}{}{}{}{}{}{}{}{}{sample}</tr>",
        cell(&row.model),
        cell(&row.model_family),
        cell(&row.model_size),
        cell(&row.method),
        cell(&row.uncensored_rate.to_string()),
        cell(&row.avg_compliance_score.to_string()),
        cell(&row.total_prompts.to_string()),
        cell(&row.timestamp),
        cell(&row.submitter),
    )
}

fn method_row(standing: &Standing) -> String {
    let delta = match standing.mean_delta {
        Some(delta) => delta.to_string(),
        None => "baseline".to_owned(),
    };
    format!(
        "<tr>{}{}{}{}{}{}{}{}{}</tr>",
        cell(&standing.method),
        cell(&standing.models.to_string()),
        cell(&standing.pairs.to_string()),
        cell(&standing.mean_rate.to_string()),
        cell(&delta),
        cell(&standing.max_rate.to_string()),
        cell(&standing.min_rate.to_string()),
        cell(&standing.mean_compliance_score.to_string()),
        cell(&standing.best_model),
    )
}

/// The page for `rows`.
pub fn render(rows: &[Entry]) -> String {
    let models: String = rows.iter().map(model_row).collect();
    let methods: String = compare(rows).iter().map(method_row).collect();
    PAGE.replace("{{models}}", &models).replace("{{methods}}", &methods)
}

/// The request line's method and path; the headers are read and dropped,
/// since the page takes no input.
fn request(stream: &TcpStream) -> Result<(String, String), String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|error| format!("the request line could not be read: {error}"))?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(path)) = (parts.next(), parts.next()) else {
        return Err(format!("{line:?} is not an HTTP request line"));
    };
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).map_err(|error| format!("a request header could not be read: {error}"))?;
        if header.trim_end().is_empty() {
            return Ok((method.to_owned(), path.to_owned()));
        }
    }
}

fn respond(mut stream: &TcpStream, status: StatusCode, content_type: &str, body: &str) -> std::io::Result<()> {
    let reason = match status.canonical_reason() {
        Some(reason) => reason,
        None => status.as_str(),
    };
    write!(
        stream,
        "HTTP/1.1 {} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        status.as_str(),
        body.len()
    )
}

/// Serve the page until the process is stopped. The file is read once before
/// listening, so a wrong path is refused at start instead of on every visit.
pub fn serve(board: &Board) -> Result<(), Error> {
    let rows = read(&board.csv)?;
    let listener = TcpListener::bind(&board.listen)
        .map_err(|error| Error::Run(format!("the leaderboard page could not listen on {}: {error}", board.listen)))?;
    let address = listener.local_addr().map_err(|error| Error::Run(error.to_string()))?;
    eprintln!("serving {} leaderboard rows from {} at http://{address}", rows.len(), board.csv.display());
    for stream in listener.incoming() {
        let stream = stream.map_err(|error| Error::Run(format!("the leaderboard page stopped accepting: {error}")))?;
        let (method, path) = match request(&stream) {
            Ok(request) => request,
            Err(refusal) => {
                eprintln!("{refusal}");
                continue;
            }
        };
        let written = match (method.as_str(), path.as_str()) {
            ("GET", "/") => match read(&board.csv) {
                Ok(rows) => respond(&stream, StatusCode::OK, "text/html; charset=utf-8", &render(&rows)),
                Err(error) => {
                    eprintln!("{error}");
                    respond(&stream, StatusCode::INTERNAL_SERVER_ERROR, "text/plain; charset=utf-8", &error.to_string())
                }
            },
            _ => respond(&stream, StatusCode::NOT_FOUND, "text/plain; charset=utf-8", &format!("no {method} {path} here")),
        };
        if let Err(error) = written {
            eprintln!("an answer to {method} {path} could not be sent: {error}");
        }
    }
    Ok(())
}
