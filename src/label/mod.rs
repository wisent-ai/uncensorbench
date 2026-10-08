//! Human labels for a run's responses: a page on this machine where a person
//! marks each answer a refusal, not a refusal, or partial, and every mark is
//! written to the labels file at once.
//!
//! The server listens on a port the operating system assigns and prints its
//! address; no port is chosen here, so two labelling sessions never collide
//! and none fails on a port something else holds. A responses file whose
//! entry lacks a field the page shows is refused with the field and where in
//! the file it is missing.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use http::StatusCode;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

use crate::Error;

const PAGE: &str = include_str!("page.html");
const STYLE: &str = include_str!("labeling.css");

/// What a person decided about one answer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Refusal,
    NotRefusal,
    Partial,
}

/// One answer to label, as a run's responses file holds it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Answer {
    pub id: String,
    pub topic: String,
    pub subtopic: String,
    pub prompt: String,
    pub response: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub censored_reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncensored_reference: Option<String>,
}

/// One label as the labels file holds it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Label {
    pub id: String,
    pub topic: String,
    pub subtopic: String,
    pub prompt: String,
    pub response: String,
    pub human_label: Verdict,
    pub is_refusal: bool,
}

/// One labelling session: the answers, the labels already given, and the
/// answer on screen (its index in the responses file).
pub struct Session {
    answers: Vec<Answer>,
    labels: Vec<Label>,
    position: usize,
    output: PathBuf,
}

fn read_list<T: DeserializeOwned>(path: &Path, what: &str) -> Result<Vec<T>, Error> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_owned(), source })?;
    serde_json::from_str(&text).map_err(|error| Error::Run(format!("{}: not a {what}: {error}", path.display())))
}

impl Session {
    /// Read `responses` and, when it exists, the labels file `output`. The
    /// first answer is on screen.
    pub fn open(responses: &Path, output: &Path) -> Result<Session, Error> {
        let answers = read_list(responses, "list of responses")?;
        let labels = if output.exists() { read_list(output, "list of labels")? } else { Vec::new() };
        Ok(Session { answers, labels, position: Default::default(), output: output.to_owned() })
    }

    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.answers.iter().enumerate().map(|(at, _)| at)
    }

    fn after(&self) -> Option<usize> {
        self.indices().find(|at| *at > self.position)
    }

    fn before(&self) -> Option<usize> {
        self.indices().filter(|at| *at < self.position).last()
    }

    fn count(&self, verdict: Verdict) -> usize {
        self.labels.iter().filter(|label| label.human_label == verdict).count()
    }

    /// What the page shows. `position` counts from the first answer.
    pub fn state(&self) -> Value {
        json!({
            "current": self.answers.get(self.position),
            "position": self.answers.get(..=self.position).map(<[Answer]>::len),
            "total": self.answers.len(),
            "labeled": self.labels.len(),
            "has_previous": self.before().is_some(),
            "has_next": self.after().is_some(),
            "counts": {
                "refusal": self.count(Verdict::Refusal),
                "not_refusal": self.count(Verdict::NotRefusal),
                "partial": self.count(Verdict::Partial),
            },
        })
    }

    /// Label the answer on screen, replacing an earlier label of it, write
    /// every label to the labels file, and move to the next answer.
    pub fn label(&mut self, verdict: Verdict) -> Result<(), Error> {
        let Some(answer) = self.answers.get(self.position) else {
            return Err(Error::Run("there is no answer on screen to label".into()));
        };
        self.labels.retain(|label| label.id != answer.id);
        self.labels.push(Label {
            id: answer.id.clone(),
            topic: answer.topic.clone(),
            subtopic: answer.subtopic.clone(),
            prompt: answer.prompt.clone(),
            response: answer.response.clone(),
            human_label: verdict,
            is_refusal: verdict == Verdict::Refusal,
        });
        let text = serde_json::to_string_pretty(&self.labels).map_err(|error| Error::Run(error.to_string()))?;
        std::fs::write(&self.output, text + "\n").map_err(|source| Error::Io { path: self.output.clone(), source })?;
        self.next();
        Ok(())
    }

    /// Show the next answer; on the last one nothing moves.
    pub fn next(&mut self) {
        if let Some(at) = self.after() {
            self.position = at;
        }
    }

    /// Show the previous answer; on the first one nothing moves.
    pub fn previous(&mut self) {
        if let Some(at) = self.before() {
            self.position = at;
        }
    }
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|error| format!("the request line could not be read: {error}"))?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(path)) = (parts.next(), parts.next()) else {
        return Err(format!("{line:?} is not an HTTP request line"));
    };
    let mut length: Option<u64> = None;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).map_err(|error| format!("a request header could not be read: {error}"))?;
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case(http::header::CONTENT_LENGTH.as_str()) {
                length =
                    Some(value.trim().parse().map_err(|_| format!("Content-Length {value:?} is not a length"))?);
            }
        }
    }
    let mut body = Vec::new();
    if let Some(length) = length {
        reader
            .take(length)
            .read_to_end(&mut body)
            .map_err(|error| format!("the request body could not be read: {error}"))?;
    }
    Ok(Request { method: method.to_owned(), path: path.to_owned(), body })
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

/// The answer to one request: its status and JSON body.
fn act(session: &mut Session, request: &Request) -> (StatusCode, Value) {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/state") => (StatusCode::OK, session.state()),
        ("POST", "/next") => {
            session.next();
            (StatusCode::OK, session.state())
        }
        ("POST", "/previous") => {
            session.previous();
            (StatusCode::OK, session.state())
        }
        ("POST", "/label") => {
            #[derive(Deserialize)]
            struct Body {
                label: Verdict,
            }
            match serde_json::from_slice::<Body>(&request.body) {
                Err(error) => (StatusCode::BAD_REQUEST, json!({ "error": format!("not a label: {error}") })),
                Ok(Body { label }) => match session.label(label) {
                    Ok(()) => (StatusCode::OK, session.state()),
                    Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, json!({ "error": error.to_string() })),
                },
            }
        }
        (method, path) => (StatusCode::NOT_FOUND, json!({ "error": format!("no {method} {path} here") })),
    }
}

/// Serve the labelling page until the process is stopped. Every label is
/// already in the labels file when the page shows it.
pub fn serve(mut session: Session) -> Result<(), Error> {
    // Port 0 asks the operating system for a free port:
    // https://doc.rust-lang.org/std/net/struct.TcpListener.html#method.bind
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| Error::Run(format!("no local port could be opened for the labelling page: {error}")))?;
    let address = listener.local_addr().map_err(|error| Error::Run(error.to_string()))?;
    eprintln!(
        "labelling {} answers ({} already labelled) at http://{address}; every label is written to {} as it is given",
        session.answers.len(),
        session.labels.len(),
        session.output.display()
    );
    for stream in listener.incoming() {
        let stream = stream.map_err(|error| Error::Run(format!("the labelling page stopped accepting: {error}")))?;
        let request = match read_request(&stream) {
            Ok(request) => request,
            Err(refusal) => {
                eprintln!("{refusal}");
                continue;
            }
        };
        let written = match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/") => respond(&stream, StatusCode::OK, "text/html; charset=utf-8", PAGE),
            ("GET", "/labeling.css") => respond(&stream, StatusCode::OK, "text/css", STYLE),
            _ => {
                let (status, body) = act(&mut session, &request);
                respond(&stream, status, "application/json", &body.to_string())
            }
        };
        if let Err(error) = written {
            eprintln!("an answer to {} {} could not be sent: {error}", request.method, request.path);
        }
    }
    Ok(())
}
