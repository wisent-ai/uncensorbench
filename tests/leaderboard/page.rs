//! The leaderboard page as the Space runs it: the built `uncensorbench
//! leaderboard serve` over a leaderboard file, read over HTTP. A method is
//! compared only with a baseline of the same family and size; a file that
//! does not exist is refused with its name before anything listens.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

const BINARY: &str = env!("CARGO_BIN_EXE_uncensorbench");

/// A directory of this test's own under cargo's scratch area.
fn scratch(name: &str) -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

/// The loopback address with the port left to the operating system (port
/// zero: https://doc.rust-lang.org/std/net/struct.TcpListener.html#method.bind).
fn any_loopback_port() -> String {
    SocketAddr::from((Ipv4Addr::LOCALHOST, Default::default())).to_string()
}

const ROWS: &str = "\
model,model_family,model_size,method,uncensored_rate,avg_compliance_score,total_prompts,timestamp,submitter,sample_responses_url
base-model,family-a,size-s,none,0.25,0.25,4,2026-10-08T00:00:00Z,tester,
steered-model,family-a,size-s,steering,0.75,0.75,4,2026-10-08T00:00:00Z,tester,https://example.org/responses
lonely-model,family-b,size-t,finetuning,0.5,0.5,4,2026-10-08T00:00:00Z,tester,
";

struct Served(Child);

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn the_page_lists_every_row_and_compares_only_paired_methods() {
    let directory = scratch("leaderboard-page");
    let csv = directory.join("leaderboard.csv");
    std::fs::write(&csv, ROWS).expect("leaderboard file");
    let mut child = Command::new(BINARY)
        .args(["leaderboard", "serve", "--csv"])
        .arg(&csv)
        .args(["--listen", &any_loopback_port()])
        .stderr(Stdio::piped())
        .spawn()
        .expect("uncensorbench starts");
    let mut announced = String::new();
    BufReader::new(child.stderr.take().expect("stderr")).read_line(&mut announced).expect("announcement");
    let served = Served(child);
    let address = announced.rsplit("http://").next().expect("address").trim().to_owned();

    let mut stream = TcpStream::connect(&address).expect("the page answers");
    write!(stream, "GET / HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n").expect("request");
    let mut page = String::new();
    stream.read_to_string(&mut page).expect("page");
    drop(served);

    assert!(page.starts_with("HTTP/1.1 200"), "{page}");
    for model in ["base-model", "steered-model", "lonely-model"] {
        assert!(page.contains(model), "{model} missing from the models table: {page}");
    }
    let (_, methods) = page.split_once("id=\"methods\"").expect("methods table");
    assert!(methods.contains("<td>steering</td>"), "the paired method is compared: {methods}");
    assert!(methods.contains("<td>0.5</td>"), "steering's delta is its rate minus its baseline's: {methods}");
    assert!(!methods.contains("finetuning"), "a method without a same-family baseline is left out: {methods}");
    assert!(page.contains("href=\"https://example.org/responses\""), "the sample responses link is kept");
}

#[test]
fn a_missing_leaderboard_file_is_refused_by_name() {
    let missing = scratch("leaderboard-missing").join("absent.csv");
    let output = Command::new(BINARY)
        .args(["leaderboard", "serve", "--csv"])
        .arg(&missing)
        .args(["--listen", &any_loopback_port()])
        .output()
        .expect("uncensorbench runs");
    assert!(!output.status.success());
    let said = String::from_utf8_lossy(&output.stderr);
    assert!(said.contains("absent.csv"), "the refusal names the file: {said}");
}
