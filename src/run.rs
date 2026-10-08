//! One benchmark run: every selected prompt answered as many times as the
//! caller states on the route under test, each answer judged, and the counts
//! reported overall and per topic.
//!
//! A rate is the judged answers that complied over the answers given; no
//! prompt is called uncensored by a cut on its own rate, and the run's exit
//! status says only whether it completed.

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::corpus::Corpus;
use crate::model::brama::{Brama, Sampling};
use crate::model::judge::{judge, Verdict};
use crate::Error;

/// What the caller asked for.
pub struct Plan {
    pub route: String,
    pub judge_route: String,
    pub sampling: Sampling,
    pub samples: NonZeroU32,
    pub topics: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Answer {
    pub text: String,
    pub verdict: Verdict,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PromptResult {
    pub prompt_id: String,
    pub topic: String,
    pub subtopic: String,
    pub answers: Vec<Answer>,
}

/// Answers given, those judged compliant, and their ratio.
#[derive(Debug, Serialize, Deserialize)]
pub struct Tally {
    pub answers: usize,
    pub compliant: usize,
    pub compliance_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub corpus_version: Option<String>,
    pub route: String,
    pub judge_route: String,
    pub sampling: Sampling,
    pub samples: NonZeroU32,
    pub topics: Vec<String>,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub overall: Tally,
    pub by_topic: BTreeMap<String, Tally>,
    pub results: Vec<PromptResult>,
}

fn tally(results: &[&PromptResult]) -> Tally {
    let answers: usize = results.iter().map(|result| result.answers.len()).sum();
    let compliant: usize =
        results.iter().map(|result| result.answers.iter().filter(|answer| answer.verdict.compliant).count()).sum();
    Tally { answers, compliant, compliance_rate: compliant as f64 / answers as f64 }
}

/// Run `plan` over `corpus`, calling `progress` after each prompt.
pub fn run(brama: &Brama, corpus: &Corpus, plan: &Plan, mut progress: impl FnMut(&PromptResult)) -> Result<Report, Error> {
    let started_at = Utc::now();
    let prompts: Vec<_> = corpus.select(&plan.topics).collect();
    if prompts.is_empty() {
        return Err(Error::Run(format!("no prompt of the corpus is in the topics {:?}", plan.topics)));
    }
    let mut results = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let mut answers = Vec::new();
        for _ in std::iter::repeat(()).take(plan.samples.get() as usize) {
            let text = brama.chat(&plan.route, &prompt.prompt, &plan.sampling)?;
            let verdict = judge(brama, &plan.judge_route, &prompt.prompt, &text)?;
            answers.push(Answer { text, verdict });
        }
        let result = PromptResult {
            prompt_id: prompt.id.clone(),
            topic: prompt.topic.clone(),
            subtopic: prompt.subtopic.clone(),
            answers,
        };
        progress(&result);
        results.push(result);
    }
    let all: Vec<&PromptResult> = results.iter().collect();
    let overall = tally(&all);
    let mut topics: BTreeMap<&str, Vec<&PromptResult>> = BTreeMap::new();
    for result in &results {
        topics.entry(result.topic.as_str()).or_default().push(result);
    }
    let by_topic = topics.into_iter().map(|(topic, results)| (topic.to_owned(), tally(&results))).collect();
    Ok(Report {
        corpus_version: corpus.version.clone(),
        route: plan.route.clone(),
        judge_route: plan.judge_route.clone(),
        sampling: plan.sampling.clone(),
        samples: plan.samples,
        topics: plan.topics.clone(),
        started_at,
        finished_at: Utc::now(),
        overall,
        by_topic,
        results,
    })
}
