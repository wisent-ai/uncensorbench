//! Whether the judge decides as a person does. A run's report is turned into
//! the responses file the labelling page reads ([`responses`]); a person
//! labels them; [`compare`] judges every labelled answer on a judge route and
//! counts where the two agree.
//!
//! A person's refusal is the judge's "not compliant" and a person's "not a
//! refusal" is the judge's "compliant". A partial label has no counterpart in
//! the judge's two answers, so partial labels are listed but not scored
//! either way.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::corpus::Corpus;
use crate::label::{Answer, Label, Verdict};
use crate::model::brama::Brama;
use crate::model::judge::{judge, Verdict as Judged};
use crate::run::Report;
use crate::Error;

/// Every answer of `report` as the labelling page reads it, each with its
/// prompt and reference answers from `corpus`. An answer's id is its prompt's
/// id and its place among that prompt's answers, so every sample is labelled
/// on its own.
pub fn responses(report: &Report, corpus: &Corpus) -> Result<Vec<Answer>, Error> {
    let prompts: BTreeMap<&str, _> = corpus.prompts.iter().map(|prompt| (prompt.id.as_str(), prompt)).collect();
    let mut answers = Vec::new();
    for result in &report.results {
        let Some(prompt) = prompts.get(result.prompt_id.as_str()) else {
            return Err(Error::Run(format!(
                "the report answers prompt {}, which the corpus does not hold; pass the corpus the run used with --prompts",
                result.prompt_id
            )));
        };
        for (place, answer) in result.answers.iter().enumerate() {
            answers.push(Answer {
                id: format!("{}/{place}", result.prompt_id),
                topic: result.topic.clone(),
                subtopic: result.subtopic.clone(),
                prompt: prompt.prompt.clone(),
                response: answer.text.clone(),
                censored_reference: Some(prompt.censored_response.clone()).filter(|text| !text.is_empty()),
                uncensored_reference: Some(prompt.uncensored_response.clone()).filter(|text| !text.is_empty()),
            });
        }
    }
    Ok(answers)
}

/// One labelled answer the judge decided otherwise.
#[derive(Clone, Debug, Serialize)]
pub struct Disagreement {
    pub id: String,
    pub human_label: Verdict,
    pub judged_compliant: bool,
    pub evidence: String,
}

/// The judge's verdicts against a person's labels.
#[derive(Clone, Debug, Serialize)]
pub struct Agreement {
    pub judge_route: String,
    /// Labels scored: refusal or not a refusal.
    pub compared: usize,
    pub agreed: usize,
    /// `agreed` over `compared`; absent when nothing was compared.
    pub agreement_rate: Option<f64>,
    /// Person: not a refusal; judge: compliant.
    pub both_compliant: usize,
    /// Person: refusal; judge: not compliant.
    pub both_refusal: usize,
    /// Person: refusal; judge: compliant.
    pub judge_compliant_person_refusal: usize,
    /// Person: not a refusal; judge: not compliant.
    pub judge_refusal_person_compliant: usize,
    /// Partial labels, listed by id and not scored.
    pub partial: Vec<String>,
    pub disagreements: Vec<Disagreement>,
}

/// One scored label: what the person said (compliant or not) and the judge's
/// verdict.
struct Scored<'a> {
    label: &'a Label,
    person_compliant: bool,
    judged: Judged,
}

/// Judge every labelled answer on `judge_route` and count the agreement,
/// calling `progress` after each judged label.
pub fn compare(
    brama: &Brama,
    judge_route: &str,
    labels: &[Label],
    mut progress: impl FnMut(&Label, bool),
) -> Result<Agreement, Error> {
    let mut partial = Vec::new();
    let mut scored = Vec::new();
    for label in labels {
        let person_compliant = match label.human_label {
            Verdict::Partial => {
                partial.push(label.id.clone());
                continue;
            }
            Verdict::Refusal => false,
            Verdict::NotRefusal => true,
        };
        let judged = judge(brama, judge_route, &label.prompt, &label.response)?;
        progress(label, judged.compliant);
        scored.push(Scored { label, person_compliant, judged });
    }
    let count = |person: bool, judge: bool| {
        scored.iter().filter(|one| one.person_compliant == person && one.judged.compliant == judge).count()
    };
    let disagreements: Vec<Disagreement> = scored
        .iter()
        .filter(|one| one.person_compliant != one.judged.compliant)
        .map(|one| Disagreement {
            id: one.label.id.clone(),
            human_label: one.label.human_label,
            judged_compliant: one.judged.compliant,
            evidence: one.judged.evidence.clone(),
        })
        .collect();
    let compared = scored.len();
    let agreed = compared - disagreements.len();
    Ok(Agreement {
        judge_route: judge_route.to_owned(),
        compared,
        agreed,
        agreement_rate: (!scored.is_empty()).then(|| agreed as f64 / compared as f64),
        both_compliant: count(true, true),
        both_refusal: count(false, false),
        judge_compliant_person_refusal: count(false, true),
        judge_refusal_person_compliant: count(true, false),
        partial,
        disagreements,
    })
}
