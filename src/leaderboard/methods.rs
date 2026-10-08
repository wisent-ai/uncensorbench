//! How each method moves a model away from its own baseline, read from paired
//! rows only: a row whose method is not the baseline counts when the same
//! model family and size also has a baseline row, and its delta is the
//! difference between the two rates. A method with no such pair is left out
//! rather than compared against some other model's baseline.

use serde::Serialize;

use super::Entry;

/// The method name a submission states for an unmodified model.
pub const BASELINE: &str = "none";

/// One method's standing over its paired rows.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Standing {
    pub method: String,
    /// Rows submitted with this method.
    pub models: usize,
    /// Rows that have a baseline of the same family and size; for the
    /// baseline itself, every baseline row.
    pub pairs: usize,
    pub mean_rate: f64,
    /// Mean of each pair's rate minus its baseline's; absent for the baseline.
    pub mean_delta: Option<f64>,
    pub max_rate: f64,
    pub min_rate: f64,
    pub mean_compliance_score: f64,
    /// The baseline row with the highest rate, or the paired row with the
    /// largest delta.
    pub best_model: String,
}

struct Pair<'a> {
    row: &'a Entry,
    delta: Option<f64>,
}

/// The mean of a non-empty list.
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn standing(method: &str, models: usize, pairs: &[Pair]) -> Option<Standing> {
    let best = pairs.iter().max_by(|left, right| match (left.delta, right.delta) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        _ => left.row.uncensored_rate.total_cmp(&right.row.uncensored_rate),
    })?;
    let rates: Vec<f64> = pairs.iter().map(|pair| pair.row.uncensored_rate).collect();
    let scores: Vec<f64> = pairs.iter().map(|pair| pair.row.avg_compliance_score).collect();
    let deltas: Option<Vec<f64>> = pairs.iter().map(|pair| pair.delta).collect();
    Some(Standing {
        method: method.to_owned(),
        models,
        pairs: pairs.len(),
        mean_rate: mean(&rates),
        mean_delta: deltas.map(|deltas| mean(&deltas)),
        max_rate: rates.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        min_rate: rates.iter().copied().fold(f64::INFINITY, f64::min),
        mean_compliance_score: mean(&scores),
        best_model: best.row.model.clone(),
    })
}

/// Every method in the order its best row ranks, each with its paired
/// standing; methods without a pair are left out.
pub fn compare(rows: &[Entry]) -> Vec<Standing> {
    let baselines: Vec<&Entry> = rows.iter().filter(|row| row.method == BASELINE).collect();
    let mut methods: Vec<&str> = Vec::new();
    for row in rows {
        if !methods.contains(&row.method.as_str()) {
            methods.push(&row.method);
        }
    }
    methods
        .into_iter()
        .filter_map(|method| {
            let submitted: Vec<&Entry> = rows.iter().filter(|row| row.method == method).collect();
            let pairs: Vec<Pair> = if method == BASELINE {
                submitted.iter().map(|row| Pair { row, delta: None }).collect()
            } else {
                submitted
                    .iter()
                    .filter_map(|row| {
                        let baseline = baselines.iter().find(|baseline| {
                            baseline.model_family == row.model_family && baseline.model_size == row.model_size
                        })?;
                        Some(Pair { row, delta: Some(row.uncensored_rate - baseline.uncensored_rate) })
                    })
                    .collect()
            };
            standing(method, submitted.len(), &pairs)
        })
        .collect()
}
