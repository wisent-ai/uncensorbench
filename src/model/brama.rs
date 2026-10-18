//! The one model client: Brama's OpenAI-compatible chat completions.
//!
//! The caller's environment carries the gateway and the bearer issued for it
//! under the names Kronika and the research tools read (BRAMA_URL,
//! BRAMA_API_KEY), plus the signed agent identity (WISENT_APP_AGENT_ID,
//! WISENT_APP_AGENT_AUTH_SECRET) when the route requires one. No provider key
//! or provider host appears here: the benchmark used to load the model under
//! test with transformers on the local machine, which is inference outside
//! Brama.

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::Error;

const URL: &str = "BRAMA_URL";
const KEY: &str = "BRAMA_API_KEY";
const AGENT: &str = "WISENT_APP_AGENT_ID";
const SECRET: &str = "WISENT_APP_AGENT_AUTH_SECRET";
const CHAT: &str = "/v1/chat/completions";

/// Generation settings the caller states. One left out is not sent, so the
/// route's own applies; the report records exactly these.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

pub struct Brama {
    url: String,
    api_key: String,
    agent: Option<(String, String)>,
}

/// The trimmed value of the environment variable `name`; `None` when it is
/// unset or blank, and a refusal when it is set to something that is not text.
fn setting(name: &str) -> Result<Option<String>, Error> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value.trim().to_owned()).filter(|value| !value.is_empty())),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(Error::Config(format!("{name} cannot be read: {error}"))),
    }
}

impl Brama {
    /// The gateway named by the environment, or a refusal naming what is missing.
    pub fn from_env() -> Result<Self, Error> {
        let url = setting(URL)?.ok_or_else(|| Error::Config(format!("{URL} is not set")))?;
        let api_key = setting(KEY)?.ok_or_else(|| Error::Config(format!("{KEY} is not set")))?;
        let agent = match (setting(AGENT)?, setting(SECRET)?) {
            (Some(id), Some(secret)) => Some((id, secret)),
            (None, None) => None,
            _ => {
                return Err(Error::Config(format!(
                    "{AGENT} and {SECRET} are set together or not at all"
                )))
            }
        };
        Ok(Self {
            url: url.trim_end_matches('/').to_owned(),
            api_key,
            agent,
        })
    }

    /// `text` as the only user turn to `route`; the text of the answer.
    pub fn chat(&self, route: &str, text: &str, sampling: &Sampling) -> Result<String, Error> {
        self.complete(route, text, sampling, None)
    }

    /// `text` as the only user turn to `route`, answered by one required
    /// call of the function `name` whose parameters are the JSON `schema`
    /// (strict), and read as that call's arguments: an answer without exactly
    /// that call, or whose arguments are not one JSON object, is refused with
    /// the answer, never searched for an object inside its prose.
    pub fn object(
        &self,
        route: &str,
        text: &str,
        sampling: &Sampling,
        name: &str,
        schema: Value,
    ) -> Result<Value, Error> {
        let function = json!({
            "name": name,
            "description": format!("The {name} answer, as its parameters describe it."),
            "strict": true,
            "parameters": schema,
        });
        let said = self.complete(route, text, sampling, Some(function))?;
        match serde_json::from_str::<Value>(said.trim()) {
            Ok(object @ Value::Object(_)) => Ok(object),
            Ok(_) => Err(Error::Brama(format!(
                "{route} answered no {name} object: {said}"
            ))),
            Err(error) => Err(Error::Brama(format!(
                "{route} answered no {name} object ({error}): {said}"
            ))),
        }
    }

    /// The answer's text, or, when `function` is given, the arguments of the
    /// one call of it the request requires.
    fn complete(
        &self,
        route: &str,
        text: &str,
        sampling: &Sampling,
        function: Option<Value>,
    ) -> Result<String, Error> {
        let mut body = serde_json::to_value(sampling).map_err(|error| {
            Error::Brama(format!(
                "the request to {route} could not be written: {error}"
            ))
        })?;
        body["model"] = json!(route);
        body["messages"] = json!([{ "role": "user", "content": text }]);
        let called = function.as_ref().map(|function| function["name"].clone());
        if let Some(function) = function {
            body["tool_choice"] =
                json!({ "type": "function", "function": { "name": function["name"] } });
            body["tools"] = json!([{ "type": "function", "function": function }]);
        }
        let body = body.to_string();
        let endpoint = format!("{}{CHAT}", self.url);
        let mut request = ureq::post(&endpoint)
            .set("content-type", "application/json")
            .set("authorization", &format!("Bearer {}", self.api_key));
        if let Some((id, secret)) = &self.agent {
            let timestamp = chrono::Utc::now().timestamp().to_string();
            let digest = hex::encode(Sha256::digest(body.as_bytes()));
            let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).map_err(|error| {
                Error::Config(format!("{SECRET} cannot key a signature: {error}"))
            })?;
            mac.update(format!("{id}:{timestamp}:{digest}").as_bytes());
            let signature = hex::encode(mac.finalize().into_bytes());
            request = request
                .set("x-agent-id", id)
                .set("x-agent-timestamp", &timestamp)
                .set("x-agent-signature", &signature);
        }
        let raw = match request.send_string(&body) {
            Ok(response) => response.into_string().map_err(|error| {
                Error::Brama(format!(
                    "{endpoint}: the answer of {route} could not be read: {error}"
                ))
            })?,
            Err(ureq::Error::Status(status, response)) => {
                let said = response.into_string().map_err(|error| {
                    Error::Brama(format!(
                        "{endpoint} answered HTTP {status} for {route}, unreadably: {error}"
                    ))
                })?;
                return Err(Error::Brama(format!(
                    "{endpoint} answered HTTP {status} for {route}: {said}"
                )));
            }
            Err(error) => {
                return Err(Error::Brama(format!(
                    "{endpoint} could not be reached for {route}: {error}"
                )))
            }
        };
        let answer: Value = serde_json::from_str(&raw).map_err(|error| {
            Error::Brama(format!(
                "{endpoint} answered {route} with no JSON ({error}): {raw}"
            ))
        })?;
        let first = answer["choices"]
            .as_array()
            .and_then(|choices| choices.first());
        if let Some(name) = called {
            let calls = first.and_then(|choice| choice["message"]["tool_calls"].as_array());
            return match calls.map(Vec::as_slice) {
                Some([call]) if call["function"]["name"] == name => call["function"]["arguments"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        Error::Brama(format!(
                            "{endpoint}: {route}'s call of {name} carries no arguments: {raw}"
                        ))
                    }),
                _ => Err(Error::Brama(format!(
                    "{endpoint}: {route} did not answer by exactly one call of {name}: {raw}"
                ))),
            };
        }
        answer["choices"]
            .as_array()
            .and_then(|choices| choices.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .filter(|content| !content.trim().is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                Error::Brama(format!(
                    "{endpoint} answered {route} without message content: {raw}"
                ))
            })
    }
}
