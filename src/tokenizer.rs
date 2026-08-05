use anyhow::{Context, Result};
use base64::Engine;
use hf_hub::api::sync::ApiBuilder;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::fs;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tiktoken_rs::CoreBPE;
use tokenizers::{FromPretrainedParameters, Tokenizer};

const KIMI_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone)]
pub enum BenchmarkTokenizer {
    HuggingFace(Tokenizer),
    Kimi(Arc<CoreBPE>),
}

impl fmt::Debug for BenchmarkTokenizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HuggingFace(_) => formatter.write_str("BenchmarkTokenizer::HuggingFace"),
            Self::Kimi(_) => formatter.write_str("BenchmarkTokenizer::Kimi"),
        }
    }
}

impl BenchmarkTokenizer {
    pub fn load(name: &str, hf_token: Option<String>) -> Result<Self> {
        let params = FromPretrainedParameters {
            token: hf_token.clone(),
            ..Default::default()
        };
        match Tokenizer::from_pretrained(name, Some(params)) {
            Ok(tokenizer) => Ok(Self::HuggingFace(tokenizer)),
            Err(hub_error) => {
                let local_path = Path::new(name).join("tokenizer.json");
                if let Ok(tokenizer) = Tokenizer::from_file(&local_path) {
                    return Ok(Self::HuggingFace(tokenizer));
                }

                Self::load_kimi(name, hf_token).with_context(|| {
                    format!(
                        "Error loading tokenizer from Hub ({hub_error}) or local path ({})",
                        local_path.display()
                    )
                })
            }
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!(e.to_string())),
            Self::Kimi(tokenizer) => Ok(tokenizer.encode_ordinary(text)),
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer
                .decode(tokens, true)
                .map_err(|e| anyhow::anyhow!(e.to_string())),
            Self::Kimi(tokenizer) => Ok(String::from_utf8_lossy(&tokenizer.decode_bytes(tokens)?).into_owned()),
        }
    }

    fn load_kimi(name: &str, hf_token: Option<String>) -> Result<Self> {
        let (model_path, config_path) = if Path::new(name).is_dir() {
            (PathBuf::from(name).join("tiktoken.model"), PathBuf::from(name).join("tokenizer_config.json"))
        } else {
            let api = ApiBuilder::from_env().with_token(hf_token).build()?;
            let repo = api.model(name.to_owned());
            (repo.get("tiktoken.model")?, repo.get("tokenizer_config.json")?)
        };

        let config: Value = serde_json::from_slice(&fs::read(&config_path)?)?;
        if config.get("tokenizer_class").and_then(Value::as_str) != Some("TikTokenTokenizer") {
            anyhow::bail!("Tokenizer is not a Kimi TikTokenTokenizer");
        }

        let mut mergeable_ranks = FxHashMap::default();
        for line in std::str::from_utf8(&fs::read(&model_path)?)?.lines() {
            let (token, rank) = line
                .split_once(' ')
                .context("Invalid tiktoken.model entry")?;
            mergeable_ranks.insert(
                base64::engine::general_purpose::STANDARD.decode(token)?,
                rank.parse::<u32>()?,
            );
        }

        let special_tokens = config["added_tokens_decoder"]
            .as_object()
            .context("Missing Kimi special-token definitions")?
            .iter()
            .filter_map(|(token_id, definition)| {
                Some((
                    definition.get("content")?.as_str()?.to_owned(),
                    token_id.parse::<u32>().ok()?,
                ))
            })
            .collect();

        Ok(Self::Kimi(Arc::new(CoreBPE::new(
            mergeable_ranks,
            special_tokens,
            KIMI_PATTERN,
        )?)))
    }
}
