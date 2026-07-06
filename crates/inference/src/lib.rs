//! Real speech to text via candle and Whisper implementing the same
//! Transcriber trait that MockTranscriber implements in rustyecho-core
//! This is the Phase 2 seam where the gateway never depends on this crate
//! internals directly only on the trait so swapping models or backends later
//! never touches any gateway code
//!
//! Decoding here is deliberately simplified relative to the reference
//! candle Whisper example using greedy only no temperature fallback no
//! timestamps and no multi segment seeking
//! That is safe because every chunk handed to us by the gateway VAD is at most 5s
//! and audio pcm_to_mel always pads up to a full 30s window so a chunk never spans more than
//! one Whisper decode pass
//! Multi temperature fallback for when greedy decoding loops or degenerates
//! is the main quality feature deferred here

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rustyecho_core::{PcmBuffer, TranscribeError, TranscriptionResult, Transcriber};
use tokenizers::Tokenizer;

const MEL_FILTERS_80: &[u8] = include_bytes!("melfilters.bytes");

/// Default English only model where .en models skip language token handling
/// entirely and multilingual support is future work
/// The pinned revision matches the upstream candle example because main on this repo lacks a
/// model.safetensors file and only has a PyTorch checkpoint
pub const DEFAULT_MODEL_ID: &str = "openai/whisper-tiny.en";
pub const DEFAULT_REVISION: &str = "refs/pr/15";

/// Holds pool_size independent model instances so up to that many
/// transcriptions can run truly in parallel instead of all serializing
/// through a single mutex
/// Dispatch is round robin where request N picks worker N % pool_size
/// so a burst larger than the pool still queues per worker rather than fanning out further
/// This is a concurrency ceiling and not a request queue with backpressure or timeouts
pub struct WhisperTranscriber {
    workers: Vec<Arc<Mutex<Inner>>>,
    next: AtomicUsize,
}

struct Inner {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    device: Device,
    mel_filters: Vec<f32>,
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
    no_speech_token: u32,
    suppress_tokens: Tensor,
}

impl WhisperTranscriber {
    /// Downloads or reuses the local Hugging Face cache for model_id at
    /// revision and loads a single instance into memory which is equivalent to
    /// load_pool with a size of 1 meaning no concurrency headroom
    pub fn load(model_id: &str, revision: &str) -> anyhow::Result<Self> {
        Self::load_pool(model_id, revision, 1)
    }

    /// Same as load but loads pool_size independent model instances so
    /// that many requests can be decoded in parallel
    /// Each instance mmaps the same weights file so the OS page cache shares the read only
    /// weight pages across instances meaning the actual added memory per extra
    /// worker is just per instance activation buffers and not another full copy of the weights
    /// This is blocking and one time so call it during startup and never per request
    pub fn load_pool(model_id: &str, revision: &str, pool_size: usize) -> anyhow::Result<Self> {
        let pool_size = pool_size.max(1);
        let workers = (0..pool_size)
            .map(|_| Ok(Arc::new(Mutex::new(load_one(model_id, revision)?))))
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            workers,
            next: AtomicUsize::new(0),
        })
    }
}

/// Downloads or reuses the cached model_id at revision and loads one
/// instance into memory
fn load_one(model_id: &str, revision: &str) -> anyhow::Result<Inner> {
    let device = Device::Cpu;

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    if config.num_mel_bins != 80 {
        anyhow::bail!(
            "model expects {} mel bins, but only the bundled 80-bin filter bank is supported",
            config.num_mel_bins
        );
    }
    let mut mel_filters = vec![0f32; MEL_FILTERS_80.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
        MEL_FILTERS_80,
        &mut mel_filters,
    );

    // Safety the safetensors file comes from a pinned HF Hub revision
    // we just downloaded ourselves not arbitrary user input
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, &device)?
    };
    let model = m::model::Whisper::load(&vb, config.clone())?;

    let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
    let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
    let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
    let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
    let no_speech_token = m::NO_SPEECH_TOKENS
        .iter()
        .find_map(|token| token_id(&tokenizer, token).ok())
        .ok_or_else(|| anyhow::anyhow!("no no-speech token found in tokenizer vocab"))?;

    let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
        .map(|i| {
            if config.suppress_tokens.contains(&i) {
                f32::NEG_INFINITY
            } else {
                0f32
            }
        })
        .collect();
    let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;

    Ok(Inner {
        model,
        tokenizer,
        device,
        mel_filters,
        sot_token,
        eot_token,
        transcribe_token,
        no_timestamps_token,
        no_speech_token,
        suppress_tokens,
    })
}

impl Inner {
    /// Greedy decodes a single chunk where samples must already be normalized
    /// -1.0..=1.0 mono PCM at 16kHz which is exactly what PcmBuffer guarantees
    ///
    /// Returns an empty string when the audio looks like silence or noise rather
    /// than speech where no_speech_prob is over NO_SPEECH_THRESHOLD just like the
    /// reference decoder because without this check Whisper never says nothing was
    /// said but instead confidently hallucinates plausible text for silent or noisy
    /// chunks
    fn decode(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        let num_mel_bins = self.model.config.num_mel_bins;
        let mel = audio::pcm_to_mel(&self.model.config, samples, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, num_mel_bins, mel_len / num_mel_bins),
            &self.device,
        )?;

        let audio_features = self.model.encoder.forward(&mel, true)?;

        let sample_len = self.model.config.max_target_positions / 2;
        let mut tokens = vec![self.sot_token, self.transcribe_token, self.no_timestamps_token];

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            if i == 0 {
                // Logits right after startoftranscript before the
                // transcribe and no_timestamps tokens are seen which is the
                // position OpenAI reference decoder reads no speech
                // probability from
                let first_step_logits = self
                    .model
                    .decoder
                    .final_linear(&ys.i((..1, 0..1))?)?
                    .i(0)?
                    .i(0)?;
                let no_speech_prob = softmax(&first_step_logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()?;
                if no_speech_prob as f64 > m::NO_SPEECH_THRESHOLD {
                    return Ok(String::new());
                }
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            let logits = logits.broadcast_add(&self.suppress_tokens)?;

            let logits_v: Vec<f32> = logits.to_vec1()?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(idx, _)| idx as u32)
                .expect("vocab is non-empty");

            tokens.push(next_token);
            if next_token == self.eot_token {
                break;
            }
        }

        self.tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> anyhow::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("no token id for {token}"))
}

#[async_trait::async_trait]
impl Transcriber for WhisperTranscriber {
    async fn transcribe(&self, chunk: PcmBuffer) -> Result<TranscriptionResult, TranscribeError> {
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        let inner = self.workers[idx].clone();

        let text = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut inner = inner.lock().expect("inference mutex poisoned");
            inner.decode(&chunk.samples)
        })
        .await
        .map_err(|e| TranscribeError::Backend(e.to_string()))?
        .map_err(|e| TranscribeError::Backend(e.to_string()))?;

        Ok(TranscriptionResult {
            text,
            is_final: true,
            confidence: None,
        })
    }
}
