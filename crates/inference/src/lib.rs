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
//!
//! Language handling adapts to whichever model is configured: `.en` model
//! tokenizers have no per-language tokens, so those stay English-only with
//! no extra cost. Multilingual tokenizers (the default) get a chunk split
//! into language-homogeneous sub-segments (see `Inner::segment_by_language`)
//! before decoding each segment separately -- a single language-confidence
//! check on the whole chunk was tried first and rejected: measured
//! confidence on a real English+Indonesian spliced test clip was 0.886,
//! indistinguishable from confidence on genuinely single-language clips
//! (0.88-0.95), so there is no cheap "looks ambiguous" signal to gate on.
//!
//! Segmentation encodes the chunk exactly once, then slices the *encoder
//! output* per detection window instead of re-encoding raw audio per
//! window. That distinction matters a lot: `pcm_to_mel` always pads to a
//! full 30s-equivalent mel regardless of actual input length, so the
//! encoder's cost is fixed per call no matter how short the window is --
//! re-encoding N small windows separately (an earlier version of this code
//! did that) multiplies that fixed cost by N, which measured out to a
//! 5-10x latency regression even on single-language audio. Slicing the
//! already-computed encoder output instead keeps the per-chunk encoder
//! cost the same as before segmentation existed; only the (much cheaper)
//! single-decoder-step language probes multiply with window count.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rustyecho_core::{PcmBuffer, TranscribeError, TranscriptionResult, Transcriber};
use tokenizers::Tokenizer;

const MEL_FILTERS_80: &[u8] = include_bytes!("melfilters.bytes");

/// Default multilingual model (not the `.en` variant) so language
/// auto-detection actually has something to detect. `base` is a middle
/// ground: meaningfully better non-English accuracy than `tiny`, still
/// CPU-friendly. Override via `WHISPER_MODEL_ID`/`WHISPER_REVISION` --
/// `small`/`medium` improve accuracy further at higher CPU latency, or
/// swap back to `tiny.en`/`base.en` for English-only + lower latency.
/// The pinned revision matches the upstream candle example because main on this repo lacks a
/// model.safetensors file and only has a PyTorch checkpoint
pub const DEFAULT_MODEL_ID: &str = "openai/whisper-base";
pub const DEFAULT_REVISION: &str = "refs/pr/22";

/// Whisper's encoder downsamples mel frames by 2x (HOP_LENGTH=160 samples
/// per mel frame, 2 mel frames per encoder position) -- 16_000 / (160 * 2).
const ENCODER_POSITIONS_PER_SEC: usize = 50;
/// Window size for per-window language detection when segmenting a chunk
/// by language (see `Inner::segment_by_language`), in encoder positions
/// (not samples -- see `ENCODER_POSITIONS_PER_SEC`). Short enough to catch
/// a language switch without a pause, long enough for language detection
/// to be reasonably reliable on real speech.
const LANGUAGE_WINDOW_POSITIONS: usize = ENCODER_POSITIONS_PER_SEC * 3 / 2; // 1.5s
/// A trailing remainder shorter than this gets folded into the previous
/// window instead of being language detected on its own because it is too short to be
/// a reliable language signal by itself
const MIN_LANGUAGE_WINDOW_POSITIONS: usize = ENCODER_POSITIONS_PER_SEC / 2; // 0.5s
/// A window detected language only overrides the running language if its
/// confidence is at least this high
/// Measured on a real pure English clip genuine language windows scored 0.92 to 0.99
/// while classifier blips that picked the wrong language scored 0.30 and 0.45 which is comfortably below this
const LANGUAGE_SWITCH_CONFIDENCE_THRESHOLD: f32 = 0.6;

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

/// Whisper's 99 supported language codes. Only used to look up which of
/// them exist as `<|code|>` tokens in the loaded tokenizer -- `.en`
/// tokenizers have none of these, multilingual tokenizers have all of them.
const LANGUAGE_CODES: [&str; 99] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
    "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
    "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
];

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
    prev_token: u32,
    /// Token ids for every `<|code|>` language token this tokenizer has.
    /// Empty for `.en` (English-only) tokenizers, which is how `decode`
    /// tells whether to run language segmentation at all.
    language_token_ids: Vec<u32>,
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

    // `Api::new()` ignores `HF_HOME` (it uses `Cache::default()`, not
    // `Cache::from_env()`) -- `from_env()` is required for the cache
    // location to actually be overridable, which the Docker build relies on
    // to pre-warm a cache directory that gets copied into the runtime image.
    let api = ApiBuilder::from_env().build()?;
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
    // Used to prefix previous chunk context onto the decode prompt see
    // Inner::decode_segment so words split across a VAD chunk boundary are not lost
    let prev_token = token_id(&tokenizer, "<|startofprev|>")?;
    // Empty for `.en` tokenizers (no per-language tokens exist), which is
    // how `decode` knows whether to run language segmentation at all.
    let language_token_ids: Vec<u32> = LANGUAGE_CODES
        .iter()
        .filter_map(|code| token_id(&tokenizer, &format!("<|{code}|>")).ok())
        .collect();

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
        prev_token,
        language_token_ids,
        suppress_tokens,
    })
}

impl Inner {
    /// Decodes a chunk where samples must already be normalized
    /// -1.0..=1.0 mono PCM at 16kHz which is exactly what PcmBuffer guarantees.
    ///
    /// Encodes the whole chunk exactly once -- `pcm_to_mel` always pads to at
    /// least a 30s-equivalent mel regardless of actual input length, so the
    /// encoder pass costs the same either way; re-encoding smaller windows
    /// separately (an earlier version of this code did that) multiplied that
    /// fixed cost by however many windows there were, which measured out to
    /// a 5-10x latency regression even on single-language audio. Instead,
    /// language segmentation below slices the *already-computed* encoder
    /// output per window, which is cheap and scales with the actual slice
    /// size.
    ///
    /// For `.en` (English-only) models this is one encode + one
    /// `decode_segment` call, same as before language support existed. For
    /// multilingual models the chunk is first split into language-homogeneous
    /// segments (`segment_by_language`); `previous_text` context is only
    /// applied to the first segment, since later segments have no previous
    /// text of their own within the same chunk.
    fn decode(&mut self, samples: &[f32], previous_text: Option<&str>) -> anyhow::Result<String> {
        let audio_features = self.encode(samples)?;

        if self.language_token_ids.is_empty() {
            return self.decode_segment(&audio_features, None, previous_text);
        }

        let (_, total_positions, _) = audio_features.dims3()?;
        // Encoder positions run at ENCODER_POSITIONS_PER_SEC regardless of
        // how much of the (padded-to-30s) mel is real audio -- clamp to
        // that so windowing doesn't wander into the silent padding tail.
        let real_positions = ((samples.len() as f64 / 16_000.0) * ENCODER_POSITIONS_PER_SEC as f64)
            .ceil() as usize;
        let real_positions = real_positions.clamp(1, total_positions);

        let mut segments = self.segment_by_language(&audio_features, real_positions)?;
        // Whisper was trained on full 30s-equivalent windows and seemingly
        // relies on seeing some trailing silence to know speech has ended:
        // decoding a segment truncated exactly at the real-audio boundary
        // (dropping the silence-derived padding tail entirely) measured out
        // to severe repetition loops ("and so my fellow Americans" x35).
        // Only the *last* segment's end gets extended back out to the full
        // padded length; earlier segments still cut where the language
        // actually changes, since extending those would let them "see" the
        // next segment's different-language audio.
        if let Some(last) = segments.last_mut() {
            last.1 = total_positions;
        }
        let mut pieces: Vec<String> = Vec::new();
        for (i, (start, end, language_token)) in segments.into_iter().enumerate() {
            let segment_features = audio_features.narrow(1, start, end - start)?.contiguous()?;
            let ctx = if i == 0 { previous_text } else { None };
            let text = self.decode_segment(&segment_features, Some(language_token), ctx)?;
            let text = text.trim();
            if !text.is_empty() {
                pieces.push(text.to_string());
            }
        }
        Ok(pieces.join(" "))
    }

    /// Computes encoder audio features for a raw PCM sample slice.
    fn encode(&mut self, samples: &[f32]) -> anyhow::Result<Tensor> {
        let num_mel_bins = self.model.config.num_mel_bins;
        let mel = audio::pcm_to_mel(&self.model.config, samples, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, num_mel_bins, mel_len / num_mel_bins),
            &self.device,
        )?;
        Ok(self.model.encoder.forward(&mel, true)?)
    }

    /// Splits the already-computed `audio_features` (only the first
    /// `real_positions` of it -- the rest is silent padding) into
    /// `LANGUAGE_WINDOW_POSITIONS`-sized windows, language-detects each
    /// independently by slicing (not re-encoding), then merges adjacent
    /// windows that detected the same language into a single span. A real
    /// English+Indonesian spliced test clip decoded as one span (garbled,
    /// language-blended text) before this segmentation existed, and as two
    /// correctly separated English/Indonesian spans after.
    fn segment_by_language(
        &mut self,
        audio_features: &Tensor,
        real_positions: usize,
    ) -> anyhow::Result<Vec<(usize, usize, u32)>> {
        let mut boundaries: Vec<usize> =
            (0..real_positions).step_by(LANGUAGE_WINDOW_POSITIONS).collect();
        boundaries.push(real_positions);
        if boundaries.len() >= 3 {
            let last_len = boundaries[boundaries.len() - 1] - boundaries[boundaries.len() - 2];
            if last_len < MIN_LANGUAGE_WINDOW_POSITIONS {
                boundaries.remove(boundaries.len() - 2);
            }
        }

        // Raw per window detections before confidence smoothing
        // Measured on a real pure English 11s clip most windows landed at 0.92 to 0.99
        // confidence for English but two windows dipped to 0.45 and 0.30
        // and flipped to a different language at that low confidence which was
        // a classifier blip and not a real switch
        // Smoothing below only trusts a language change when it is confident about it
        // A low confidence window is treated as probably still whatever state it was in
        let mut windows: Vec<(usize, usize, u32, f32)> = Vec::with_capacity(boundaries.len());
        for pair in boundaries.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            let slice = audio_features.narrow(1, start, end - start)?.contiguous()?;
            let (token, confidence) = self.detect_language_token(&slice)?;
            tracing::debug!(start, end, token, confidence, "language window detected");
            windows.push((start, end, token, confidence));
        }

        let mut current_language: Option<u32> = None;
        let mut merged: Vec<(usize, usize, u32)> = Vec::with_capacity(windows.len());
        for (start, end, token, confidence) in windows {
            let effective_token = match current_language {
                Some(current) if token != current && confidence < LANGUAGE_SWITCH_CONFIDENCE_THRESHOLD => {
                    current
                }
                _ => token,
            };
            current_language = Some(effective_token);

            match merged.last_mut() {
                Some((_, last_end, last_token)) if *last_token == effective_token => {
                    *last_end = end;
                }
                _ => merged.push((start, end, effective_token)),
            }
        }
        Ok(merged)
    }

    /// Greedy decodes one language-homogeneous segment given its already
    /// computed encoder `audio_features` and (for multilingual models) the
    /// language token to condition on.
    ///
    /// `previous_text`, when given, is prepended as `<|startofprev|>`-conditioned
    /// context (same mechanism as OpenAI's `condition_on_previous_text`) so the
    /// model has a chance to continue a word or sentence that got cut off at
    /// the previous chunk boundary, rather than decoding each chunk in total
    /// isolation.
    ///
    /// Returns an empty string when the audio looks like silence or noise rather
    /// than speech where no_speech_prob is over NO_SPEECH_THRESHOLD just like the
    /// reference decoder because without this check Whisper never says nothing was
    /// said but instead confidently hallucinates plausible text for silent or noisy
    /// chunks
    fn decode_segment(
        &mut self,
        audio_features: &Tensor,
        language_token: Option<u32>,
        previous_text: Option<&str>,
    ) -> anyhow::Result<String> {
        let sample_len = self.model.config.max_target_positions / 2;

        let mut tokens: Vec<u32> = Vec::new();
        if let Some(prev) = previous_text.map(str::trim).filter(|s| !s.is_empty()) {
            let prev_ids = self
                .tokenizer
                .encode(prev, false)
                .map_err(anyhow::Error::msg)?;
            let prev_ids = prev_ids.get_ids();
            // Total sequence length prompt plus sot sequence plus generated must
            // stay within max_target_positions or the position embedding
            // lookup goes out of bounds
            // We reserve prev_token plus the 3 sot sequence tokens plus this call up to sample_len generation
            // budget with a small safety margin before capping the prompt
            let max_prev_tokens = sample_len.saturating_sub(8);
            let start = prev_ids.len().saturating_sub(max_prev_tokens);
            tokens.push(self.prev_token);
            tokens.extend_from_slice(&prev_ids[start..]);
        }
        // Position of sot_token within tokens shifts when a prompt is
        // prepended so no speech detection below reads logits from here and not
        // always index 0 matching how the OpenAI reference decoder does it
        let sot_index = tokens.len();
        tokens.push(self.sot_token);
        if let Some(language_token) = language_token {
            tokens.push(language_token);
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&tokens_t, audio_features, i == 0)?;

            if i == 0 {
                // Logits right after startoftranscript before the
                // transcribe and no_timestamps tokens are seen which is the position
                // the OpenAI reference decoder reads no speech probability
                // from
                // Causal attention still lets this position see any
                // prepended previous text context matching upstream
                let first_step_logits = self
                    .model
                    .decoder
                    .final_linear(&ys.i((..1, sot_index..sot_index + 1))?)?
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

        // Only decode the tokens generated for this chunk and not the prepended
        // previous text prompt because that part is not new output
        self.tokenizer
            .decode(&tokens[sot_index..], true)
            .map_err(anyhow::Error::msg)
    }

    /// Picks the most likely language token for this audio: a single decoder
    /// step conditioned only on `<|startoftranscript|>`, softmax restricted
    /// to just the language tokens, argmax -- same mechanism as the OpenAI
    /// reference decoder. Only called when `language_token_ids` is
    /// non-empty (i.e. the tokenizer actually has per-language tokens).
    /// Returns the detected language token and its confidence (top softmax
    /// probability among the language tokens) for logging/diagnostics --
    /// confidence turned out NOT to reliably distinguish code-switched audio
    /// from single-language audio, so it's not used for any decision anymore.
    fn detect_language_token(&mut self, audio_features: &Tensor) -> anyhow::Result<(u32, f32)> {
        let tokens = Tensor::new(&[[self.sot_token]], &self.device)?;
        let ys = self.model.decoder.forward(&tokens, audio_features, true)?;
        let logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;

        let language_ids_t = Tensor::new(self.language_token_ids.as_slice(), &self.device)?;
        let logits = logits.index_select(&language_ids_t, 0)?;
        let probs: Vec<f32> = softmax(&logits, 0)?.to_vec1()?;

        let (best, confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, &p)| (idx, p))
            .expect("language_token_ids is non-empty when this is called");
        Ok((self.language_token_ids[best], confidence))
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> anyhow::Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("no token id for {token}"))
}

#[async_trait::async_trait]
impl Transcriber for WhisperTranscriber {
    async fn transcribe(
        &self,
        chunk: PcmBuffer,
        previous_text: Option<&str>,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        let inner = self.workers[idx].clone();
        let previous_text = previous_text.map(str::to_owned);

        let text = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut inner = inner.lock().expect("inference mutex poisoned");
            inner.decode(&chunk.samples, previous_text.as_deref())
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
