//! Integration tests against the real Whisper model which are ignored by default
//! because they download ~75MB from Hugging Face Hub on first run and take
//! several seconds to load and run so execute explicitly with
//!
//!   cargo test -p rustyecho-inference -- --ignored

use std::sync::OnceLock;

use rustyecho_core::{AudioFormat, PcmBuffer, Transcriber};
use rustyecho_inference::{WhisperTranscriber, DEFAULT_MODEL_ID, DEFAULT_REVISION};

fn transcriber() -> &'static WhisperTranscriber {
    static TRANSCRIBER: OnceLock<WhisperTranscriber> = OnceLock::new();
    TRANSCRIBER.get_or_init(|| {
        WhisperTranscriber::load(DEFAULT_MODEL_ID, DEFAULT_REVISION)
            .expect("failed to load Whisper model")
    })
}

fn pooled_transcriber() -> &'static WhisperTranscriber {
    static TRANSCRIBER: OnceLock<WhisperTranscriber> = OnceLock::new();
    TRANSCRIBER.get_or_init(|| {
        WhisperTranscriber::load_pool(DEFAULT_MODEL_ID, DEFAULT_REVISION, 2)
            .expect("failed to load Whisper model pool")
    })
}

#[tokio::test]
#[ignore = "downloads a real model from Hugging Face Hub; run with --ignored"]
async fn transcribes_known_jfk_sample() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../jfk_sample.wav");
    let bytes = std::fs::read(path).expect("jfk_sample.wav should exist at repo root");
    let pcm = rustyecho_audio::decode_wav(&bytes).expect("valid wav");

    let result = transcriber()
        .transcribe(pcm, None)
        .await
        .expect("transcription should succeed");

    let text = result.text.to_lowercase();
    assert!(
        text.contains("ask not what your country"),
        "unexpected transcription: {text}"
    );
}

/// Regression test for the no speech gating fix because without it Whisper
/// confidently hallucinates plausible sounding text for silent or noisy audio
/// instead of reporting that nothing was said
#[tokio::test]
#[ignore = "downloads a real model from Hugging Face Hub; run with --ignored"]
async fn silence_does_not_hallucinate_text() {
    let pcm = PcmBuffer {
        samples: vec![0.0; AudioFormat::TARGET.sample_rate as usize * 3],
        format: AudioFormat::TARGET,
    };

    let result = transcriber()
        .transcribe(pcm, None)
        .await
        .expect("transcription should succeed");

    assert!(
        result.text.trim().is_empty(),
        "expected no-speech gating to suppress hallucinated text on silence, got: {:?}",
        result.text
    );
}

/// Regression test for the worker pool fix where with pool_size=2 two
/// transcriptions running at once should take meaningfully less than 2x a
/// single one
/// Before the pool existed a single mutex serialized every
/// request through one model instance so this would have taken around 2x no
/// matter how many workers were configured
#[tokio::test]
#[ignore = "downloads a real model from Hugging Face Hub; run with --ignored"]
async fn pool_allows_concurrent_transcriptions() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../jfk_sample.wav");
    let bytes = std::fs::read(path).expect("jfk_sample.wav should exist at repo root");
    let pcm = rustyecho_audio::decode_wav(&bytes).expect("valid wav");

    let transcriber = pooled_transcriber();

    // Warm up so the timing below is not dominated by one time setup costs
    let _ = transcriber.transcribe(pcm.clone(), None).await;

    let single_start = std::time::Instant::now();
    transcriber
        .transcribe(pcm.clone(), None)
        .await
        .expect("solo transcription should succeed");
    let single_elapsed = single_start.elapsed();

    let concurrent_start = std::time::Instant::now();
    let (a, b) = tokio::join!(
        transcriber.transcribe(pcm.clone(), None),
        transcriber.transcribe(pcm.clone(), None),
    );
    a.expect("first concurrent transcription should succeed");
    b.expect("second concurrent transcription should succeed");
    let concurrent_elapsed = concurrent_start.elapsed();

    assert!(
        concurrent_elapsed < single_elapsed * 2,
        "two concurrent transcriptions took {concurrent_elapsed:?}, a single one took \
         {single_elapsed:?} -- expected the pool to run them in parallel, not fully \
         serialize like a single mutex would"
    );
}

/// Sanity check for the cross chunk context feature where supplying benign
/// previous chunk context should not corrupt an otherwise well known
/// transcript
#[tokio::test]
#[ignore = "downloads a real model from Hugging Face Hub; run with --ignored"]
async fn context_conditioning_does_not_corrupt_transcription() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../jfk_sample.wav");
    let bytes = std::fs::read(path).expect("jfk_sample.wav should exist at repo root");
    let pcm = rustyecho_audio::decode_wav(&bytes).expect("valid wav");

    let result = transcriber()
        .transcribe(pcm, Some("Good morning, everyone."))
        .await
        .expect("transcription should succeed");

    let text = result.text.to_lowercase();
    assert!(
        text.contains("ask not what your country"),
        "context conditioning should not corrupt the known-good transcript: {text}"
    );
}

/// Regression test for the token budget fix where previous_text longer than the
/// model context window must be truncated keeping the most recent
/// tokens and not overflow max_target_positions and fail the decode
#[tokio::test]
#[ignore = "downloads a real model from Hugging Face Hub; run with --ignored"]
async fn long_previous_text_is_truncated_not_rejected() {
    let long_context = "the quick brown fox jumps over the lazy dog ".repeat(100);
    let pcm = PcmBuffer {
        samples: vec![0.0; AudioFormat::TARGET.sample_rate as usize * 2],
        format: AudioFormat::TARGET,
    };

    let result = transcriber().transcribe(pcm, Some(&long_context)).await;

    assert!(
        result.is_ok(),
        "expected long previous_text to be truncated to fit the model's context \
         window, not cause a decode error: {result:?}"
    );
}
