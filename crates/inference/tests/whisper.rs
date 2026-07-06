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

#[tokio::test]
#[ignore = "downloads a real model from Hugging Face Hub; run with --ignored"]
async fn transcribes_known_jfk_sample() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../jfk_sample.wav");
    let bytes = std::fs::read(path).expect("jfk_sample.wav should exist at repo root");
    let pcm = rustyecho_audio::decode_wav(&bytes).expect("valid wav");

    let result = transcriber()
        .transcribe(pcm)
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
        .transcribe(pcm)
        .await
        .expect("transcription should succeed");

    assert!(
        result.text.trim().is_empty(),
        "expected no-speech gating to suppress hallucinated text on silence, got: {:?}",
        result.text
    );
}
