//! Standalone correctness check run manually not part of the automated
//! test suite because it downloads a real model on first run
//!
//!   cargo run -p rustyecho-inference --example transcribe_file --release -- jfk_sample.wav
//!
//! jfk_sample.wav is the same known transcript sample the upstream candle
//! Whisper example uses so the expected output is roughly
//! And so my fellow Americans ask not what your country can do for you
//! ask what you can do for your country

use rustyecho_core::Transcriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "jfk_sample.wav".to_string());

    let bytes = std::fs::read(&path)?;
    let pcm = rustyecho_audio::decode_wav(&bytes).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    println!(
        "loading {} (downloads from Hugging Face Hub on first run)...",
        rustyecho_inference::DEFAULT_MODEL_ID
    );
    let load_start = std::time::Instant::now();
    let transcriber = rustyecho_inference::WhisperTranscriber::load(
        rustyecho_inference::DEFAULT_MODEL_ID,
        rustyecho_inference::DEFAULT_REVISION,
    )?;
    println!("model loaded in {:?}", load_start.elapsed());

    println!("transcribing {path} ({} ms of audio)...", pcm.duration_ms());
    let decode_start = std::time::Instant::now();
    let result = transcriber
        .transcribe(pcm)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    println!("--- result (decoded in {:?}) ---", decode_start.elapsed());
    println!("{}", result.text);
    Ok(())
}
