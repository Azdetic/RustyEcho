//! Downloads and validates the Whisper model without transcribing anything.
//! Used by the Dockerfile's builder stage to warm the Hugging Face cache at
//! image-build time, so the runtime container doesn't need network access
//! (or pay the download latency) for the default model on first request.
//!
//!   cargo run -p rustyecho-inference --example prefetch_model --release

fn main() -> anyhow::Result<()> {
    let model_id = std::env::var("WHISPER_MODEL_ID")
        .unwrap_or_else(|_| rustyecho_inference::DEFAULT_MODEL_ID.to_string());
    let revision = std::env::var("WHISPER_REVISION")
        .unwrap_or_else(|_| rustyecho_inference::DEFAULT_REVISION.to_string());

    println!("prefetching {model_id} @ {revision}...");
    let start = std::time::Instant::now();
    rustyecho_inference::WhisperTranscriber::load(&model_id, &revision)?;
    println!("done in {:?}", start.elapsed());
    Ok(())
}
