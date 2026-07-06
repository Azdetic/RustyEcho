use std::{sync::Arc, time::Duration};

use rustyecho_core::BoundedTranscriber;
use rustyecho_gateway::{app::build_router, state::AppState};
use rustyecho_inference::WhisperTranscriber;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let model_id = std::env::var("WHISPER_MODEL_ID")
        .unwrap_or_else(|_| rustyecho_inference::DEFAULT_MODEL_ID.to_string());
    let revision = std::env::var("WHISPER_REVISION")
        .unwrap_or_else(|_| rustyecho_inference::DEFAULT_REVISION.to_string());
    // How many model instances to load which means how many transcriptions can run
    // truly in parallel instead of serializing through one mutex
    // The default of 2 is conservative where each extra instance mmaps the same weights file
    // shared via the OS page cache but still duplicates its own KV cache
    // and activation buffers so this should scale with available RAM and CPU
    // and not be bumped blindly
    let pool_size: usize = std::env::var("WHISPER_POOL_SIZE")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(2);

    tracing::info!(model_id, revision, pool_size, "loading Whisper model");
    let transcriber = tokio::task::spawn_blocking({
        let model_id = model_id.clone();
        let revision = revision.clone();
        move || WhisperTranscriber::load_pool(&model_id, &revision, pool_size)
    })
    .await
    .expect("model loading task panicked")
    .expect("failed to load Whisper model");
    tracing::info!("model loaded");

    // Cap in flight requests beyond raw pool capacity so a burst fails fast
    // with 503/OVERLOADED instead of queueing silently and unboundedly on a
    // busy worker
    // Slack of 4x pool_size allows short bursts to smooth out
    // and the wait timeout bounds how long a request sits queued before giving up
    let max_in_flight = pool_size * 4;
    let queue_timeout_secs: u64 = std::env::var("WHISPER_QUEUE_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let transcriber = BoundedTranscriber::new(
        transcriber,
        max_in_flight,
        Duration::from_secs(queue_timeout_secs),
    );

    let state = AppState {
        // Phase 2 real inference
        // Swapping backends later like whisper-rs only means changing this line
        // never gateway code because everything downstream only depends on Transcriber
        transcriber: Arc::new(transcriber),
        max_upload_bytes: rustyecho_audio::MAX_FILE_BYTES,
    };

    let app = build_router(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("failed to bind port");

    tracing::info!(
        "rustyecho-gateway listening on {}",
        listener.local_addr().unwrap()
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

/// Waits for Ctrl+C or on Unix SIGTERM whichever comes first so
/// in flight WebSocket streams get a chance to finish instead of being
/// killed mid utterance when the process is stopped
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutdown signal received, draining connections");
}
