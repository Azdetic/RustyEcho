use std::sync::Arc;

use rustyecho_core::MockTranscriber;
use rustyecho_gateway::{app::build_router, state::AppState};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let state = AppState {
        // Phase 1 placeholder swapped for the real `candle` or `whisper-rs`
        // backed transcriber in Phase 2 without touching any gateway code
        transcriber: Arc::new(MockTranscriber),
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

    tracing::info!("rustyecho-gateway listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app)
        .await
        .expect("server error");
}
