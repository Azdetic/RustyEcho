use axum::{
    routing::{get, post},
    Router,
};
use tower_http::{limit::RequestBodyLimitLayer, trace::TraceLayer};

use crate::{routes, state::AppState};

/// Builds the router without binding a socket so integration tests can
/// drive it directly via `tower::ServiceExt::oneshot`
pub fn build_router(state: AppState) -> Router {
    let max_upload_bytes = state.max_upload_bytes;

    Router::new()
        .route("/healthz", get(routes::health::health))
        .route(
            "/v1/transcriptions",
            post(routes::transcriptions::create_transcription),
        )
        .route("/v1/stream", get(routes::stream::stream_handler))
        .layer(RequestBodyLimitLayer::new(max_upload_bytes))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
