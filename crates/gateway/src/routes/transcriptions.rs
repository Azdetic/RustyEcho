use axum::{
    extract::{Multipart, State},
    Json,
};
use rustyecho_core::GatewayError;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{error::ApiError, state::AppState};

pub async fn create_transcription(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, ApiError> {
    let request_id = Uuid::new_v4().to_string();

    let mut file_bytes: Option<Vec<u8>> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| GatewayError::DecodeFailed(e.to_string()))?
    {
        if field.name() == Some("file") {
            let bytes = field
                .bytes()
                .await
                .map_err(|e| GatewayError::DecodeFailed(e.to_string()))?;
            file_bytes = Some(bytes.to_vec());
        }
    }

    let bytes = file_bytes.ok_or_else(|| {
        GatewayError::InvalidAudioFormat("missing required multipart field 'file'".into())
    })?;

    let pcm = rustyecho_audio::decode_wav(&bytes)?;
    let duration_ms = pcm.duration_ms();

    let result = state
        .transcriber
        .transcribe(pcm)
        .await
        .map_err(|e| GatewayError::TranscribeFailed(e.to_string()))?;

    tracing::info!(request_id, duration_ms, "transcription completed");

    Ok(Json(json!({
        "text": result.text,
        "duration_ms": duration_ms,
        "request_id": request_id,
    })))
}
