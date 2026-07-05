use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use rustyecho_core::GatewayError;
use serde_json::json;

/// Newtype so we can implement the axum foreign `IntoResponse` trait for the
/// foreign `GatewayError` type defined in `rustyecho-core`
pub struct ApiError(pub GatewayError);

impl From<GatewayError> for ApiError {
    fn from(err: GatewayError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = match &self.0 {
            GatewayError::InvalidAudioFormat(_) => {
                (StatusCode::UNSUPPORTED_MEDIA_TYPE, "INVALID_AUDIO_FORMAT")
            }
            GatewayError::FileTooLarge { .. } => {
                (StatusCode::PAYLOAD_TOO_LARGE, "FILE_TOO_LARGE")
            }
            GatewayError::DecodeFailed(_) => (StatusCode::BAD_REQUEST, "DECODE_FAILED"),
            GatewayError::Timeout => (StatusCode::REQUEST_TIMEOUT, "TIMEOUT"),
            GatewayError::TranscribeFailed(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "TRANSCRIBE_FAILED")
            }
        };

        let body = Json(json!({
            "error": { "code": code, "message": self.0.to_string() }
        }));

        (status, body).into_response()
    }
}
