use std::{f32::consts::PI, io::Cursor, sync::Arc};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use rustyecho_core::MockTranscriber;
use rustyecho_gateway::{app::build_router, state::AppState};
use tower::ServiceExt;

fn make_test_wav() -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for i in 0..16_000u32 {
            let t = i as f32 / 16_000.0;
            let value = ((t * 440.0 * 2.0 * PI).sin() * i16::MAX as f32) as i16;
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}

fn multipart_body(field_name: &str, filename: &str, content_type: &str, data: &[u8]) -> (String, Vec<u8>) {
    let boundary = "RUSTYECHO_TEST_BOUNDARY";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(data);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn test_state() -> AppState {
    AppState {
        transcriber: Arc::new(MockTranscriber),
        max_upload_bytes: rustyecho_audio::MAX_FILE_BYTES,
    }
}

#[tokio::test]
async fn upload_valid_wav_returns_mock_transcription() {
    let app = build_router(test_state());
    let (content_type, body) = multipart_body("file", "test.wav", "audio/wav", &make_test_wav());

    let request = Request::builder()
        .method("POST")
        .uri("/v1/transcriptions")
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["duration_ms"], 1000);
    assert!(json["text"].as_str().unwrap().contains("mock transcription"));
    assert!(json["request_id"].is_string());
}

#[tokio::test]
async fn missing_file_field_returns_structured_error() {
    let app = build_router(test_state());
    let (content_type, body) = multipart_body("wrong_field", "test.wav", "audio/wav", b"not audio");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/transcriptions")
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "INVALID_AUDIO_FORMAT");
}

#[tokio::test]
async fn corrupt_audio_returns_bad_request() {
    let app = build_router(test_state());
    let (content_type, body) = multipart_body("file", "test.wav", "audio/wav", b"not a real wav file");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/transcriptions")
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn healthz_returns_ok() {
    let app = build_router(test_state());
    let request = Request::builder()
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
