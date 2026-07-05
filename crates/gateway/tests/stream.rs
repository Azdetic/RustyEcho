use std::{net::SocketAddr, sync::Arc, time::Duration};

use futures_util::{SinkExt, StreamExt};
use rustyecho_core::MockTranscriber;
use rustyecho_gateway::{app::build_router, state::AppState};
use tokio_tungstenite::tungstenite::Message as WsMessage;

async fn spawn_test_server() -> SocketAddr {
    let state = AppState {
        transcriber: Arc::new(MockTranscriber),
        max_upload_bytes: rustyecho_audio::MAX_FILE_BYTES,
    };
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

fn pcm16le_silence(duration_ms: u64) -> Vec<u8> {
    let n_samples = (16_000 * duration_ms / 1000) as usize;
    let mut bytes = Vec::with_capacity(n_samples * 2);
    for _ in 0..n_samples {
        bytes.extend_from_slice(&0i16.to_le_bytes());
    }
    bytes
}

async fn next_json(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> serde_json::Value {
    let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timed out waiting for server message")
        .expect("stream ended unexpectedly")
        .expect("websocket error");
    let text = msg.into_text().expect("expected text frame");
    serde_json::from_str(&text).expect("expected valid JSON")
}

#[tokio::test]
async fn stream_emits_partial_then_final() {
    let addr = spawn_test_server().await;
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/stream"))
        .await
        .unwrap();

    ws.send(WsMessage::Text(
        r#"{"type":"start","sample_rate":16000,"channels":1}"#.to_string(),
    ))
    .await
    .unwrap();

    // 1s of silence == exactly one fixed-interval chunk -> one partial.
    ws.send(WsMessage::Binary(pcm16le_silence(1000)))
        .await
        .unwrap();

    let partial = next_json(&mut ws).await;
    assert_eq!(partial["type"], "partial");

    ws.send(WsMessage::Text(r#"{"type":"stop"}"#.to_string()))
        .await
        .unwrap();

    let final_msg = next_json(&mut ws).await;
    assert_eq!(final_msg["type"], "final");
}

#[tokio::test]
async fn stream_rejects_unsupported_format() {
    let addr = spawn_test_server().await;
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/stream"))
        .await
        .unwrap();

    ws.send(WsMessage::Text(
        r#"{"type":"start","sample_rate":44100,"channels":2}"#.to_string(),
    ))
    .await
    .unwrap();

    let err = next_json(&mut ws).await;
    assert_eq!(err["type"], "error");
    assert_eq!(err["code"], "UNSUPPORTED_FORMAT");
}

#[tokio::test]
async fn stream_rejects_missing_handshake() {
    let addr = spawn_test_server().await;
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/stream"))
        .await
        .unwrap();

    // Sending binary audio before the start handshake is invalid.
    ws.send(WsMessage::Binary(pcm16le_silence(100)))
        .await
        .unwrap();

    let result = tokio::time::timeout(Duration::from_secs(5), ws.next()).await;
    match result {
        Ok(Some(Ok(msg))) => {
            let json: serde_json::Value = serde_json::from_str(&msg.into_text().unwrap()).unwrap();
            assert_eq!(json["type"], "error");
        }
        // Server closing the connection outright is also an acceptable
        // rejection of a malformed handshake.
        Ok(Some(Err(_))) | Ok(None) => {}
        Err(_) => panic!("server neither responded nor closed the connection"),
    }
}
