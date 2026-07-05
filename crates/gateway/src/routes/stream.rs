use std::time::Duration;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures_util::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use rustyecho_core::{AudioFormat, PcmBuffer};
use serde::{Deserialize, Serialize};
use tokio::time::timeout;

use crate::state::AppState;

const IDLE_TIMEOUT: Duration = Duration::from_secs(60);
/// Fixed-interval placeholder chunking. Real VAD-based chunking is
/// deferred to Milestone 4 once actual inference latency is known
/// (see planning.md — the "right" chunk size depends on it).
const CHUNK_DURATION_MS: u64 = 1000;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ClientMessage {
    Start { sample_rate: u32, channels: u16 },
    Stop,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ServerMessage {
    Partial { text: String },
    Final { text: String },
    Error { code: String, message: String },
}

pub async fn stream_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    if !handshake(&mut sender, &mut receiver).await {
        return;
    }

    let chunk_len_samples =
        (AudioFormat::TARGET.sample_rate as u64 * CHUNK_DURATION_MS / 1000) as usize;
    let mut pcm_buf: Vec<f32> = Vec::new();

    loop {
        let msg = match timeout(IDLE_TIMEOUT, receiver.next()).await {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(_))) | Ok(None) => break,
            Err(_) => {
                let _ = send_error(&mut sender, "TIMEOUT", "no data received within 60s").await;
                break;
            }
        };

        match msg {
            Message::Binary(bytes) => {
                append_pcm16le(&mut pcm_buf, &bytes);
                while pcm_buf.len() >= chunk_len_samples {
                    let chunk: Vec<f32> = pcm_buf.drain(..chunk_len_samples).collect();
                    if emit_result(&state, &mut sender, chunk, false).await.is_err() {
                        return;
                    }
                }
            }
            Message::Text(text) => {
                if matches!(
                    serde_json::from_str::<ClientMessage>(&text),
                    Ok(ClientMessage::Stop)
                ) {
                    let remaining = std::mem::take(&mut pcm_buf);
                    let _ = emit_result(&state, &mut sender, remaining, true).await;
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// Reads the first client message and validates it is a `start` handshake
/// declaring exactly the format every `Transcriber` expects. Phase 1
/// intentionally does not resample inside the streaming path (unlike the
/// batch `/v1/transcriptions` endpoint) — the client must send 16kHz mono
/// PCM16LE directly.
async fn handshake(
    sender: &mut SplitSink<WebSocket, Message>,
    receiver: &mut SplitStream<WebSocket>,
) -> bool {
    let text = match timeout(IDLE_TIMEOUT, receiver.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => text,
        _ => return false,
    };

    match serde_json::from_str::<ClientMessage>(&text) {
        Ok(ClientMessage::Start { sample_rate, channels })
            if sample_rate == AudioFormat::TARGET.sample_rate
                && channels == AudioFormat::TARGET.channels =>
        {
            true
        }
        Ok(ClientMessage::Start { .. }) => {
            let _ = send_error(
                sender,
                "UNSUPPORTED_FORMAT",
                "streaming only supports 16000Hz mono PCM16LE",
            )
            .await;
            false
        }
        _ => {
            let _ = send_error(
                sender,
                "INVALID_HANDSHAKE",
                "first message must be {\"type\":\"start\",\"sample_rate\":16000,\"channels\":1}",
            )
            .await;
            false
        }
    }
}

fn append_pcm16le(buf: &mut Vec<f32>, bytes: &[u8]) {
    for chunk in bytes.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        buf.push(sample as f32 / i16::MAX as f32);
    }
}

async fn emit_result(
    state: &AppState,
    sender: &mut SplitSink<WebSocket, Message>,
    samples: Vec<f32>,
    is_final: bool,
) -> Result<(), axum::Error> {
    let pcm = PcmBuffer {
        samples,
        format: AudioFormat::TARGET,
    };

    let result = match state.transcriber.transcribe(pcm).await {
        Ok(result) => result,
        Err(e) => {
            return send_error(sender, "TRANSCRIBE_FAILED", &e.to_string()).await;
        }
    };

    let msg = if is_final {
        ServerMessage::Final { text: result.text }
    } else {
        ServerMessage::Partial { text: result.text }
    };
    sender.send(to_ws_message(&msg)).await
}

async fn send_error(
    sender: &mut SplitSink<WebSocket, Message>,
    code: &str,
    message: &str,
) -> Result<(), axum::Error> {
    let msg = ServerMessage::Error {
        code: code.to_string(),
        message: message.to_string(),
    };
    sender.send(to_ws_message(&msg)).await
}

fn to_ws_message(msg: &ServerMessage) -> Message {
    Message::Text(serde_json::to_string(msg).expect("ServerMessage always serializes"))
}
