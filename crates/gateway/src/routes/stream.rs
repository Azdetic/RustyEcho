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

const FRAME_MS: u64 = 20;
const FRAME_SAMPLES: usize = (AudioFormat::TARGET.sample_rate as u64 * FRAME_MS / 1000) as usize;
/// RMS threshold on samples normalized to -1.0..=1.0 below which a frame
/// counts as silence
/// Not calibrated against real speech and noise just a
/// placeholder to validate the chunking pipeline with the same caveat as the
/// fixed interval version it replaces
const SILENCE_RMS_THRESHOLD: f32 = 0.02;
/// Trailing silence frames required after detected speech before a chunk
/// is cut and handed to the `Transcriber` 20 * 20ms = 400ms
const SILENCE_HANG_FRAMES: usize = 20;
/// Hard cap on buffered audio so continuous speech with no pauses still
/// gets cut somewhere to bound worst case latency and memory per connection
const MAX_BUFFER_MS: u64 = 5000;
const MAX_BUFFER_SAMPLES: usize = (AudioFormat::TARGET.sample_rate as u64 * MAX_BUFFER_MS / 1000) as usize;

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

    let mut vad = EnergyVad::new();
    // Text from the most recently transcribed non empty chunk in this
    // session fed back in as Whisper prompt conditioning context on the
    // next chunk so a word split across a VAD cut is not lost
    // Reset never happens mid session by design because only the latest chunk text is kept
    // rather than the full session history to bound memory and the prompt size
    let mut previous_text: Option<String> = None;

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
                if let Some(chunk) = vad.push(&bytes) {
                    match emit_result(
                        &state,
                        &mut sender,
                        chunk,
                        false,
                        previous_text.as_deref(),
                    )
                    .await
                    {
                        Ok(Some(text)) => previous_text = Some(text),
                        Ok(None) => {}
                        Err(_) => return,
                    }
                }
            }
            Message::Text(text) => {
                if matches!(
                    serde_json::from_str::<ClientMessage>(&text),
                    Ok(ClientMessage::Stop)
                ) {
                    let remaining = vad.take();
                    let _ = emit_result(
                        &state,
                        &mut sender,
                        remaining,
                        true,
                        previous_text.as_deref(),
                    )
                    .await;
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// Reads the first client message and validates it is a start handshake
/// declaring exactly the format every Transcriber expects
/// Phase 1 intentionally does not resample inside the streaming path unlike the
/// batch /v1/transcriptions endpoint so the client must send 16kHz mono
/// PCM16LE directly
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

/// Minimal energy based voice activity detector cuts a chunk once speech
/// has been seen followed by SILENCE_HANG_FRAMES of near silence or once
/// MAX_BUFFER_SAMPLES is reached regardless of silence
/// This is a Milestone 4 placeholder for the fixed interval chunker see
/// planning.md because the correct strategy depends on real inference latency
/// which only exists once Phase 2 swaps in a real Transcriber
struct EnergyVad {
    buf: Vec<f32>,
    analyzed_samples: usize,
    has_speech: bool,
    trailing_silence_frames: usize,
}

impl EnergyVad {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            analyzed_samples: 0,
            has_speech: false,
            trailing_silence_frames: 0,
        }
    }

    /// Appends raw PCM16LE bytes and returns a chunk ready for transcription
    /// if a cut point was reached
    fn push(&mut self, bytes: &[u8]) -> Option<Vec<f32>> {
        append_pcm16le(&mut self.buf, bytes);

        while self.analyzed_samples + FRAME_SAMPLES <= self.buf.len() {
            let frame = &self.buf[self.analyzed_samples..self.analyzed_samples + FRAME_SAMPLES];
            if rms(frame) >= SILENCE_RMS_THRESHOLD {
                self.has_speech = true;
                self.trailing_silence_frames = 0;
            } else {
                self.trailing_silence_frames += 1;
            }
            self.analyzed_samples += FRAME_SAMPLES;
        }

        let silence_hang_reached =
            self.has_speech && self.trailing_silence_frames >= SILENCE_HANG_FRAMES;
        let hard_cap_reached = self.buf.len() >= MAX_BUFFER_SAMPLES;

        if silence_hang_reached || hard_cap_reached {
            Some(self.take())
        } else {
            None
        }
    }

    /// Takes whatever is buffered used for VAD triggered cuts and for the
    /// final flush on stop and resets state for the next chunk
    fn take(&mut self) -> Vec<f32> {
        self.analyzed_samples = 0;
        self.has_speech = false;
        self.trailing_silence_frames = 0;
        std::mem::take(&mut self.buf)
    }
}

fn rms(frame: &[f32]) -> f32 {
    let sum_sq: f32 = frame.iter().map(|s| s * s).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

fn append_pcm16le(buf: &mut Vec<f32>, bytes: &[u8]) {
    for chunk in bytes.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        buf.push(sample as f32 / i16::MAX as f32);
    }
}

/// Sends the transcription result for one chunk and returns the text that
/// should become previous_text context for the next chunk or None when
/// there is nothing worth carrying forward like a suppressed empty partial
async fn emit_result(
    state: &AppState,
    sender: &mut SplitSink<WebSocket, Message>,
    samples: Vec<f32>,
    is_final: bool,
    previous_text: Option<&str>,
) -> Result<Option<String>, axum::Error> {
    let pcm = PcmBuffer {
        samples,
        format: AudioFormat::TARGET,
    };

    let result = match state.transcriber.transcribe(pcm, previous_text).await {
        Ok(result) => result,
        Err(rustyecho_core::TranscribeError::Overloaded) => {
            send_error(
                sender,
                "OVERLOADED",
                "server is overloaded, try again shortly",
            )
            .await?;
            return Ok(None);
        }
        Err(e) => {
            send_error(sender, "TRANSCRIBE_FAILED", &e.to_string()).await?;
            return Ok(None);
        }
    };

    // A Transcriber returns an empty string when a chunk turns out to be
    // silence or noise rather than speech see rustyecho-inference no speech
    // gating and we only suppress that for partials because an empty final still
    // tells the client the stream ended
    if !is_final && result.text.trim().is_empty() {
        return Ok(None);
    }

    let context = (!result.text.trim().is_empty()).then(|| result.text.clone());

    let msg = if is_final {
        ServerMessage::Final { text: result.text }
    } else {
        ServerMessage::Partial { text: result.text }
    };
    sender.send(to_ws_message(&msg)).await?;
    Ok(context)
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
