//! Domain types shared across the RustyEcho gateway and audio crates
//!
//! `Transcriber` is the seam between Phase 1 I/O gateway and Phase 2
//! inference engine so the gateway only ever depends on this trait and never
//! on `candle` directly so the real engine can be swapped in later without
//! touching any gateway code

/// Sample rate and channel count of a [`PcmBuffer`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioFormat {
    /// Format every buffer is normalized to before reaching a `Transcriber`
    pub const TARGET: AudioFormat = AudioFormat {
        sample_rate: 16_000,
        channels: 1,
    };
}

/// Normalized PCM audio interleaved samples in `-1.0..=1.0` matching `format`
#[derive(Debug, Clone)]
pub struct PcmBuffer {
    pub samples: Vec<f32>,
    pub format: AudioFormat,
}

impl PcmBuffer {
    pub fn duration_ms(&self) -> u64 {
        if self.format.sample_rate == 0 || self.format.channels == 0 {
            return 0;
        }
        let frames = self.samples.len() as u64 / self.format.channels as u64;
        frames * 1000 / self.format.sample_rate as u64
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub is_final: bool,
    pub confidence: Option<f32>,
}

/// Errors surfaced by the gateway REST or WS layer each mapped to an HTTP
/// status or WS error code at the edge never exposed as raw Rust errors
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("unsupported audio format: {0}")]
    InvalidAudioFormat(String),
    #[error("file exceeds maximum size of {max_mb}MB")]
    FileTooLarge { max_mb: u64 },
    #[error("failed to decode audio: {0}")]
    DecodeFailed(String),
    #[error("operation timed out")]
    Timeout,
    #[error("transcription failed: {0}")]
    TranscribeFailed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error("transcription backend error: {0}")]
    Backend(String),
}

#[async_trait::async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, chunk: PcmBuffer) -> Result<TranscriptionResult, TranscribeError>;
}

/// Placeholder used throughout Phase 1 so the I/O pipeline can be built and
/// tested end to end before the real inference engine Phase 2 exists
pub struct MockTranscriber;

#[async_trait::async_trait]
impl Transcriber for MockTranscriber {
    async fn transcribe(&self, chunk: PcmBuffer) -> Result<TranscriptionResult, TranscribeError> {
        Ok(TranscriptionResult {
            text: format!("[mock transcription of {}ms audio]", chunk.duration_ms()),
            is_final: true,
            confidence: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_ms_mono() {
        let buf = PcmBuffer {
            samples: vec![0.0; 16_000],
            format: AudioFormat {
                sample_rate: 16_000,
                channels: 1,
            },
        };
        assert_eq!(buf.duration_ms(), 1000);
    }

    #[test]
    fn duration_ms_stereo() {
        let buf = PcmBuffer {
            samples: vec![0.0; 32_000], // 16_000 frames * 2 channels
            format: AudioFormat {
                sample_rate: 16_000,
                channels: 2,
            },
        };
        assert_eq!(buf.duration_ms(), 1000);
    }

    #[tokio::test]
    async fn mock_transcriber_returns_final() {
        let t = MockTranscriber;
        let buf = PcmBuffer {
            samples: vec![0.0; 8_000],
            format: AudioFormat::TARGET,
        };
        let result = t.transcribe(buf).await.unwrap();
        assert!(result.is_final);
    }
}
