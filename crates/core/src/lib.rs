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
    #[error("service overloaded, try again shortly")]
    Overloaded,
}

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error("transcription backend error: {0}")]
    Backend(String),
    #[error("transcription backend is overloaded")]
    Overloaded,
}

impl From<TranscribeError> for GatewayError {
    fn from(err: TranscribeError) -> Self {
        match err {
            TranscribeError::Overloaded => GatewayError::Overloaded,
            TranscribeError::Backend(msg) => GatewayError::TranscribeFailed(msg),
        }
    }
}

#[async_trait::async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, chunk: PcmBuffer) -> Result<TranscriptionResult, TranscribeError>;
}

/// Wraps any `Transcriber` with a hard cap on in-flight requests, so a burst
/// beyond that cap fails fast with `TranscribeError::Overloaded` instead of
/// queueing silently and unboundedly behind whatever concurrency limit the
/// inner transcriber happens to have (e.g. a worker pool).
///
/// A request waits up to `acquire_timeout` for a free slot before giving up
/// — this bounds worst-case latency under overload instead of leaving
/// callers to guess how long a request might sit queued.
pub struct BoundedTranscriber<T> {
    inner: T,
    semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    acquire_timeout: std::time::Duration,
}

impl<T: Transcriber> BoundedTranscriber<T> {
    pub fn new(inner: T, max_in_flight: usize, acquire_timeout: std::time::Duration) -> Self {
        Self {
            inner,
            semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(max_in_flight.max(1))),
            acquire_timeout,
        }
    }
}

#[async_trait::async_trait]
impl<T: Transcriber> Transcriber for BoundedTranscriber<T> {
    async fn transcribe(&self, chunk: PcmBuffer) -> Result<TranscriptionResult, TranscribeError> {
        let _permit = tokio::time::timeout(self.acquire_timeout, self.semaphore.acquire())
            .await
            .map_err(|_| TranscribeError::Overloaded)?
            .expect("semaphore is never closed");

        self.inner.transcribe(chunk).await
    }
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

    struct SlowTranscriber {
        delay: std::time::Duration,
    }

    #[async_trait::async_trait]
    impl Transcriber for SlowTranscriber {
        async fn transcribe(&self, _chunk: PcmBuffer) -> Result<TranscriptionResult, TranscribeError> {
            tokio::time::sleep(self.delay).await;
            Ok(TranscriptionResult {
                text: String::new(),
                is_final: true,
                confidence: None,
            })
        }
    }

    fn empty_buf() -> PcmBuffer {
        PcmBuffer {
            samples: vec![],
            format: AudioFormat::TARGET,
        }
    }

    #[tokio::test]
    async fn bounded_transcriber_rejects_when_saturated() {
        use std::{sync::Arc, time::Duration};

        let bounded = Arc::new(BoundedTranscriber::new(
            SlowTranscriber {
                delay: Duration::from_millis(200),
            },
            1, // only one in-flight request allowed
            Duration::from_millis(50), // much shorter than the slow transcribe
        ));

        // Occupy the only slot for 200ms.
        let occupier = {
            let bounded = bounded.clone();
            tokio::spawn(async move { bounded.transcribe(empty_buf()).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;

        // A second request can't get a permit within 50ms, so it should
        // fail fast with Overloaded rather than waiting behind the first.
        let rejected = bounded.transcribe(empty_buf()).await;
        assert!(matches!(rejected, Err(TranscribeError::Overloaded)));

        // The first request should still complete successfully once its
        // slot frees up -- saturation rejects new work, it doesn't break
        // work already in flight.
        let occupier_result = occupier.await.unwrap();
        assert!(occupier_result.is_ok());
    }
}
