//! WAV parsing and PCM normalization decodes WAV files and resamples or
//! downmixes them to `AudioFormat::TARGET` 16kHz mono that every
//! `Transcriber` expects

use rustyecho_core::{AudioFormat, GatewayError, PcmBuffer};
use std::io::Cursor;

pub const MAX_FILE_BYTES: usize = 25 * 1024 * 1024; // 25MB

/// Decode a WAV file into a normalized [`PcmBuffer`] at [`AudioFormat::TARGET`]
///
/// Rejects anything that is not 16-bit PCM and anything larger than
/// [`MAX_FILE_BYTES`] before the sample data is decoded
pub fn decode_wav(bytes: &[u8]) -> Result<PcmBuffer, GatewayError> {
    if bytes.len() > MAX_FILE_BYTES {
        return Err(GatewayError::FileTooLarge {
            max_mb: (MAX_FILE_BYTES / (1024 * 1024)) as u64,
        });
    }

    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| GatewayError::DecodeFailed(e.to_string()))?;
    let spec = reader.spec();

    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(GatewayError::InvalidAudioFormat(format!(
            "expected 16-bit PCM, got {:?} {}-bit",
            spec.sample_format, spec.bits_per_sample
        )));
    }
    if spec.channels == 0 {
        return Err(GatewayError::InvalidAudioFormat("zero channels".into()));
    }

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<_, _>>()
        .map_err(|e| GatewayError::DecodeFailed(e.to_string()))?;

    let mono = downmix_to_mono(&samples, spec.channels);
    let normalized = normalize_i16(&mono);

    let resampled = if spec.sample_rate == AudioFormat::TARGET.sample_rate {
        normalized
    } else {
        resample(&normalized, spec.sample_rate, AudioFormat::TARGET.sample_rate)
            .map_err(GatewayError::DecodeFailed)?
    };

    Ok(PcmBuffer {
        samples: resampled,
        format: AudioFormat::TARGET,
    })
}

fn downmix_to_mono(samples: &[i16], channels: u16) -> Vec<i16> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let channels = channels as usize;
    samples
        .chunks_exact(channels)
        .map(|frame| {
            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

fn normalize_i16(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect()
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, String> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )
    .map_err(|e| e.to_string())?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| e.to_string())?;
    Ok(waves_out.into_iter().next().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generates an in-memory sine wave WAV so tests do not depend on
    /// checked-in binary fixture files
    fn make_wav(sample_rate: u32, channels: u16, seconds: f32) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            let n = (sample_rate as f32 * seconds) as u32;
            for i in 0..n {
                let t = i as f32 / sample_rate as f32;
                let sample = (t * 440.0 * 2.0 * PI).sin();
                let value = (sample * i16::MAX as f32) as i16;
                for _ in 0..channels {
                    writer.write_sample(value).unwrap();
                }
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn decodes_mono_16k_to_target_format() {
        let wav = make_wav(16_000, 1, 1.0);
        let buf = decode_wav(&wav).unwrap();
        assert_eq!(buf.format, AudioFormat::TARGET);
        assert_eq!(buf.samples.len(), 16_000);
    }

    #[test]
    fn downmixes_stereo_and_resamples() {
        let wav = make_wav(44_100, 2, 1.0);
        let buf = decode_wav(&wav).unwrap();
        assert_eq!(buf.format, AudioFormat::TARGET);
        let expected = 16_000i64;
        let actual = buf.samples.len() as i64;
        assert!((actual - expected).abs() < 200, "got {actual} samples");
    }

    #[test]
    fn rejects_oversized_file() {
        let mut fake = vec![0u8; MAX_FILE_BYTES + 1];
        fake[0..4].copy_from_slice(b"RIFF");
        let err = decode_wav(&fake).unwrap_err();
        assert!(matches!(err, GatewayError::FileTooLarge { .. }));
    }

    #[test]
    fn rejects_corrupt_header() {
        let err = decode_wav(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, GatewayError::DecodeFailed(_)));
    }
}
