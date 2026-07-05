//! Manual load spike harness for `/v1/stream` not part of the automated
//! test suite see planning.md Milestone 4
//! Run against a real server
//!
//!   cargo run -p rustyecho-gateway --release
//!   cargo run -p rustyecho-gateway --example load_test --release -- 100 20 ws://127.0.0.1:8080/v1/stream
//!
//! Watch the server process memory and CPU in a separate terminal
//! This produces real numbers for the roadmap benchmark table instead of
//! unverified claims

use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::task::JoinSet;
use tokio_tungstenite::tungstenite::Message;

#[derive(Default)]
struct ConnStats {
    latencies: Vec<Duration>,
    errors: usize,
}

async fn run_connection(url: String, duration: Duration) -> ConnStats {
    let mut stats = ConnStats::default();

    let (mut ws, _) = match tokio_tungstenite::connect_async(&url).await {
        Ok(pair) => pair,
        Err(_) => {
            stats.errors += 1;
            return stats;
        }
    };

    if ws
        .send(Message::Text(
            r#"{"type":"start","sample_rate":16000,"channels":1}"#.to_string(),
        ))
        .await
        .is_err()
    {
        stats.errors += 1;
        return stats;
    }

    let start = Instant::now();
    while start.elapsed() < duration {
        let sent_at = Instant::now();
        let mut chunk = sine_pcm16le(500, 440.0);
        chunk.extend(silence_pcm16le(500));

        if ws.send(Message::Binary(chunk)).await.is_err() {
            stats.errors += 1;
            break;
        }

        match tokio::time::timeout(Duration::from_secs(5), ws.next()).await {
            Ok(Some(Ok(_))) => stats.latencies.push(sent_at.elapsed()),
            _ => {
                stats.errors += 1;
                break;
            }
        }
    }

    let _ = ws
        .send(Message::Text(r#"{"type":"stop"}"#.to_string()))
        .await;
    stats
}

fn sine_pcm16le(duration_ms: u64, freq_hz: f32) -> Vec<u8> {
    let n = (16_000 * duration_ms / 1000) as usize;
    let mut bytes = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / 16_000.0;
        let sample = (t * freq_hz * 2.0 * std::f32::consts::PI).sin() * 0.5;
        let value = (sample * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn silence_pcm16le(duration_ms: u64) -> Vec<u8> {
    vec![0u8; (16_000 * duration_ms / 1000) as usize * 2]
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_connections: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let duration_secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let url = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "ws://127.0.0.1:8080/v1/stream".to_string());

    println!("spawning {n_connections} connections to {url} for {duration_secs}s ...");

    let overall_start = Instant::now();
    let mut set = JoinSet::new();
    for _ in 0..n_connections {
        let url = url.clone();
        set.spawn(run_connection(url, Duration::from_secs(duration_secs)));
    }

    let mut all_latencies = Vec::new();
    let mut total_errors = 0usize;
    while let Some(res) = set.join_next().await {
        let stats = res.expect("connection task panicked");
        total_errors += stats.errors;
        all_latencies.extend(stats.latencies);
    }

    let elapsed = overall_start.elapsed();
    all_latencies.sort();
    let n = all_latencies.len();
    let avg = if n > 0 {
        all_latencies.iter().sum::<Duration>() / n as u32
    } else {
        Duration::ZERO
    };
    let p50 = all_latencies.get(n / 2).copied().unwrap_or_default();
    let p99 = all_latencies
        .get((n * 99) / 100)
        .copied()
        .unwrap_or_default();

    println!("--- results ---");
    println!("connections    : {n_connections}");
    println!("total duration : {:.1}s", elapsed.as_secs_f32());
    println!("total responses: {n}");
    println!("total errors   : {total_errors}");
    println!("latency avg    : {:.1}ms", avg.as_secs_f32() * 1000.0);
    println!("latency p50    : {:.1}ms", p50.as_secs_f32() * 1000.0);
    println!("latency p99    : {:.1}ms", p99.as_secs_f32() * 1000.0);
}
