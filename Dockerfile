# syntax=docker/dockerfile:1

########## Builder ##########
FROM rust:1-slim-bookworm AS builder
WORKDIR /app

# build-essential: cc linker needed by several dependency build scripts.
# pkg-config/libssl-dev: ureq's TLS backend (used by hf-hub for downloads).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo build --release -p rustyecho-gateway

# Warm the Hugging Face cache for the default model at build time so the
# runtime container is self-contained and doesn't need network access (or
# pay download latency) for the default model on first request. Overriding
# WHISPER_MODEL_ID/WHISPER_REVISION at `docker run` time still works -- that
# just falls back to downloading fresh into this same cache path at startup.
ENV HF_HOME=/app/hf-cache
RUN cargo run --release -p rustyecho-inference --example prefetch_model

########## Runtime ##########
FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --shell /usr/sbin/nologin rustyecho

COPY --from=builder /app/target/release/rustyecho-gateway /usr/local/bin/rustyecho-gateway
COPY --from=builder --chown=rustyecho:rustyecho /app/hf-cache /home/rustyecho/.cache/huggingface

ENV HF_HOME=/home/rustyecho/.cache/huggingface
ENV PORT=8080
USER rustyecho

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl --fail "http://localhost:${PORT}/healthz" || exit 1

ENTRYPOINT ["/usr/local/bin/rustyecho-gateway"]
