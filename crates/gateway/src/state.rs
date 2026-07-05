use std::sync::Arc;

use rustyecho_core::Transcriber;

#[derive(Clone)]
pub struct AppState {
    pub transcriber: Arc<dyn Transcriber>,
    pub max_upload_bytes: usize,
}
