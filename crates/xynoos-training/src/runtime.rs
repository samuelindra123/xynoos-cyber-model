use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use serde_json::{Value, json};

use crate::{Result, RunIdentity, TrainingConfig, TrainingError, config::experiment_id};

#[derive(Debug, Clone)]
pub struct StopToken {
    requested: Arc<AtomicBool>,
    stop_file: Option<PathBuf>,
}

impl StopToken {
    pub fn new() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            stop_file: None,
        }
    }

    pub fn from_artifact_root(root: &Path) -> Result<Self> {
        validate_root(root)?;
        Ok(Self {
            requested: Arc::new(AtomicBool::new(false)),
            stop_file: Some(root.join("stop.request")),
        })
    }

    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    pub fn is_requested(&self) -> Result<bool> {
        if self.requested.load(Ordering::Acquire) {
            return Ok(true);
        }
        let Some(path) = &self.stop_file else {
            return Ok(false);
        };
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(true),
            Ok(_) => Err(TrainingError::InvalidConfiguration),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// External stop command helper. The runtime polls this regular file only
    /// at bounded safe points; blocked filesystem calls are not cancellable.
    pub fn write_request(root: &Path) -> Result<()> {
        validate_root(root)?;
        let path = root.join("stop.request");
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                file.write_all(b"stop\n")?;
                file.sync_all()?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(&path)?;
                if metadata.is_file() && !metadata.file_type().is_symlink() {
                    Ok(())
                } else {
                    Err(TrainingError::InvalidConfiguration)
                }
            }
            Err(error) => Err(error.into()),
        }
    }
}

impl Default for StopToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming JSONL writer. Every event is self-identifying and excludes input
/// text, host names, and personal machine details.
pub struct RunLogWriter<W: Write> {
    writer: W,
    experiment_id: String,
    source_revision: String,
    config_digest: [u8; 32],
    dataset_id: String,
    dataset_digest: [u8; 32],
    tokenizer_digest: [u8; 32],
    started: Instant,
}

impl<W: Write> RunLogWriter<W> {
    pub fn new(writer: W, identity: &RunIdentity, config: TrainingConfig) -> Self {
        Self {
            writer,
            experiment_id: experiment_id(identity, config),
            source_revision: identity.source_revision().to_owned(),
            config_digest: config.digest(identity),
            dataset_id: identity.dataset_id().to_owned(),
            dataset_digest: identity.dataset_manifest_sha256(),
            tokenizer_digest: identity.tokenizer_sha256(),
            started: Instant::now(),
        }
    }

    pub fn write_event(&mut self, kind: &str, fields: Value) -> Result<()> {
        let mut record = match fields {
            Value::Object(map) => map,
            _ => return Err(TrainingError::InvalidConfiguration),
        };
        record.insert("schema_version".into(), json!(1));
        record.insert("type".into(), json!(kind));
        record.insert("experiment_id".into(), json!(self.experiment_id));
        record.insert("source_sha".into(), json!(self.source_revision));
        record.insert("config_sha256".into(), json!(hex(&self.config_digest)));
        record.insert("model_id".into(), json!(crate::MODEL_ID));
        record.insert("dataset_id".into(), json!(self.dataset_id));
        record.insert("dataset_sha256".into(), json!(hex(&self.dataset_digest)));
        record.insert(
            "tokenizer_sha256".into(),
            json!(hex(&self.tokenizer_digest)),
        );
        record.insert(
            "elapsed_ns".into(),
            json!(self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64),
        );
        serde_json::to_writer(&mut self.writer, &Value::Object(record))
            .map_err(|_| TrainingError::AllocationFailed)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().map_err(Into::into)
    }
    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunHandoff {
    pub experiment_id: String,
    pub source_sha: String,
    pub config_sha256: String,
    pub dataset_sha256: String,
    pub tokenizer_sha256: String,
    pub run_identity: String,
    pub checkpoint_path: String,
    pub metrics_path: String,
    pub stop_instructions: String,
    pub max_updates: u64,
    pub consumed_token_cap: u64,
    pub supervised_token_cap: u64,
    pub cumulative_work_cap: u64,
    pub owned_memory_bytes: usize,
    pub active_runtime_cap_ns: u64,
}

impl RunHandoff {
    pub fn for_run(
        identity: &RunIdentity,
        config: TrainingConfig,
        checkpoint_path: String,
        metrics_path: String,
    ) -> Self {
        let limits = config.limits();
        let digest = config.digest(identity);
        Self {
            experiment_id: experiment_id(identity, config),
            source_sha: identity.source_revision().to_owned(),
            config_sha256: hex(&digest),
            dataset_sha256: hex(&identity.dataset_manifest_sha256()),
            tokenizer_sha256: hex(&identity.tokenizer_sha256()),
            run_identity: hex(&digest),
            checkpoint_path,
            metrics_path,
            stop_instructions: "Create stop.request in the artifact root; the runtime stops at the next safe boundary.".into(),
            max_updates: limits.max_updates(),
            consumed_token_cap: limits.consumed_tokens(),
            supervised_token_cap: limits.supervised_tokens(),
            cumulative_work_cap: limits.cumulative_work(),
            owned_memory_bytes: limits.owned_memory_bytes(),
            active_runtime_cap_ns: limits.max_runtime().as_nanos() as u64,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "source_sha": self.source_sha,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "run_identity": self.run_identity,
            "checkpoint_path": self.checkpoint_path,
            "metrics_path": self.metrics_path,
            "stop_instructions": self.stop_instructions,
            "budget": {
                "max_updates": self.max_updates,
                "consumed_token_cap": self.consumed_token_cap,
                "supervised_token_cap": self.supervised_token_cap,
                "cumulative_work_cap": self.cumulative_work_cap,
                "owned_memory_bytes": self.owned_memory_bytes,
                "active_runtime_cap_ns": self.active_runtime_cap_ns,
            }
        })
    }

    pub fn write_atomic(&self, root: &Path) -> Result<()> {
        validate_root(root)?;
        let path = root.join("handoff.json");
        let parent = root;
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
            Ok(_) => return Err(TrainingError::InvalidConfiguration),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let name = path
            .file_name()
            .ok_or(TrainingError::InvalidConfiguration)?;
        if !name
            .to_string_lossy()
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        let temp = parent.join(format!(
            ".handoff-{}-{}.tmp",
            std::process::id(),
            self.experiment_id
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let result = (|| {
            serde_json::to_writer(&mut file, &self.to_json())
                .map_err(|_| TrainingError::AllocationFailed)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            fs::rename(&temp, &path)?;
            File::open(parent)?
                .sync_all()
                .map_err(|_| TrainingError::PublicationDurabilityUncertain)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}

pub fn process_vm_hwm_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

pub fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(text, "{byte:02x}");
    }
    text
}

pub(crate) fn validate_root(root: &Path) -> Result<()> {
    if !root.is_absolute() {
        return Err(TrainingError::InvalidConfiguration);
    }
    let mut current = PathBuf::new();
    for component in root.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(TrainingError::InvalidConfiguration);
        }
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(TrainingError::InvalidConfiguration);
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn duration_ns(duration: Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).map_err(|_| TrainingError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "xyruntime-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn stop_requests_are_observed_at_a_safe_point_and_symlinks_reject() {
        let path = root("stop");
        let stop = StopToken::from_artifact_root(&path).unwrap();
        assert!(!stop.is_requested().unwrap());
        StopToken::write_request(&path).unwrap();
        assert!(stop.is_requested().unwrap());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            fs::remove_file(path.join("stop.request")).unwrap();
            fs::write(path.join("target"), b"target").unwrap();
            symlink(path.join("target"), path.join("stop.request")).unwrap();
            assert_eq!(
                stop.is_requested(),
                Err(TrainingError::InvalidConfiguration)
            );
            assert_eq!(
                StopToken::write_request(&path),
                Err(TrainingError::InvalidConfiguration)
            );
        }
        fs::remove_dir_all(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn artifact_roots_reject_parent_traversal() {
        assert_eq!(
            validate_root(Path::new("/tmp/../tmp")),
            Err(TrainingError::InvalidConfiguration)
        );
    }
}
