use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};

use crate::{CheckpointCounters, Result, RunLimits, TrainingError, runtime::validate_root};

const MAGIC: &[u8; 8] = b"XYNOOSJ1";
const PREFIX: usize = 96;
const FILE_BYTES: usize = PREFIX + 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenMode {
    CreateIfMissing,
    RequireExisting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalProgress {
    pub attempted_consumed_tokens: u64,
    pub attempted_supervised_tokens: u64,
    pub cumulative_work: u64,
    pub active_runtime_ns: u64,
    pub reserved_runtime_ns: u64,
    pub attempted_updates: u64,
    pub checkpoint_sequence: u64,
}

#[derive(Debug)]
pub struct RunJournal {
    root: PathBuf,
    path: PathBuf,
    _lock: LockFile,
    config_digest: [u8; 32],
    progress: JournalProgress,
    limits: RunLimits,
    attempt_base_runtime_ns: u64,
    attempt_reservation_ns: u64,
    attempt_started: Option<Instant>,
}

impl RunJournal {
    pub fn open(
        root: &Path,
        config_digest: [u8; 32],
        checkpoint: CheckpointCounters,
        limits: RunLimits,
    ) -> Result<Self> {
        Self::open_mode(
            root,
            config_digest,
            checkpoint,
            limits,
            OpenMode::CreateIfMissing,
        )
    }

    /// Open a lineage for resume. A missing journal is an error: checkpoint
    /// counters cannot reconstruct precharged work that happened after it.
    pub fn open_existing(
        root: &Path,
        config_digest: [u8; 32],
        checkpoint: CheckpointCounters,
        limits: RunLimits,
    ) -> Result<Self> {
        Self::open_mode(
            root,
            config_digest,
            checkpoint,
            limits,
            OpenMode::RequireExisting,
        )
    }

    fn open_mode(
        root: &Path,
        config_digest: [u8; 32],
        checkpoint: CheckpointCounters,
        limits: RunLimits,
        mode: OpenMode,
    ) -> Result<Self> {
        validate_root(root)?;
        let lock_path = root.join(".run.lock");
        let lock = acquire_lock(&lock_path)?;
        let path = root.join("run.journal");
        validate_regular_or_missing(&path)?;
        let loaded = read_journal(&path, config_digest)?;
        if loaded.is_none() && mode == OpenMode::RequireExisting {
            return Err(TrainingError::Checkpoint("run_journal_missing_for_resume"));
        }
        let mut progress = loaded.unwrap_or(JournalProgress {
            attempted_consumed_tokens: checkpoint.attempted_consumed_tokens,
            attempted_supervised_tokens: checkpoint.attempted_supervised_tokens,
            cumulative_work: checkpoint.cumulative_work,
            active_runtime_ns: checkpoint.cumulative_runtime_ns,
            reserved_runtime_ns: 0,
            attempted_updates: checkpoint.attempted_updates,
            checkpoint_sequence: checkpoint.checkpoint_sequence,
        });
        validate_progress(progress, checkpoint, limits)?;
        if progress.reserved_runtime_ns != 0 {
            progress.active_runtime_ns = progress
                .active_runtime_ns
                .checked_add(progress.reserved_runtime_ns)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            progress.reserved_runtime_ns = 0;
        }
        let journal = Self {
            root: root.to_path_buf(),
            path,
            _lock: lock,
            config_digest,
            progress,
            limits,
            attempt_base_runtime_ns: 0,
            attempt_reservation_ns: 0,
            attempt_started: None,
        };
        journal.persist()?;
        Ok(journal)
    }

    /// Read and validate a lineage without acquiring a run lock or writing it.
    pub fn inspect(
        root: &Path,
        config_digest: [u8; 32],
        limits: RunLimits,
    ) -> Result<JournalProgress> {
        validate_root(root)?;
        let path = root.join("run.journal");
        validate_regular_or_missing(&path)?;
        let progress = read_journal(&path, config_digest)?
            .ok_or(TrainingError::Checkpoint("run_journal_missing"))?;
        validate_progress(progress, CheckpointCounters::default(), limits)?;
        Ok(progress)
    }

    pub fn progress(&self) -> JournalProgress {
        self.progress
    }

    pub fn reconcile_checkpoint(&mut self, checkpoint: CheckpointCounters) -> Result<()> {
        validate_progress(self.progress, checkpoint, self.limits)?;
        Ok(())
    }

    /// Reserve at most one attempt's full runtime before admitting any work.
    /// A crash leaves the reservation in the journal and resume charges it in
    /// full, so process restarts cannot refund unrecorded runtime.
    pub fn begin_attempt(&mut self) -> Result<()> {
        if self.attempt_started.is_some() || self.progress.reserved_runtime_ns != 0 {
            return Err(TrainingError::InvalidConfiguration);
        }
        let total_cap = duration_ns(self.limits.max_runtime())?;
        let attempt_cap = duration_ns(self.limits.max_attempt_runtime())?;
        let remaining = total_cap.saturating_sub(self.progress.active_runtime_ns);
        let reservation = remaining.min(attempt_cap);
        if reservation == 0 {
            return Err(TrainingError::RuntimeLimit {
                limit: self.limits.max_runtime(),
            });
        }
        self.attempt_base_runtime_ns = self.progress.active_runtime_ns;
        self.attempt_reservation_ns = reservation;
        self.progress.reserved_runtime_ns = reservation;
        self.attempt_started = Some(Instant::now());
        self.persist()
    }

    /// Persist cumulative input, supervision, work, and current active-time
    /// accounting before the candidate policy admits the operation.
    pub fn precharge_microbatch(
        &mut self,
        consumed: u64,
        supervised: u64,
        work: u64,
    ) -> Result<()> {
        self.sync_elapsed(false)?;
        let next_consumed = self
            .progress
            .attempted_consumed_tokens
            .checked_add(consumed)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        let next_supervised = self
            .progress
            .attempted_supervised_tokens
            .checked_add(supervised)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        let next_work = self
            .progress
            .cumulative_work
            .checked_add(work)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if next_consumed > self.limits.consumed_tokens() {
            return Err(TrainingError::BudgetExhausted("consumed_tokens"));
        }
        if next_supervised > self.limits.supervised_tokens() {
            return Err(TrainingError::BudgetExhausted("supervised_tokens"));
        }
        if next_work > self.limits.cumulative_work() {
            return Err(TrainingError::BudgetExhausted("cumulative_work"));
        }
        self.progress.attempted_consumed_tokens = next_consumed;
        self.progress.attempted_supervised_tokens = next_supervised;
        self.progress.cumulative_work = next_work;
        self.persist()
    }

    pub fn precharge_update(&mut self, work: u64) -> Result<()> {
        self.sync_elapsed(false)?;
        let next_work = self
            .progress
            .cumulative_work
            .checked_add(work)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        let next_updates = self
            .progress
            .attempted_updates
            .checked_add(1)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if next_work > self.limits.cumulative_work() {
            return Err(TrainingError::BudgetExhausted("cumulative_work"));
        }
        if next_updates > self.limits.max_updates() {
            return Err(TrainingError::BudgetExhausted("optimizer_updates"));
        }
        self.progress.cumulative_work = next_work;
        self.progress.attempted_updates = next_updates;
        self.persist()
    }

    pub fn precharge_auxiliary_work(&mut self, work: u64) -> Result<()> {
        self.sync_elapsed(false)?;
        let next_work = self
            .progress
            .cumulative_work
            .checked_add(work)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if next_work > self.limits.cumulative_work() {
            return Err(TrainingError::BudgetExhausted("cumulative_work"));
        }
        self.progress.cumulative_work = next_work;
        self.persist()
    }

    pub fn next_checkpoint_sequence(&mut self) -> Result<u64> {
        let next = self
            .progress
            .checkpoint_sequence
            .checked_add(1)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        self.progress.checkpoint_sequence = next;
        self.persist()?;
        Ok(next)
    }

    /// Flush actual active runtime while retaining the unspent part of the
    /// crash reservation. Call immediately before checkpoint publication.
    pub fn sync_runtime(&mut self) -> Result<u64> {
        self.sync_elapsed(false)?;
        Ok(self.progress.active_runtime_ns)
    }

    /// Finalize an orderly attempt with measured active time. If the process
    /// dies before this call, the outstanding reservation is charged on resume.
    pub fn finish_attempt(&mut self) -> Result<u64> {
        self.sync_elapsed(true)?;
        self.attempt_started = None;
        self.attempt_reservation_ns = 0;
        Ok(self.progress.active_runtime_ns)
    }

    fn sync_elapsed(&mut self, finish: bool) -> Result<()> {
        let Some(started) = self.attempt_started else {
            return Ok(());
        };
        let elapsed = duration_ns(started.elapsed())?;
        self.progress.active_runtime_ns = self
            .attempt_base_runtime_ns
            .checked_add(elapsed)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        self.progress.reserved_runtime_ns = if finish {
            0
        } else {
            self.attempt_reservation_ns.saturating_sub(elapsed)
        };
        self.persist()
    }

    fn persist(&self) -> Result<()> {
        let mut bytes = [0_u8; FILE_BYTES];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..40].copy_from_slice(&self.config_digest);
        let values = [
            self.progress.attempted_consumed_tokens,
            self.progress.attempted_supervised_tokens,
            self.progress.cumulative_work,
            self.progress.active_runtime_ns,
            self.progress.reserved_runtime_ns,
            self.progress.attempted_updates,
            self.progress.checkpoint_sequence,
        ];
        for (index, value) in values.into_iter().enumerate() {
            let start = 40 + index * 8;
            bytes[start..start + 8].copy_from_slice(&value.to_le_bytes());
        }
        let digest: [u8; 32] = Sha256::digest(&bytes[..PREFIX]).into();
        bytes[PREFIX..].copy_from_slice(&digest);
        write_atomic(&self.root, &self.path, &bytes)
    }
}

fn validate_progress(
    progress: JournalProgress,
    checkpoint: CheckpointCounters,
    limits: RunLimits,
) -> Result<()> {
    let max_runtime = duration_ns(limits.max_runtime())?;
    let active_and_reserved = progress
        .active_runtime_ns
        .checked_add(progress.reserved_runtime_ns)
        .ok_or(TrainingError::ArithmeticOverflow)?;
    if progress.attempted_consumed_tokens < checkpoint.attempted_consumed_tokens
        || progress.attempted_supervised_tokens < checkpoint.attempted_supervised_tokens
        || progress.cumulative_work < checkpoint.cumulative_work
        || progress.active_runtime_ns < checkpoint.cumulative_runtime_ns
        || progress.attempted_updates < checkpoint.attempted_updates
        || progress.checkpoint_sequence < checkpoint.checkpoint_sequence
        || progress.attempted_supervised_tokens > progress.attempted_consumed_tokens
        || progress.attempted_consumed_tokens > limits.consumed_tokens()
        || progress.attempted_supervised_tokens > limits.supervised_tokens()
        || progress.cumulative_work > limits.cumulative_work()
        || progress.attempted_updates > limits.max_updates()
        || progress.reserved_runtime_ns > duration_ns(limits.max_attempt_runtime())?
        || active_and_reserved > max_runtime
    {
        return Err(TrainingError::Checkpoint("run_journal_counter_progression"));
    }
    Ok(())
}

#[derive(Debug)]
struct LockFile {
    path: PathBuf,
    _file: File,
}

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn read_journal(path: &Path, digest: [u8; 32]) -> Result<Option<JournalProgress>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() != FILE_BYTES as u64
    {
        return Err(TrainingError::Checkpoint("invalid_run_journal_file"));
    }
    let mut bytes = [0_u8; FILE_BYTES];
    File::open(path)?.read_exact(&mut bytes)?;
    let expected: [u8; 32] = Sha256::digest(&bytes[..PREFIX]).into();
    if &bytes[..8] != MAGIC || bytes[8..40] != digest || bytes[PREFIX..] != expected {
        return Err(TrainingError::Checkpoint(
            "run_journal_identity_or_checksum",
        ));
    }
    let mut values = [0_u64; 7];
    for (index, value) in values.iter_mut().enumerate() {
        let start = 40 + index * 8;
        *value = u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .map_err(|_| TrainingError::Checkpoint("run_journal_field"))?,
        );
    }
    Ok(Some(JournalProgress {
        attempted_consumed_tokens: values[0],
        attempted_supervised_tokens: values[1],
        cumulative_work: values[2],
        active_runtime_ns: values[3],
        reserved_runtime_ns: values[4],
        attempted_updates: values[5],
        checkpoint_sequence: values[6],
    }))
}

fn write_atomic(root: &Path, target: &Path, bytes: &[u8]) -> Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    validate_regular_or_missing(target)?;
    let serial = NEXT.fetch_add(1, Ordering::Relaxed);
    let temp = root.join(format!(".run-journal-{}-{serial}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp, target)?;
        File::open(root)?
            .sync_all()
            .map_err(|_| TrainingError::PublicationDurabilityUncertain)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn validate_regular_or_missing(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(TrainingError::InvalidConfiguration),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn acquire_lock(path: &Path) -> Result<LockFile> {
    for _ in 0..2 {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                file.sync_all()?;
                return Ok(LockFile {
                    path: path.to_path_buf(),
                    _file: file,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(path)?;
                if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 32 {
                    return Err(TrainingError::InvalidConfiguration);
                }
                let pid = fs::read_to_string(path)?
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| TrainingError::InvalidConfiguration)?;
                if Path::new("/proc").join(pid.to_string()).exists() {
                    return Err(TrainingError::InvalidConfiguration);
                }
                fs::remove_file(path)?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(TrainingError::InvalidConfiguration)
}

fn duration_ns(value: Duration) -> Result<u64> {
    u64::try_from(value.as_nanos()).map_err(|_| TrainingError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn root() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "xyjournal-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn limits() -> RunLimits {
        RunLimits::new(
            1,
            2,
            1000,
            500,
            10_000,
            128 * 1024 * 1024,
            Duration::from_secs(20),
            Duration::from_secs(10),
            1,
            1,
        )
        .unwrap()
    }

    #[test]
    fn admitted_work_survives_crash_and_runtime_reservation_is_not_refunded() {
        let root = root();
        let digest = [0x6b; 32];
        let mut journal =
            RunJournal::open(&root, digest, CheckpointCounters::default(), limits()).unwrap();
        journal.begin_attempt().unwrap();
        journal.precharge_microbatch(32, 20, 300).unwrap();
        let pre_crash = journal.progress();
        assert_eq!(pre_crash.attempted_consumed_tokens, 32);
        assert_eq!(pre_crash.attempted_supervised_tokens, 20);
        assert_eq!(pre_crash.cumulative_work, 300);
        assert!(pre_crash.reserved_runtime_ns > 0);
        drop(journal);

        let recovered =
            RunJournal::open(&root, digest, CheckpointCounters::default(), limits()).unwrap();
        let after_crash = recovered.progress();
        assert_eq!(after_crash.attempted_consumed_tokens, 32);
        assert_eq!(after_crash.attempted_supervised_tokens, 20);
        assert_eq!(after_crash.cumulative_work, 300);
        assert_eq!(
            after_crash.active_runtime_ns,
            Duration::from_secs(10).as_nanos() as u64
        );
        assert_eq!(after_crash.reserved_runtime_ns, 0);
        drop(recovered);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn journal_is_bound_to_configuration_and_rejects_symlinks() {
        let root = root();
        let journal =
            RunJournal::open(&root, [1; 32], CheckpointCounters::default(), limits()).unwrap();
        drop(journal);
        assert!(matches!(
            RunJournal::open(&root, [2; 32], CheckpointCounters::default(), limits()),
            Err(TrainingError::Checkpoint(
                "run_journal_identity_or_checksum"
            ))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_and_status_fail_closed_when_journal_is_missing() {
        let root = root();
        assert!(matches!(
            RunJournal::open_existing(&root, [8; 32], CheckpointCounters::default(), limits()),
            Err(TrainingError::Checkpoint("run_journal_missing_for_resume"))
        ));
        assert!(!root.join("run.journal").exists());
        assert!(matches!(
            RunJournal::inspect(&root, [8; 32], limits()),
            Err(TrainingError::Checkpoint("run_journal_missing"))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn older_checkpoint_cannot_refund_journal_precharges_or_runtime_reservation() {
        let root = root();
        let digest = [0x42; 32];
        let mut journal =
            RunJournal::open(&root, digest, CheckpointCounters::default(), limits()).unwrap();
        journal.begin_attempt().unwrap();
        journal.precharge_microbatch(32, 20, 300).unwrap();
        drop(journal);

        let recovered =
            RunJournal::open_existing(&root, digest, CheckpointCounters::default(), limits())
                .unwrap();
        assert_eq!(recovered.progress().attempted_consumed_tokens, 32);
        assert_eq!(recovered.progress().attempted_supervised_tokens, 20);
        assert_eq!(recovered.progress().cumulative_work, 300);
        assert_eq!(
            recovered.progress().active_runtime_ns,
            Duration::from_secs(10).as_nanos() as u64
        );
        assert_eq!(recovered.progress().reserved_runtime_ns, 0);
        drop(recovered);

        let incompatible = CheckpointCounters {
            attempted_consumed_tokens: 33,
            ..CheckpointCounters::default()
        };
        assert!(matches!(
            RunJournal::open_existing(&root, digest, incompatible, limits()),
            Err(TrainingError::Checkpoint("run_journal_counter_progression"))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checksum_valid_invalid_counter_progression_is_rejected_read_only() {
        let root = root();
        let digest = [0x6a; 32];
        drop(RunJournal::open(&root, digest, CheckpointCounters::default(), limits()).unwrap());
        let path = root.join("run.journal");
        let mut bytes = fs::read(&path).unwrap();
        // attempted_supervised_tokens > attempted_consumed_tokens.
        bytes[48..56].copy_from_slice(&1_u64.to_le_bytes());
        let checksum: [u8; 32] = Sha256::digest(&bytes[..PREFIX]).into();
        bytes[PREFIX..].copy_from_slice(&checksum);
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            RunJournal::inspect(&root, digest, limits()),
            Err(TrainingError::Checkpoint("run_journal_counter_progression"))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn simultaneous_live_lineage_open_is_rejected() {
        let root = root();
        let live =
            RunJournal::open(&root, [0x71; 32], CheckpointCounters::default(), limits()).unwrap();
        assert_eq!(
            RunJournal::open_existing(&root, [0x71; 32], CheckpointCounters::default(), limits())
                .unwrap_err(),
            TrainingError::InvalidConfiguration
        );
        drop(live);
        fs::remove_dir_all(root).unwrap();
    }
}
