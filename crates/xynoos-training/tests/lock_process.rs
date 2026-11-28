use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use xynoos_training::{
    AdvisoryLock, CheckpointCounters, LockError, RunJournal, RunLimits, TrainingError,
};

const WORKER_ROOT: &str = "XYNOOS_LOCK_TEST_ROOT";
const WORKER_GATE: &str = "XYNOOS_LOCK_TEST_GATE";
const WORKER_RELEASE: &str = "XYNOOS_LOCK_TEST_RELEASE";
const WORKER_READY: &str = "XYNOOS_LOCK_TEST_READY";
const WORKER_RESULT: &str = "XYNOOS_LOCK_TEST_RESULT";
const WORKER_MODE: &str = "XYNOOS_LOCK_TEST_MODE";
const DIGEST: [u8; 32] = [0x5A; 32];

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "task006-lock-process-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn limits() -> RunLimits {
    RunLimits::new(
        1,
        10,
        320,
        320,
        8_000_000_000,
        64 * 1024 * 1024,
        Duration::from_secs(30),
        Duration::from_secs(15),
        1,
        1,
    )
    .unwrap()
}

fn wait_until(mut predicate: impl FnMut() -> bool) {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(10) {
        if predicate() {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
    panic!("condition was not observed within ten seconds");
}

fn spawn_worker(
    root: &Path,
    gate: &Path,
    release: &Path,
    ready: &Path,
    result: &Path,
    mode: &str,
) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "process_lock_worker_entrypoint", "--nocapture"])
        .env(WORKER_ROOT, root)
        .env(WORKER_GATE, gate)
        .env(WORKER_RELEASE, release)
        .env(WORKER_READY, ready)
        .env(WORKER_RESULT, result)
        .env(WORKER_MODE, mode)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}

#[test]
fn process_lock_worker_entrypoint() {
    let Ok(root) = std::env::var(WORKER_ROOT) else {
        return;
    };
    let gate = PathBuf::from(std::env::var(WORKER_GATE).unwrap());
    let release = PathBuf::from(std::env::var(WORKER_RELEASE).unwrap());
    let ready = PathBuf::from(std::env::var(WORKER_READY).unwrap());
    let result = PathBuf::from(std::env::var(WORKER_RESULT).unwrap());
    wait_until(|| gate.exists());
    if std::env::var(WORKER_MODE)
        .as_deref()
        .is_ok_and(|mode| mode == "kernel-only")
    {
        let lock = AdvisoryLock::acquire(&Path::new(&root).join(".run.lock")).unwrap();
        fs::write(&ready, std::process::id().to_string()).unwrap();
        wait_until(|| release.exists());
        drop(lock);
        fs::write(&result, "owner").unwrap();
        return;
    }
    match RunJournal::open_existing(
        Path::new(&root),
        DIGEST,
        CheckpointCounters::default(),
        limits(),
    ) {
        Ok(journal) => {
            fs::write(&ready, std::process::id().to_string()).unwrap();
            wait_until(|| release.exists());
            drop(journal);
            fs::write(&result, "owner").unwrap();
        }
        Err(TrainingError::RunLockContended) => fs::write(&result, "contended").unwrap(),
        Err(error) => panic!("unexpected lock worker failure: {error:?}"),
    }
}

#[test]
fn repeated_multiprocess_recovery_has_one_owner_per_lineage() {
    let scratch = Scratch::new();
    let root = scratch.0.join("lineage");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    drop(RunJournal::open(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap());

    for round in 0..20 {
        // A previous PID record is deliberately stale. It is not the lock
        // authority and must not permit a process to unlink the lock inode.
        fs::write(root.join(".run.lock"), format!("stale-pid-{}\n", u32::MAX)).unwrap();
        let gate = scratch.0.join(format!("gate-{round}"));
        let release = scratch.0.join(format!("release-{round}"));
        let mut workers = Vec::new();
        let mut outputs = Vec::new();
        for worker in 0..4 {
            let ready = scratch.0.join(format!("ready-{round}-{worker}"));
            let result = scratch.0.join(format!("result-{round}-{worker}"));
            workers.push(spawn_worker(
                &root, &gate, &release, &ready, &result, "journal",
            ));
            outputs.push((ready, result));
        }
        fs::write(&gate, b"go\n").unwrap();
        wait_until(|| {
            // `fs::write` creates the result path before all bytes have been
            // written. Wait for complete records so this assertion tests
            // lock admission rather than racing the worker's file write.
            outputs
                .iter()
                .filter(|(_, result)| {
                    fs::read_to_string(result).is_ok_and(|value| value == "contended")
                })
                .count()
                == 3
                && outputs.iter().any(|(ready, _)| ready.exists())
        });
        assert_eq!(
            outputs.iter().filter(|(ready, _)| ready.exists()).count(),
            1
        );
        assert_eq!(
            outputs
                .iter()
                .filter(|(_, result)| fs::read_to_string(result).is_ok_and(|v| v == "contended"))
                .count(),
            3
        );
        fs::write(&release, b"release\n").unwrap();
        for mut worker in workers {
            assert!(worker.wait().unwrap().success());
        }
        assert!(!AdvisoryLock::is_held(&root.join(".run.lock")).unwrap());
    }
}

#[test]
fn crashed_owner_releases_kernel_lock_and_restart_recovers_journal() {
    let scratch = Scratch::new();
    let root = scratch.0.join("crashed-lineage");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    drop(RunJournal::open(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap());

    let gate = scratch.0.join("crash-gate");
    let release = scratch.0.join("unused-release");
    let ready = scratch.0.join("crash-ready");
    let result = scratch.0.join("crash-result");
    let mut worker = spawn_worker(&root, &gate, &release, &ready, &result, "journal");
    fs::write(&gate, b"go\n").unwrap();
    wait_until(|| ready.exists());
    assert!(AdvisoryLock::is_held(&root.join(".run.lock")).unwrap());
    worker.kill().unwrap();
    let _ = worker.wait().unwrap();

    assert!(root.join(".run.lock").is_file());
    assert!(!AdvisoryLock::is_held(&root.join(".run.lock")).unwrap());
    let recovered =
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap();
    assert_eq!(recovered.progress().checkpoint_sequence, 0);
    drop(recovered);
}

#[test]
fn process_crash_immediately_after_os_lock_acquisition_is_recoverable() {
    let scratch = Scratch::new();
    let root = scratch.0.join("immediate-crash-lineage");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    drop(RunJournal::open(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap());

    let gate = scratch.0.join("immediate-gate");
    let release = scratch.0.join("immediate-release");
    let ready = scratch.0.join("immediate-ready");
    let result = scratch.0.join("immediate-result");
    let mut worker = spawn_worker(&root, &gate, &release, &ready, &result, "kernel-only");
    fs::write(&gate, b"go\n").unwrap();
    wait_until(|| ready.exists());
    assert!(AdvisoryLock::is_held(&root.join(".run.lock")).unwrap());
    worker.kill().unwrap();
    let _ = worker.wait().unwrap();
    assert!(root.join(".run.lock").is_file());
    assert!(!AdvisoryLock::is_held(&root.join(".run.lock")).unwrap());

    let recovered =
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap();
    drop(recovered);
}

#[test]
fn typed_lock_contention_and_recovery_errors_are_distinct() {
    let scratch = Scratch::new();
    let path = scratch.0.join(".run.lock");
    let owner = AdvisoryLock::acquire(&path).unwrap();
    assert!(matches!(
        AdvisoryLock::acquire(&path),
        Err(LockError::Contended)
    ));
    drop(owner);
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    assert!(matches!(
        AdvisoryLock::acquire(&path),
        Err(LockError::RecoveryFailure(_))
    ));
}
