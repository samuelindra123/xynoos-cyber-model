use std::{
    fs,
    os::unix::{fs::PermissionsExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use xynoos_training::{AdvisoryLock, CheckpointCounters, RunJournal, RunLimits, TrainingError};

const DIGEST: [u8; 32] = [0xF6; 32];
const WORKER_ROOT: &str = "XYNOOS_TASK006F_WORKER_ROOT";
const WORKER_READY: &str = "XYNOOS_TASK006F_WORKER_READY";

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "task006f-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
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
        100,
        100,
        8_000_000_000,
        64 * 1024 * 1024,
        Duration::from_secs(20),
        Duration::from_secs(5),
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
    panic!("auditor process did not reach its ready state within ten seconds");
}

fn private_directory(path: &Path) {
    fs::create_dir(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

#[test]
#[ignore = "subprocess entrypoint for the SIGKILL recovery control"]
fn sigkill_owner_worker() {
    let Ok(root) = std::env::var(WORKER_ROOT) else {
        return;
    };
    let ready = PathBuf::from(std::env::var(WORKER_READY).unwrap());
    let mut journal = RunJournal::open_existing(
        Path::new(&root),
        DIGEST,
        CheckpointCounters::default(),
        limits(),
    )
    .unwrap();
    journal.begin_attempt().unwrap();
    journal.precharge_microbatch(7, 6, 1_234).unwrap();
    fs::write(ready, b"owner holds the journal lock\n").unwrap();
    loop {
        thread::sleep(Duration::from_secs(60));
    }
}

#[test]
fn sigkill_releases_kernel_lock_without_replacing_lineage_inode_or_refunding_precharge() {
    use std::os::unix::fs::MetadataExt;

    let scratch = Scratch::new("sigkill");
    let root = scratch.0.join("lineage");
    private_directory(&root);
    drop(RunJournal::open(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap());
    let lock_path = root.join(".run.lock");
    let inode = fs::metadata(&lock_path).unwrap().ino();
    let ready = scratch.0.join("owner-ready");
    let mut owner = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "sigkill_owner_worker",
            "--nocapture",
        ])
        .env(WORKER_ROOT, &root)
        .env(WORKER_READY, &ready)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    wait_until(|| ready.exists());
    assert!(AdvisoryLock::is_held(&lock_path).unwrap());
    assert!(matches!(
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits()),
        Err(TrainingError::RunLockContended)
    ));

    owner.kill().unwrap();
    let status = owner.wait().unwrap();
    assert_eq!(status.signal(), Some(9), "owner must terminate by SIGKILL");
    assert!(!AdvisoryLock::is_held(&lock_path).unwrap());
    assert_eq!(fs::metadata(&lock_path).unwrap().ino(), inode);

    let recovered =
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap();
    let progress = recovered.progress();
    assert_eq!(progress.attempted_consumed_tokens, 7);
    assert_eq!(progress.attempted_supervised_tokens, 6);
    assert_eq!(progress.cumulative_work, 1_234);
    assert_eq!(progress.reserved_runtime_ns, 0);
    assert!(progress.active_runtime_ns >= Duration::from_secs(5).as_nanos() as u64);
    assert_eq!(fs::metadata(&lock_path).unwrap().ino(), inode);
}

#[test]
fn corrupt_and_missing_journals_fail_closed_without_rewriting_or_recreating_state() {
    let scratch = Scratch::new("journal-integrity");
    let root = scratch.0.join("lineage");
    private_directory(&root);
    drop(RunJournal::open(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap());

    let journal_path = root.join("run.journal");
    let mut corrupted = fs::read(&journal_path).unwrap();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x80;
    fs::write(&journal_path, &corrupted).unwrap();
    assert!(
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits(),).is_err()
    );
    assert_eq!(fs::read(&journal_path).unwrap(), corrupted);

    fs::remove_file(&journal_path).unwrap();
    assert!(
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits(),).is_err()
    );
    assert!(!journal_path.exists());
}
