use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use xynoos_training::{
    AdvisoryLock, CheckpointCounters, RunJournal, RunLimits, TrainingError, attest_source_checkout,
};

const WORKER_ROOT: &str = "XYNOOS_C3_WORKER_ROOT";
const WORKER_GATE: &str = "XYNOOS_C3_WORKER_GATE";
const WORKER_RELEASE: &str = "XYNOOS_C3_WORKER_RELEASE";
const WORKER_RESULT: &str = "XYNOOS_C3_WORKER_RESULT";
const WORKER_READY: &str = "XYNOOS_C3_WORKER_READY";
const DIGEST: [u8; 32] = [0xA3; 32];
const DATA_FIXTURE: &str = include_str!("../../xynoos-data/tests/fixtures/task004.json");
const TRAIN_EXE: &str = env!("CARGO_BIN_EXE_xynoos-train");

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "task006c3-{label}-{}-{}",
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
    while started.elapsed() < Duration::from_secs(15) {
        if predicate() {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
    panic!("independent audit condition was not observed within 15 seconds");
}

fn spawn_journal_worker(
    root: &Path,
    gate: &Path,
    release: &Path,
    ready: &Path,
    result: &Path,
) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "independent_journal_worker", "--nocapture"])
        .env(WORKER_ROOT, root)
        .env(WORKER_GATE, gate)
        .env(WORKER_RELEASE, release)
        .env(WORKER_READY, ready)
        .env(WORKER_RESULT, result)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}

#[test]
fn independent_journal_worker() {
    let Ok(root) = std::env::var(WORKER_ROOT) else {
        return;
    };
    let gate = PathBuf::from(std::env::var(WORKER_GATE).unwrap());
    let release = PathBuf::from(std::env::var(WORKER_RELEASE).unwrap());
    let ready = PathBuf::from(std::env::var(WORKER_READY).unwrap());
    let result = PathBuf::from(std::env::var(WORKER_RESULT).unwrap());
    wait_until(|| gate.exists());
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
        Err(error) => fs::write(&result, format!("error:{error:?}")).unwrap(),
    }
}

#[test]
fn independent_process_races_never_obtain_two_journal_owners() {
    let scratch = Scratch::new("ownership-race");
    let root = scratch.0.join("lineage");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    drop(RunJournal::open(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap());

    for round in 0..12 {
        // A legacy-looking dead PID is deliberately left in the persistent
        // lock inode. The test does not grant stale recovery by unlinking it.
        fs::write(root.join(".run.lock"), "2147483647\n").unwrap();
        let gate = scratch.0.join(format!("gate-{round}"));
        let release = scratch.0.join(format!("release-{round}"));
        let mut workers = Vec::new();
        let mut evidence = Vec::new();
        for worker in 0..6 {
            let ready = scratch.0.join(format!("ready-{round}-{worker}"));
            let result = scratch.0.join(format!("result-{round}-{worker}"));
            workers.push(spawn_journal_worker(
                &root, &gate, &release, &ready, &result,
            ));
            evidence.push((ready, result));
        }
        fs::write(&gate, b"release contenders\n").unwrap();
        wait_until(|| {
            evidence
                .iter()
                .filter(|(_, result)| result.exists())
                .count()
                == 5
                && evidence.iter().any(|(ready, _)| ready.exists())
        });
        assert_eq!(
            evidence.iter().filter(|(ready, _)| ready.exists()).count(),
            1,
            "round {round} admitted more than one process"
        );
        assert_eq!(
            evidence
                .iter()
                .filter(|(_, result)| fs::read_to_string(result)
                    .is_ok_and(|value| value == "contended"))
                .count(),
            5,
            "round {round} did not reject every non-owner"
        );
        fs::write(&release, b"release owner\n").unwrap();
        for mut worker in workers {
            assert!(worker.wait().unwrap().success());
        }
        assert_eq!(
            evidence
                .iter()
                .filter(|(_, result)| fs::read_to_string(result).is_ok_and(|value| value == "owner"))
                .count(),
            1
        );
        assert!(!AdvisoryLock::is_held(&root.join(".run.lock")).unwrap());
    }
}

#[test]
fn independent_kernel_lock_blocks_journal_open_and_releases_after_process_exit() {
    let scratch = Scratch::new("external-flock");
    let root = scratch.0.join("lineage");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    drop(RunJournal::open(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap());
    let ready = scratch.0.join("python-owner-ready");
    let release = scratch.0.join("python-owner-release");
    let script = r#"
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
open(sys.argv[2], 'wb').close()
while not os.path.exists(sys.argv[3]):
    time.sleep(0.002)
"#;
    let mut owner = Command::new("python3")
        .args([
            "-c",
            script,
            root.join(".run.lock").to_str().unwrap(),
            ready.to_str().unwrap(),
            release.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_until(|| ready.exists());
    assert_eq!(
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits())
            .unwrap_err(),
        TrainingError::RunLockContended
    );
    fs::write(&release, b"release\n").unwrap();
    assert!(owner.wait().unwrap().success());
    drop(
        RunJournal::open_existing(&root, DIGEST, CheckpointCounters::default(), limits()).unwrap(),
    );
}

#[test]
fn independent_source_attestation_rejects_wrong_revision_dirty_and_ignored_sources() {
    let scratch = Scratch::new("source");
    let root = scratch.0.join("checkout");
    let build_root = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .unwrap();
    assert!(build_root.status.success());
    let build_root = String::from_utf8(build_root.stdout)
        .unwrap()
        .trim()
        .to_owned();
    let clone = Command::new("git")
        .args(["clone", "--quiet", "--no-hardlinks"])
        .arg(&build_root)
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        clone.status.success(),
        "git clone: {}",
        String::from_utf8_lossy(&clone.stderr)
    );
    let git = |arguments: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(arguments)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {arguments:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    };
    git(&[
        "remote",
        "set-url",
        "origin",
        "https://github.com/samuelindra123/xynoos-cyber-model.git",
    ]);
    let revision = git(&["rev-parse", "HEAD"]);
    let canonical = xynoos_training::SOURCE_REPOSITORY_ID;
    assert_eq!(revision, env!("XYNOOS_BUILD_SOURCE_REVISION"));
    let identity = attest_source_checkout(&root, canonical, &revision).unwrap();
    assert_eq!(identity.repository_id(), canonical);
    assert_eq!(identity.revision(), revision);
    assert!(attest_source_checkout(&root, canonical, &"0".repeat(40)).is_err());
    assert!(attest_source_checkout(&root, "other/project", &revision).is_err());

    let cargo_manifest = fs::read(root.join("Cargo.toml")).unwrap();
    fs::write(root.join("Cargo.toml"), b"[workspace]\n# modified\n").unwrap();
    assert!(
        attest_source_checkout(&root, canonical, &revision)
            .unwrap_err()
            .contains("dirty")
    );
    fs::write(root.join("Cargo.toml"), cargo_manifest).unwrap();
    let exclude = root.join(".git/info/exclude");
    let mut excluded = fs::read_to_string(&exclude).unwrap_or_default();
    excluded.push_str("\n/crates/xynoos-training/src/ignored.rs\n");
    fs::write(exclude, excluded).unwrap();
    fs::write(
        root.join("crates/xynoos-training/src/ignored.rs"),
        "// ignored implementation\n",
    )
    .unwrap();
    assert!(
        attest_source_checkout(&root, canonical, &revision)
            .unwrap_err()
            .contains("ignored untracked")
    );
}

#[test]
fn independent_sidecar_interruption_leaves_only_recoverable_staging() {
    let scratch = Scratch::new("sidecar-recovery");
    let fixture: Value = serde_json::from_str(DATA_FIXTURE).unwrap();
    let dataset = scratch.0.join("dataset");
    fs::create_dir_all(dataset.join("payload")).unwrap();
    for (name, content) in [
        ("manifest.json", fixture["manifest"].as_str().unwrap()),
        ("sources.jsonl", fixture["sources"].as_str().unwrap()),
        ("documents.jsonl", fixture["documents"].as_str().unwrap()),
    ] {
        fs::write(dataset.join(name), content).unwrap();
    }
    for (name, content) in fixture["payloads"].as_object().unwrap() {
        let path = dataset.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content.as_str().unwrap()).unwrap();
    }
    let tokenizer = scratch.0.join("tokenizer.tsv");
    let tokenizer_bytes = fixture["artifact"].as_str().unwrap().as_bytes();
    fs::write(&tokenizer, tokenizer_bytes).unwrap();
    let fit = scratch.0.join("tokenizer-fit.jsonl");
    fs::write(&fit, fixture["fit"].as_str().unwrap()).unwrap();
    let manifest: Value = serde_json::from_str(fixture["manifest"].as_str().unwrap()).unwrap();
    let source_revision = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(source_revision.status.success());
    let source_revision = String::from_utf8(source_revision.stdout)
        .unwrap()
        .trim()
        .to_owned();
    let config_path = scratch.0.join("experiment.json");
    let config = json!({
        "schema_version":2,
        "experiment_id":"task006c3-sidecar-recovery",
        "repository_id":"samuelindra123/xynoos-cyber-model",
        "source_revision":source_revision,
        "source_kind":"manifest-v1-stream",
        "dataset_root":"dataset",
        "tokenizer_artifact":"tokenizer.tsv",
        "tokenizer_fit_record":"tokenizer-fit.jsonl",
        "dataset_id":manifest["dataset_id"],
        "dataset_manifest_sha256":fixture["manifest_sha256"],
        "tokenizer_sha256":hex(&Sha256::digest(tokenizer_bytes)),
        "optimizer":{"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"weight_decay":0.01},
        "schedule":{"total_updates":10,"warmup_updates":2,"peak_rate":0.0001,"final_rate":0.00001},
        "limits":{
            "microbatches_per_update":1,
            "max_updates":10,
            "consumed_tokens":320,
            "supervised_tokens":320,
            "cumulative_work":8000000000_u64,
            "owned_memory_bytes":67108864,
            "max_runtime_seconds":30,
            "max_attempt_runtime_seconds":15,
            "checkpoint_every_updates":1,
            "validation_every_updates":1
        },
        "memory":{"data_allowance_bytes":4194304},
        "autograd":{"bytes":67108864,"values":100000,"operations":100000,"work":100000000},
        "gradient_clip":1.0,
        "seed":42
    });
    fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    let root = scratch.0.join("artifact-root");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

    let python = r#"
import resource, signal, subprocess, sys
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
result = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.exit(0 if result.returncode == -signal.SIGXFSZ else 1)
"#;
    let interrupted = Command::new("python3")
        .args([
            "-c",
            python,
            TRAIN_EXE,
            "run",
            "--config",
            config_path.to_str().unwrap(),
            "--artifact-root",
            root.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        interrupted.status.success(),
        "independent file-size interruption was not observed: {}",
        String::from_utf8_lossy(&interrupted.stderr)
    );
    assert!(!root.join("experiment.json").exists());
    let staging: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name.to_string_lossy().starts_with(".experiment.json-init-"))
        .collect();
    assert_eq!(staging.len(), 1, "expected the interrupted staging file");
    assert_eq!(fs::metadata(root.join(&staging[0])).unwrap().len(), 0);

    let recovered = Command::new(TRAIN_EXE)
        .args([
            "run",
            "--config",
            config_path.to_str().unwrap(),
            "--artifact-root",
            root.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        recovered.status.success(),
        "restart did not recover initial sidecar state: stdout={} stderr={}",
        String::from_utf8_lossy(&recovered.stdout),
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert!(root.join("experiment.json").is_file());
    assert!(root.join("final-report.json").is_file());
    assert!(fs::read_dir(&root).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".experiment.json-init-")
    }));
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}
