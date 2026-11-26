use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use xynoos_training::AdvisoryLock;

const DATA_FIXTURE: &str = include_str!("../../xynoos-data/tests/fixtures/task004.json");
const EXE: &str = env!("CARGO_BIN_EXE_xynoos-train");

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "task006r-process-{}-{}",
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

fn source_head() -> String {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn setup(base: &Path, experiment_id: &str) -> (PathBuf, PathBuf) {
    let fixture: Value = serde_json::from_str(DATA_FIXTURE).unwrap();
    let data_root = base.join("dataset");
    fs::create_dir_all(&data_root).unwrap();
    for (name, key) in [
        ("manifest.json", "manifest"),
        ("sources.jsonl", "sources"),
        ("documents.jsonl", "documents"),
    ] {
        fs::write(data_root.join(name), fixture[key].as_str().unwrap()).unwrap();
    }
    for (name, payload) in fixture["payloads"].as_object().unwrap() {
        let path = data_root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, payload.as_str().unwrap()).unwrap();
    }
    let tokenizer = base.join("tokenizer.tsv");
    let artifact = fixture["artifact"].as_str().unwrap();
    fs::write(&tokenizer, artifact).unwrap();
    let fit_record = base.join("tokenizer-fit.jsonl");
    fs::write(&fit_record, fixture["fit"].as_str().unwrap()).unwrap();
    let tokenizer_sha = hex(&Sha256::digest(artifact.as_bytes()));
    let manifest: Value = serde_json::from_str(fixture["manifest"].as_str().unwrap()).unwrap();
    let config = json!({
        "schema_version": 2,
        "experiment_id": experiment_id,
        "repository_id": "samuelindra123/xynoos-cyber-model",
        "source_revision": source_head(),
        "source_kind": "manifest-v1-stream",
        "dataset_root": "dataset",
        "tokenizer_artifact": "tokenizer.tsv",
        "tokenizer_fit_record": "tokenizer-fit.jsonl",
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": fixture["manifest_sha256"],
        "tokenizer_sha256": tokenizer_sha,
        "optimizer": {"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"weight_decay":0.01},
        "schedule": {"total_updates":10,"warmup_updates":2,"peak_rate":0.0001,"final_rate":0.00001},
        "limits": {
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
    let config_path = base.join("experiment.json");
    fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    (config_path, data_root)
}

fn cli(command: &str, config: Option<&Path>, root: &Path, extras: &[&str]) -> Output {
    let mut process = Command::new(EXE);
    process.arg(command);
    if let Some(config) = config {
        process.arg("--config").arg(config);
    }
    process.arg("--artifact-root").arg(root).args(extras);
    process.output().unwrap()
}

fn json_stdout(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "successful command returned invalid JSON: {error}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn wait_for(timeout: Duration, mut predicate: impl FnMut() -> bool) {
    let started = Instant::now();
    while started.elapsed() < timeout {
        if predicate() {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("condition not observed within {timeout:?}");
}

fn pid_alive(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

fn run_lock_held(root: &Path) -> bool {
    AdvisoryLock::is_held(&root.join(".run.lock")).unwrap()
}

fn process_session(pid: u32) -> u32 {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
    let (_, fields) = stat.rsplit_once(") ").unwrap();
    fields.split_whitespace().nth(3).unwrap().parse().unwrap()
}

fn process_start_time(pid: u32) -> u64 {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
    let (_, fields) = stat.rsplit_once(") ").unwrap();
    fields.split_whitespace().nth(19).unwrap().parse().unwrap()
}

fn wait_for_update(root: &Path) {
    wait_for(Duration::from_secs(45), || {
        fs::read_to_string(root.join("metrics.jsonl")).is_ok_and(|log| {
            let update = log
                .lines()
                .any(|line| line.contains("\"type\":\"optimizer_update\""));
            let checkpoint = log
                .lines()
                .any(|line| line.contains("\"type\":\"checkpoint\""));
            update && checkpoint
        })
    });
}

fn process_id(value: &Value) -> u32 {
    value["pid"].as_u64().unwrap().try_into().unwrap()
}

fn write_config(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn build_over_budget_dataset(base: &Path, fixture: &Value) -> (PathBuf, String, String) {
    let root = base.join("over-budget-dataset");
    fs::create_dir_all(root.join("payload")).unwrap();
    for (name, payload) in fixture["payloads"].as_object().unwrap() {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, payload.as_str().unwrap()).unwrap();
    }
    let sources = format!("{}\n", fixture["sources"].as_str().unwrap().trim_end());
    let original = fixture["documents"].as_str().unwrap().trim_end();
    let first = original.lines().next().unwrap();
    let id = serde_json::from_str::<Value>(first).unwrap()["document_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let duplicate = first.replace(
        &format!("\"document_id\":\"{id}\""),
        "\"document_id\":\"duplicate000000\"",
    );
    let mut documents = String::new();
    for line in original.lines() {
        documents.push_str(line);
        documents.push('\n');
    }
    for index in 0..6000 {
        let line = duplicate.replace("duplicate000000", &format!("duplicate{index:06}"));
        documents.push_str(&line);
        documents.push('\n');
    }
    let source_sha = hex(&Sha256::digest(sources.as_bytes()));
    let documents_sha = hex(&Sha256::digest(documents.as_bytes()));
    let mut dataset_identity = Sha256::new();
    dataset_identity.update(b"XYNOOS-DATASET-V1\0");
    dataset_identity.update((sources.len() as u64).to_be_bytes());
    dataset_identity.update(sources.as_bytes());
    dataset_identity.update((documents.len() as u64).to_be_bytes());
    dataset_identity.update(documents.as_bytes());
    let dataset_id = hex(&dataset_identity.finalize());
    let mut manifest = fixture["manifest"].as_str().unwrap().to_owned();
    let original_manifest: Value = serde_json::from_str(&manifest).unwrap();
    manifest = manifest
        .replace(
            original_manifest["dataset_id"].as_str().unwrap(),
            &dataset_id,
        )
        .replace(
            original_manifest["sources_sha256"].as_str().unwrap(),
            &source_sha,
        )
        .replace(
            original_manifest["documents_sha256"].as_str().unwrap(),
            &documents_sha,
        )
        .replace(
            &format!("\"record_count\":{}", original_manifest["record_count"]),
            &format!("\"record_count\":{}", 16 + 6000),
        );
    let manifest_sha = hex(&Sha256::digest(manifest.as_bytes()));
    fs::write(root.join("sources.jsonl"), sources).unwrap();
    fs::write(root.join("documents.jsonl"), documents).unwrap();
    fs::write(root.join("manifest.json"), manifest).unwrap();
    (root, dataset_id, manifest_sha)
}

fn stop_and_wait(config: &Path, root: &Path, pid: u32) {
    assert!(cli("stop", None, root, &[]).status.success());
    wait_for(Duration::from_secs(20), || {
        fs::read(root.join("final-report.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|report| report["stop_reason"] == "requested")
            && !run_lock_held(root)
    });
    wait_for(Duration::from_secs(5), || !pid_alive(pid));
    let report: Value =
        serde_json::from_slice(&fs::read(root.join("final-report.json")).unwrap()).unwrap();
    println!(
        "TASK006R detached-process updates={} owned_peak_bytes={} autograd_peak_bytes={} process_vm_hwm_bytes={}",
        report["counters"]["completed_updates"],
        report["memory"]["charged_owned_peak_bytes"],
        report["memory"]["autograd_ledger_peak_bytes"],
        report["memory"]["process_vm_hwm_bytes"],
    );
    let status = json_stdout(&cli("status", Some(config), root, &[]));
    assert_eq!(status["status"], "finished");
    let checkpoint = json_stdout(&cli("inspect-checkpoint", Some(config), root, &[]));
    assert!(checkpoint["optimizer_step"].as_u64().unwrap() >= 1);
}

#[test]
fn detached_process_survives_controller_exit_supports_stop_status_resume_and_crash_recovery() {
    let scratch = Scratch::new();
    let (config, _) = setup(&scratch.0, "task006r-detached-one");
    let root = scratch.0.join("run-one");
    fs::create_dir(&root).unwrap();
    let validated = json_stdout(&cli("validate", Some(&config), &root, &[]));
    assert_eq!(validated["status"], "valid");
    assert_eq!(validated["parameters"], 914400);

    // `run --detach` returns after the controller process exits. The native
    // runtime child continues independently and owns logs/checkpoints.
    let launched = json_stdout(&cli("run", Some(&config), &root, &["--detach"]));
    let first_pid = process_id(&launched);
    let first_process = ProcessGuard(first_pid);
    assert_ne!(
        process_session(first_pid),
        process_session(std::process::id())
    );
    wait_for_update(&root);
    let concurrent = cli("resume", Some(&config), &root, &["--detach"]);
    assert_eq!(concurrent.status.code(), Some(2));
    let concurrent_error = String::from_utf8_lossy(&concurrent.stderr);
    assert!(
        concurrent_error.contains("owns admission")
            || concurrent_error.contains("owns this experiment lineage"),
        "unexpected concurrent resume error: {concurrent_error}"
    );
    assert!(
        pid_alive(first_pid),
        "training process died with its launching controller"
    );
    assert!(root.join("checkpoint.xyckpt").is_file());
    assert!(root.join("metrics.jsonl").metadata().unwrap().len() > 0);
    stop_and_wait(&config, &root, first_pid);
    drop(first_process);

    // Resume must use the existing immutable experiment ID and journal.
    let previous_sequence = json_stdout(&cli("status", Some(&config), &root, &[]))["final_report"]
        ["counters"]["checkpoint_sequence"]
        .as_u64()
        .unwrap();
    let resumed = json_stdout(&cli("resume", Some(&config), &root, &["--detach"]));
    assert_eq!(resumed["experiment_id"], "task006r-detached-one");
    let resumed_pid = process_id(&resumed);
    let resumed_process = ProcessGuard(resumed_pid);
    wait_for(Duration::from_secs(45), || {
        fs::read(root.join("final-report.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|report| {
                report["stop_reason"] == "requested"
                    && report["counters"]["checkpoint_sequence"]
                        .as_u64()
                        .unwrap_or(0)
                        > previous_sequence
            })
            && !run_lock_held(&root)
    });
    wait_for(Duration::from_secs(5), || !pid_alive(resumed_pid));
    drop(resumed_process);

    // A second lineage is killed after a committed checkpoint. The persisted
    // runtime reservation remains and resume charges it before admitting work.
    let (crash_config, _) = setup(&scratch.0, "task006r-abrupt-two");
    let crash_root = scratch.0.join("run-two");
    fs::create_dir(&crash_root).unwrap();
    let launched = json_stdout(&cli("run", Some(&crash_config), &crash_root, &["--detach"]));
    let crash_pid = process_id(&launched);
    let crash_guard = ProcessGuard(crash_pid);
    wait_for_update(&crash_root);
    let before = json_stdout(&cli("status", Some(&crash_config), &crash_root, &[]));
    assert!(before["journal"]["reserved_runtime_ns"].as_u64().unwrap() > 0);
    let killed = Command::new("kill")
        .args(["-KILL", &crash_pid.to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    wait_for(Duration::from_secs(5), || !pid_alive(crash_pid));
    assert!(crash_root.join("checkpoint.xyckpt").is_file());
    assert!(crash_root.join("run.journal").is_file());
    assert!(
        crash_root.join(".run.lock").exists(),
        "abrupt exit keeps the persistent lock inode for kernel-backed recovery"
    );
    assert!(!run_lock_held(&crash_root));
    let interrupted = json_stdout(&cli("status", Some(&crash_config), &crash_root, &[]));
    assert_eq!(interrupted["status"], "interrupted");
    assert!(
        interrupted["journal"]["reserved_runtime_ns"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(cli("stop", None, &crash_root, &[]).status.success());

    // Reuse a currently live PID with a deliberately different process start
    // time. Recovery must distinguish the process incarnation, then launch.
    let reused_pid = std::process::id();
    let start_time = process_start_time(reused_pid);
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .unwrap()
        .trim()
        .to_owned();
    fs::write(
        crash_root.join("process.pid"),
        format!(
            "XYNOOS-PROCESS-1\t{reused_pid}\t{}\t{boot_id}\n",
            start_time + 1
        ),
    )
    .unwrap();

    let resumed = json_stdout(&cli(
        "resume",
        Some(&crash_config),
        &crash_root,
        &["--detach"],
    ));
    let recovered_pid = process_id(&resumed);
    let recovered_guard = ProcessGuard(recovered_pid);
    wait_for(Duration::from_secs(45), || {
        fs::read(crash_root.join("final-report.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|report| report["stop_reason"] == "requested")
            && !run_lock_held(&crash_root)
    });
    wait_for(Duration::from_secs(5), || !pid_alive(recovered_pid));
    let after = json_stdout(&cli("status", Some(&crash_config), &crash_root, &[]));
    assert!(
        after["journal"]["active_runtime_ns"].as_u64().unwrap()
            >= before["journal"]["active_runtime_ns"].as_u64().unwrap()
    );
    assert!(
        after["journal"]["cumulative_work"].as_u64().unwrap()
            >= before["journal"]["cumulative_work"].as_u64().unwrap()
    );

    // A failed recovery publishes a separate failure report and preserves the
    // last committed checkpoint byte-for-byte.
    let failed_root = scratch.0.join("failed-recovery");
    fs::create_dir(&failed_root).unwrap();
    for name in ["experiment.json", "checkpoint.xyckpt"] {
        fs::copy(crash_root.join(name), failed_root.join(name)).unwrap();
    }
    fs::write(failed_root.join("run.journal"), b"corrupted lineage").unwrap();
    let checkpoint_before =
        Sha256::digest(fs::read(failed_root.join("checkpoint.xyckpt")).unwrap());
    let failed = cli("resume", Some(&crash_config), &failed_root, &[]);
    assert_eq!(failed.status.code(), Some(2));
    let failure: Value =
        serde_json::from_slice(&fs::read(failed_root.join("failure-report.json")).unwrap())
            .unwrap();
    assert_eq!(failure["status"], "failed");
    assert_eq!(failure["phase"], "trainer_initialization");
    assert_eq!(
        Sha256::digest(fs::read(failed_root.join("checkpoint.xyckpt")).unwrap()),
        checkpoint_before
    );
    drop((crash_guard, recovered_guard));
}

#[test]
fn concurrent_detached_launch_attempts_start_at_most_one_lineage_owner() {
    let scratch = Scratch::new();
    let (config, _) = setup(&scratch.0, "task006r-concurrent-launch");
    let root = scratch.0.join("concurrent-launch-root");
    fs::create_dir(&root).unwrap();
    let _owner_cleanup = ArtifactProcessGuard(root.clone());

    let mut launchers = Vec::new();
    for _ in 0..4 {
        launchers.push(
            Command::new(EXE)
                .arg("run")
                .arg("--config")
                .arg(&config)
                .arg("--artifact-root")
                .arg(&root)
                .arg("--detach")
                .spawn()
                .unwrap(),
        );
    }
    let mut launched = Vec::new();
    let mut rejected = 0;
    for launcher in launchers {
        let output = launcher.wait_with_output().unwrap();
        if output.status.success() {
            launched.push(json_stdout(&output));
        } else {
            rejected += 1;
            assert_eq!(output.status.code(), Some(2));
        }
    }
    assert_eq!(launched.len(), 1);
    assert_eq!(rejected, 3);
    let owner_pid = process_id(&launched[0]);
    let owner_guard = ProcessGuard(owner_pid);
    assert!(run_lock_held(&root));
    assert!(cli("stop", None, &root, &[]).status.success());
    wait_for(Duration::from_secs(30), || !run_lock_held(&root));
    wait_for(Duration::from_secs(5), || !pid_alive(owner_pid));
    drop(owner_guard);
}

#[test]
fn validate_rejects_untrusted_source_configuration_and_over_budget_retained_metadata() {
    let scratch = Scratch::new();
    let (config_path, _) = setup(&scratch.0, "task006r-admission");
    let root = scratch.0.join("validation-root");
    fs::create_dir(&root).unwrap();
    let pristine: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();

    let mut changed = pristine.clone();
    changed["unrecognized"] = json!(true);
    write_config(&config_path, &changed);
    let output = cli("validate", Some(&config_path), &root, &[]);
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown configuration field"));

    let mut changed = pristine.clone();
    changed["source_kind"] = json!("custom-batch-source");
    write_config(&config_path, &changed);
    let output = cli("validate", Some(&config_path), &root, &[]);
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));

    let mut changed = pristine.clone();
    changed["dataset_id"] = json!("00".repeat(32));
    write_config(&config_path, &changed);
    assert!(
        !cli("validate", Some(&config_path), &root, &[])
            .status
            .success()
    );

    let mut changed = pristine.clone();
    changed["tokenizer_sha256"] = json!("ff".repeat(32));
    write_config(&config_path, &changed);
    assert!(
        !cli("validate", Some(&config_path), &root, &[])
            .status
            .success()
    );

    let fixture: Value = serde_json::from_str(DATA_FIXTURE).unwrap();
    let (oversized, dataset_id, manifest_sha) = build_over_budget_dataset(&scratch.0, &fixture);
    let mut changed = pristine;
    changed["dataset_root"] = json!(oversized.to_string_lossy());
    changed["dataset_id"] = json!(dataset_id);
    changed["dataset_manifest_sha256"] = json!(manifest_sha);
    write_config(&config_path, &changed);
    let output = cli("validate", Some(&config_path), &root, &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("dataset admission"));
    assert!(!root.join("checkpoint.xyckpt").exists());
}

struct ProcessGuard(u32);

struct ArtifactProcessGuard(PathBuf);

impl Drop for ArtifactProcessGuard {
    fn drop(&mut self) {
        let Ok(identity) = fs::read_to_string(self.0.join("process.pid")) else {
            return;
        };
        let fields: Vec<_> = identity.split_whitespace().collect();
        if fields.len() != 4 || fields[0] != "XYNOOS-PROCESS-1" {
            return;
        }
        let Ok(pid) = fields[1].parse::<u32>() else {
            return;
        };
        let Ok(start_time) = fields[2].parse::<u64>() else {
            return;
        };
        let Some(stat) = fs::read_to_string(format!("/proc/{pid}/stat")).ok() else {
            return;
        };
        let actual_start = stat
            .rsplit_once(") ")
            .and_then(|(_, values)| values.split_whitespace().nth(19))
            .and_then(|value| value.parse::<u64>().ok());
        if actual_start == Some(start_time) {
            let _ = Command::new("kill")
                .args(["-KILL", &pid.to_string()])
                .status();
            let started = Instant::now();
            while started.elapsed() < Duration::from_secs(2) && pid_alive(pid) {
                thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if pid_alive(self.0) {
            let _ = Command::new("kill")
                .args(["-KILL", &self.0.to_string()])
                .status();
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(2) && pid_alive(self.0) {
                thread::sleep(Duration::from_millis(20));
            }
        }
    }
}
