#![forbid(unsafe_code)]

use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use xynoos_autograd::{AdamWConfig, ResourceLimits};
use xynoos_data::{
    BatchConfig, Dataset, DatasetLimits, FittedTokenizer, ManifestValidator, Split, StreamConfig,
    Tail,
};
use xynoos_tokenizer::{Boundaries, TokenizerLimits};
use xynoos_training::{
    AdvisoryLock, CheckpointCodec, EvaluationLimits, LinearWarmupDecay, LockError, RunIdentity,
    RunJournal, RunLimits, StopToken, Trainer, TrainingConfig, TrainingMemoryPlan,
    attest_source_checkout,
};

const CONFIG_MAX: usize = 16 * 1024;
const FIT_RECORD_MAX: usize = 16 * 1024;
const SIDECAR: &str = "experiment.json";
const PROCESS_PID: &str = "process.pid";
const MODEL_T: usize = 32;

type CliResult<T> = Result<T, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandKind {
    Validate,
    Run,
    Resume,
    Status,
    Stop,
    InspectCheckpoint,
}

struct Args {
    command: CommandKind,
    config: Option<PathBuf>,
    root: PathBuf,
    detached: bool,
}

struct Prepared {
    experiment_id: String,
    source_repository_id: String,
    source_revision: String,
    dataset_id: String,
    config_sha256: [u8; 32],
    identity: RunIdentity,
    config: TrainingConfig,
    dataset: Dataset,
    tokenizer: FittedTokenizer,
    stream: StreamConfig,
    dataset_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessIdentity {
    pid: u32,
    start_time_ticks: u64,
    boot_id: String,
}

impl ProcessIdentity {
    fn current() -> CliResult<Self> {
        let pid = std::process::id();
        Ok(Self {
            pid,
            start_time_ticks: process_start_time(pid)?,
            boot_id: read_boot_id()?,
        })
    }

    fn encode(&self) -> String {
        format!(
            "XYNOOS-PROCESS-1\t{}\t{}\t{}\n",
            self.pid, self.start_time_ticks, self.boot_id
        )
    }

    fn parse(value: &str) -> CliResult<Self> {
        let fields: Vec<_> = value.split_whitespace().collect();
        if fields.len() != 4 || fields[0] != "XYNOOS-PROCESS-1" {
            return Err("invalid process identity record".into());
        }
        let pid = fields[1]
            .parse::<u32>()
            .map_err(|_| "invalid process identity PID")?;
        let start_time_ticks = fields[2]
            .parse::<u64>()
            .map_err(|_| "invalid process start-time identity")?;
        let boot_id = fields[3].to_owned();
        if pid == 0 || !is_valid_boot_id(&boot_id) {
            return Err("invalid process identity fields".into());
        }
        Ok(Self {
            pid,
            start_time_ticks,
            boot_id,
        })
    }

    fn is_same_process_alive(&self) -> CliResult<bool> {
        if read_boot_id()? != self.boot_id {
            return Ok(false);
        }
        let state = match process_state(self.pid) {
            Ok(state) => state,
            Err(error) if error == "process not found" => return Ok(false),
            Err(error) => return Err(error),
        };
        if matches!(state.as_str(), "Z" | "X") {
            return Ok(false);
        }
        match process_start_time(self.pid) {
            Ok(start_time) => Ok(start_time == self.start_time_ticks),
            Err(error) if error == "process not found" => Ok(false),
            Err(error) => Err(error),
        }
    }
}

struct ProcessIdentityGuard {
    root: PathBuf,
    identity: ProcessIdentity,
    active: bool,
}

impl ProcessIdentityGuard {
    fn register(root: &Path) -> CliResult<Self> {
        if !live_run_lock(root)? {
            return Err("process identity cannot be registered without journal ownership".into());
        }
        let identity = ProcessIdentity::current()?;
        if let Some(previous) = read_process_identity(root)?
            && previous != identity
            && previous.is_same_process_alive()?
        {
            return Err("another live process identity remains registered".into());
        }
        write_process_identity(root, &identity)?;
        Ok(Self {
            root: root.to_path_buf(),
            identity,
            active: true,
        })
    }

    fn remove(mut self) -> CliResult<()> {
        self.remove_inner()?;
        self.active = false;
        Ok(())
    }

    fn remove_inner(&self) -> CliResult<()> {
        let Some(current) = read_process_identity(&self.root)? else {
            return Ok(());
        };
        if current != self.identity {
            return Err("process identity changed before owner cleanup".into());
        }
        fs::remove_file(self.root.join(PROCESS_PID))
            .map_err(|error| format!("remove process identity: {error}"))?;
        sync_directory(&self.root, "process identity cleanup")
    }
}

impl Drop for ProcessIdentityGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = self.remove_inner();
        }
    }
}

struct LaunchLock {
    _lock: AdvisoryLock,
}

fn main() {
    let values: Vec<String> = env::args().skip(1).collect();
    if values.iter().any(|item| item == "--help" || item == "-h") {
        print_help();
        return;
    }
    let parsed = parse_args(values);
    let code = match parsed {
        Ok(args) => dispatch(args).err().unwrap_or(0),
        Err(error) => {
            eprintln!("xynoos-train: {error}");
            2
        }
    };
    if code != 0 {
        std::process::exit(code);
    }
}

fn dispatch(args: Args) -> Result<(), i32> {
    if let Err(error) = validate_root(&args.root) {
        return Err(report(error));
    }
    let detached_child = env::var_os("XYNOOS_TRAIN_DETACHED_CHILD").is_some();
    if matches!(args.command, CommandKind::Run | CommandKind::Resume) {
        ensure_artifact_root_outside_checkout(&args.root).map_err(report)?;
    }
    let _launch_lock =
        if matches!(args.command, CommandKind::Run | CommandKind::Resume) && !detached_child {
            Some(acquire_launch_lock(&args.root).map_err(report)?)
        } else {
            None
        };
    match args.command {
        CommandKind::Stop => {
            if let Err(error) = read_sidecar(&args.root) {
                return Err(report(error));
            }
            StopToken::write_request(&args.root).map_err(|error| report(format!("{error:?}")))?;
            println!(
                "{}",
                json!({"status":"stop_requested","artifact_root":args.root})
            );
            Ok(())
        }
        _ => {
            let config_path = args
                .config
                .as_deref()
                .ok_or_else(|| report("--config is required for this command".to_owned()))?;
            let mut prepared = prepare(config_path).map_err(report)?;
            check_root_state(&args.root, &prepared, args.command).map_err(report)?;
            if matches!(args.command, CommandKind::Run | CommandKind::Resume)
                && live_run_lock(&args.root).map_err(report)?
            {
                return Err(report("another process owns this experiment lineage"));
            }
            match args.command {
                CommandKind::Validate => {
                    let result = validation_json(&prepared, &args.root);
                    println!("{result}");
                    Ok(())
                }
                CommandKind::Run | CommandKind::Resume => {
                    if args.detached && !detached_child {
                        let experiment_id = prepared.experiment_id.clone();
                        drop(prepared);
                        return detach(&args, config_path, &experiment_id);
                    }
                    let initial_recovery =
                        initial_sidecar_recovery(&args.root, &prepared).map_err(report)?;
                    if args.command == CommandKind::Run && !initial_recovery {
                        write_sidecar(&args.root, &prepared).map_err(report)?;
                    } else {
                        verify_sidecar(&args.root, &prepared).map_err(report)?;
                    }
                    execute(
                        &args.root,
                        &mut prepared,
                        args.command == CommandKind::Resume && !initial_recovery,
                    )
                    .map_err(report)
                }
                CommandKind::Status => {
                    verify_sidecar(&args.root, &prepared).map_err(report)?;
                    let progress = RunJournal::inspect(
                        &args.root,
                        prepared.config.digest(&prepared.identity),
                        prepared.config.limits(),
                    )
                    .map_err(|error| report(format!("{error:?}")))?;
                    let final_report = read_optional_json(&args.root.join("final-report.json"))?;
                    let failure_report =
                        read_optional_json(&args.root.join("failure-report.json"))?;
                    let failed_final = final_report.as_ref().is_some_and(|report| {
                        report.get("error").is_some()
                            || matches!(
                                report.get("stop_reason").and_then(Value::as_str),
                                Some("failure" | "numerical_failure")
                            )
                    });
                    println!(
                        "{}",
                        json!({
                            "status": if live_run_lock(&args.root).map_err(report)? { "active" } else if failure_report.is_some() || failed_final { "failed" } else if final_report.is_some() { "finished" } else { "interrupted" },
                            "experiment_id": prepared.experiment_id,
                            "run_config_sha256": hex(&prepared.config.digest(&prepared.identity)),
                            "journal": {
                                "attempted_consumed_tokens": progress.attempted_consumed_tokens,
                                "attempted_supervised_tokens": progress.attempted_supervised_tokens,
                                "cumulative_work": progress.cumulative_work,
                                "active_runtime_ns": progress.active_runtime_ns,
                                "reserved_runtime_ns": progress.reserved_runtime_ns,
                                "attempted_updates": progress.attempted_updates,
                                "checkpoint_sequence": progress.checkpoint_sequence,
                            },
                            "final_report": final_report,
                            "failure_report": failure_report,
                        })
                    );
                    Ok(())
                }
                CommandKind::InspectCheckpoint => {
                    verify_sidecar(&args.root, &prepared).map_err(report)?;
                    let data = CheckpointCodec::load(
                        &args.root,
                        &prepared.identity,
                        prepared.config,
                        prepared.tokenizer.model(),
                    )
                    .map_err(|error| report(format!("{error:?}")))?;
                    println!(
                        "{}",
                        json!({
                            "experiment_id": prepared.experiment_id,
                            "sha256": hex(&data.digest),
                            "configuration_sha256": hex(&data.configuration_digest),
                            "cursor": data.cursor,
                            "optimizer_step": data.optimizer_step,
                            "beta1_power_bits": format!("{:016x}", data.beta1_power.to_bits()),
                            "beta2_power_bits": format!("{:016x}", data.beta2_power.to_bits()),
                            "counters": {
                                "committed_consumed_tokens": data.counters.committed_consumed_tokens,
                                "committed_supervised_tokens": data.counters.committed_supervised_tokens,
                                "completed_updates": data.counters.completed_updates,
                                "checkpoint_sequence": data.counters.checkpoint_sequence,
                            }
                        })
                    );
                    Ok(())
                }
                CommandKind::Stop => unreachable!(),
            }
        }
    }
}

fn parse_args(args: Vec<String>) -> CliResult<Args> {
    if args.is_empty() || args.iter().any(|item| item == "--help" || item == "-h") {
        print_help();
        return Err("help".into());
    }
    let command = match args[0].as_str() {
        "validate" => CommandKind::Validate,
        "run" => CommandKind::Run,
        "resume" => CommandKind::Resume,
        "status" => CommandKind::Status,
        "stop" => CommandKind::Stop,
        "inspect-checkpoint" => CommandKind::InspectCheckpoint,
        _ => return Err("unknown command".into()),
    };
    let mut config = None;
    let mut root = None;
    let mut detached = false;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--config" if config.is_none() => {
                index += 1;
                config = Some(PathBuf::from(
                    args.get(index).ok_or("missing --config value")?,
                ));
            }
            "--artifact-root" if root.is_none() => {
                index += 1;
                root = Some(PathBuf::from(
                    args.get(index).ok_or("missing --artifact-root value")?,
                ));
            }
            "--detach"
                if !detached && matches!(command, CommandKind::Run | CommandKind::Resume) =>
            {
                detached = true;
            }
            _ => return Err(format!("unknown or duplicate argument: {}", args[index])),
        }
        index += 1;
    }
    let root = root.ok_or("--artifact-root is required")?;
    if !root.is_absolute()
        || root
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err("--artifact-root must be an absolute path without parent traversal".into());
    }
    if command != CommandKind::Stop && config.is_none() {
        return Err("--config is required".into());
    }
    if command == CommandKind::Stop && config.is_some() {
        return Err("stop accepts only --artifact-root".into());
    }
    Ok(Args {
        command,
        config,
        root,
        detached,
    })
}

fn prepare(config_path: &Path) -> CliResult<Prepared> {
    let config_path =
        canonicalize_non_symlink(config_path).map_err(|e| format!("config path: {e}"))?;
    let metadata =
        fs::symlink_metadata(&config_path).map_err(|e| format!("config metadata: {e}"))?;
    let config_file_bytes = usize::try_from(metadata.len())
        .map_err(|_| "configuration size does not fit this platform")?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || config_file_bytes > CONFIG_MAX {
        return Err("configuration must be a regular file no larger than 16 KiB".into());
    }
    let raw_config = read_bounded(&config_path, CONFIG_MAX)?;
    let value: Value =
        serde_json::from_slice(&raw_config).map_err(|_| "invalid JSON configuration")?;
    let top = object_with_optional(
        &value,
        &[
            "schema_version",
            "experiment_id",
            "repository_id",
            "source_revision",
            "source_kind",
            "dataset_root",
            "tokenizer_artifact",
            "tokenizer_fit_record",
            "dataset_id",
            "dataset_manifest_sha256",
            "tokenizer_sha256",
            "optimizer",
            "schedule",
            "limits",
            "memory",
            "autograd",
            "gradient_clip",
            "seed",
        ],
        &["evaluation"],
    )?;
    if u64_value(top, "schema_version")? != 2 {
        return Err("unsupported configuration schema".into());
    }
    let experiment_id = string_value(top, "experiment_id")?.to_owned();
    if experiment_id.is_empty()
        || experiment_id.len() > 96
        || !experiment_id.as_bytes()[0].is_ascii_alphanumeric()
        || !experiment_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    {
        return Err("experiment_id must be a nonempty immutable identifier".into());
    }
    let source_revision = string_value(top, "source_revision")?.to_owned();
    if source_revision.len() != 40 || !source_revision.bytes().all(is_lower_hex) {
        return Err("source_revision must be a lowercase 40-character commit SHA".into());
    }
    let repository_id = string_value(top, "repository_id")?;
    let source_identity = attest_source_checkout(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        repository_id,
        &source_revision,
    )
    .map_err(|error| format!("source identity: {error}"))?;
    let source_repository_id = source_identity.repository_id().to_owned();
    let verified_source_revision = source_identity.revision().to_owned();
    if string_value(top, "source_kind")? != "manifest-v1-stream" {
        return Err("unsupported source_kind; only manifest-v1-stream is admitted".into());
    }
    let config_dir = config_path
        .parent()
        .ok_or("configuration has no parent directory")?;
    let dataset_root = resolve_config_path(config_dir, string_value(top, "dataset_root")?)?;
    let tokenizer_path = resolve_config_path(config_dir, string_value(top, "tokenizer_artifact")?)?;
    let fit_path = resolve_config_path(config_dir, string_value(top, "tokenizer_fit_record")?)?;

    let limits_obj = object(
        required(top, "limits")?,
        &[
            "microbatches_per_update",
            "max_updates",
            "consumed_tokens",
            "supervised_tokens",
            "cumulative_work",
            "owned_memory_bytes",
            "max_runtime_seconds",
            "max_attempt_runtime_seconds",
            "checkpoint_every_updates",
            "validation_every_updates",
        ],
    )?;
    let owned = usize_value(limits_obj, "owned_memory_bytes")?;
    let memory_obj = object(required(top, "memory")?, &["data_allowance_bytes"])?;
    let memory =
        TrainingMemoryPlan::candidate(usize_value(memory_obj, "data_allowance_bytes")?, owned)
            .map_err(|error| format!("memory plan: {error:?}"))?;
    let run_limits = RunLimits::new(
        usize_value(limits_obj, "microbatches_per_update")?,
        u64_value(limits_obj, "max_updates")?,
        u64_value(limits_obj, "consumed_tokens")?,
        u64_value(limits_obj, "supervised_tokens")?,
        u64_value(limits_obj, "cumulative_work")?,
        owned,
        Duration::from_secs(u64_value(limits_obj, "max_runtime_seconds")?),
        Duration::from_secs(u64_value(limits_obj, "max_attempt_runtime_seconds")?),
        u64_value(limits_obj, "checkpoint_every_updates")?,
        u64_value(limits_obj, "validation_every_updates")?,
    )
    .map_err(|error| format!("run limits: {error:?}"))?;
    let autograd_obj = object(
        required(top, "autograd")?,
        &["bytes", "values", "operations", "work"],
    )?;
    let autograd = ResourceLimits::new(
        usize_value(autograd_obj, "bytes")?,
        usize_value(autograd_obj, "values")?,
        usize_value(autograd_obj, "operations")?,
        usize_value(autograd_obj, "work")?,
    );
    if autograd.max_bytes() != usize_value(autograd_obj, "bytes")?
        || autograd.max_values() != usize_value(autograd_obj, "values")?
        || autograd.max_operations() != usize_value(autograd_obj, "operations")?
        || autograd.max_work() != usize_value(autograd_obj, "work")?
    {
        return Err("autograd caps exceed accepted project limits".into());
    }
    let optimizer_obj = object(
        required(top, "optimizer")?,
        &["beta1", "beta2", "epsilon", "weight_decay"],
    )?;
    let optimizer = AdamWConfig::new(
        f32_value(optimizer_obj, "beta1")?,
        f32_value(optimizer_obj, "beta2")?,
        f32_value(optimizer_obj, "epsilon")?,
        f32_value(optimizer_obj, "weight_decay")?,
    )
    .map_err(|error| format!("optimizer: {error:?}"))?;
    let schedule_obj = object(
        required(top, "schedule")?,
        &["total_updates", "warmup_updates", "peak_rate", "final_rate"],
    )?;
    let schedule = LinearWarmupDecay::new(
        u64_value(schedule_obj, "total_updates")?,
        u64_value(schedule_obj, "warmup_updates")?,
        f32_value(schedule_obj, "peak_rate")?,
        f32_value(schedule_obj, "final_rate")?,
    )
    .map_err(|error| format!("schedule: {error:?}"))?;
    let gradient_clip = match required(top, "gradient_clip")? {
        Value::Null => None,
        value => Some(as_f32(value, "gradient_clip")?),
    };
    let config = TrainingConfig::new(
        optimizer,
        schedule,
        run_limits,
        memory,
        autograd,
        gradient_clip,
        u64_value(top, "seed")?,
    )
    .map_err(|error| format!("training configuration: {error:?}"))?;
    let evaluation_limits = if let Some(value) = top.get("evaluation") {
        let evaluation = object(
            value,
            &[
                "max_batches",
                "max_consumed_tokens",
                "max_forward_work",
                "max_source_bytes",
                "max_batch_positions",
            ],
        )?;
        EvaluationLimits::new(
            u64_value(evaluation, "max_batches")?,
            u64_value(evaluation, "max_consumed_tokens")?,
            u64_value(evaluation, "max_forward_work")?,
            usize_value(evaluation, "max_source_bytes")?,
            usize_value(evaluation, "max_batch_positions")?,
        )
    } else {
        EvaluationLimits::new(
            1,
            65_536,
            ResourceLimits::MAX_WORK as u64,
            owned.min(64 * 1024 * 1024),
            65_536,
        )
    }
    .map_err(|error| format!("evaluation limits: {error:?}"))?;
    let config = config
        .with_validation_limits(evaluation_limits)
        .map_err(|error| format!("training evaluation policy: {error:?}"))?;

    let mut data_limits = DatasetLimits::MAX;
    // Train and validation streams are simultaneously live in Trainer. Split
    // reader and dataset-buffer reservations between both concrete streams.
    let per_stream_dataset_bytes = memory.dataset_buffer_bytes() / 2;
    let document_bytes = (per_stream_dataset_bytes / 2).max(1);
    let stream_buffer = per_stream_dataset_bytes
        .checked_sub(document_bytes)
        .ok_or("invalid dataset buffer split")?;
    let per_stream_reader_bytes = memory.reader_buffer_bytes() / 2;
    if stream_buffer < 4096 {
        return Err("dataset allowance is too small for the fixed training stream".into());
    }
    data_limits.metadata_bytes = memory.dataset_metadata_bytes();
    data_limits.document_bytes = document_bytes.min(DatasetLimits::MAX.document_bytes);
    data_limits.reader_bytes = per_stream_reader_bytes;
    data_limits.token_buffer_bytes = stream_buffer;
    data_limits.sequence_length = MODEL_T;
    data_limits.batch_size = 1;
    data_limits.batch_positions = MODEL_T;
    let dataset = ManifestValidator::open(&dataset_root, data_limits)
        .map_err(|error| format!("dataset admission: {error:?}"))?;

    let tokenizer_meta =
        fs::symlink_metadata(&tokenizer_path).map_err(|e| format!("tokenizer metadata: {e}"))?;
    if !tokenizer_meta.is_file() || tokenizer_meta.file_type().is_symlink() {
        return Err("tokenizer artifact must be a regular non-symlink file".into());
    }
    let artifact_size = usize::try_from(tokenizer_meta.len())
        .map_err(|_| "tokenizer artifact size does not fit this platform")?;
    let token_cap = memory.tokenizer_buffer_bytes();
    if artifact_size
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(4096))
        .is_none_or(|bytes| bytes > token_cap)
    {
        return Err(
            "tokenizer artifact and retained model exceed the reserved tokenizer budget".into(),
        );
    }
    let mut tokenizer_limits = TokenizerLimits::MAX;
    tokenizer_limits.buffer_bytes = token_cap - artifact_size;
    tokenizer_limits.artifact_bytes = tokenizer_limits.artifact_bytes.min(artifact_size);
    let tokenizer_bytes = read_bounded(&tokenizer_path, tokenizer_limits.artifact_bytes)?;
    if u64::try_from(tokenizer_bytes.len()).map_err(|_| "tokenizer length does not fit u64")?
        != tokenizer_meta.len()
    {
        return Err("tokenizer artifact changed while reading".into());
    }
    let tokenizer_sha: [u8; 32] = Sha256::digest(&tokenizer_bytes).into();
    if hex(&tokenizer_sha) != string_value(top, "tokenizer_sha256")? {
        return Err("tokenizer identity does not match configuration".into());
    }
    let fit_metadata = fs::symlink_metadata(&fit_path)
        .map_err(|error| format!("tokenizer fit metadata: {error}"))?;
    if !fit_metadata.is_file()
        || fit_metadata.file_type().is_symlink()
        || fit_metadata.len() > FIT_RECORD_MAX as u64
    {
        return Err("tokenizer fit record must be a regular file no larger than 16 KiB".into());
    }
    let fit_line = read_bounded(&fit_path, FIT_RECORD_MAX)?;
    let tokenizer = dataset
        .verify_fit(&fit_line, &tokenizer_bytes, tokenizer_limits)
        .map_err(|error| format!("dataset-bound tokenizer fit admission: {error:?}"))?;
    let tokenizer_model = tokenizer.model();
    if tokenizer_model.vocabulary_size() != 292 {
        return Err("tokenizer does not match the accepted 292-token model or memory cap".into());
    }
    let stream = StreamConfig {
        batch: BatchConfig {
            sequence_length: MODEL_T,
            stride: MODEL_T,
            batch_size: 1,
            tail: Tail::Drop,
            pad_final_batch: false,
        },
        boundaries: Boundaries {
            bos: true,
            eos: true,
        },
        tokenizer_limits: TokenizerLimits {
            buffer_bytes: stream_buffer,
            ..TokenizerLimits::MAX
        },
    };
    let train_source = dataset
        .fitted_batches(Split::Train, &tokenizer, stream)
        .map_err(|error| format!("train stream admission: {error:?}"))?;
    let validation_source = dataset
        .fitted_batches(Split::Validation, &tokenizer, stream)
        .map_err(|error| format!("validation stream admission: {error:?}"))?;
    let train_order = train_source.order_identity();
    let validation_order = validation_source.order_identity();
    let source_limit = memory
        .dataset_metadata_bytes()
        .checked_add(memory.dataset_buffer_bytes())
        .and_then(|bytes| bytes.checked_add(memory.reader_buffer_bytes()))
        .ok_or("dataset source memory limit overflow")?;
    let mut simultaneous_source_bytes = 0usize;
    for requested in [
        train_source.reserved_memory_bytes(),
        validation_source.reserved_memory_bytes(),
    ] {
        let requested =
            requested.map_err(|error| format!("source memory accounting: {error:?}"))?;
        simultaneous_source_bytes = simultaneous_source_bytes
            .checked_add(requested)
            .ok_or("combined dataset source memory overflow")?;
        if requested > source_limit {
            return Err(format!(
                "standard source reservation {requested} exceeds admitted {source_limit} bytes"
            ));
        }
    }
    simultaneous_source_bytes = simultaneous_source_bytes
        .checked_sub(dataset.retained_metadata_bytes())
        .ok_or("dataset metadata accounting underflow")?;
    if simultaneous_source_bytes > source_limit {
        return Err(format!(
            "combined standard-source reservation {simultaneous_source_bytes} exceeds admitted {source_limit} bytes"
        ));
    }
    if dataset.manifest().dataset_id != string_value(top, "dataset_id")?
        || hex(&dataset.manifest_sha256()) != string_value(top, "dataset_manifest_sha256")?
    {
        return Err("dataset identity does not match immutable configuration".into());
    }
    let identity = RunIdentity::from_standard_dataset(
        source_identity,
        &dataset,
        tokenizer_model,
        train_order,
        validation_order,
    )
    .map_err(|error| format!("run identity: {error:?}"))?;
    config
        .validate(&identity, tokenizer_model)
        .map_err(|error| format!("configuration identity: {error:?}"))?;
    let tokenizer_accounted = tokenizer
        .model()
        .retained_bytes()
        .map_err(|error| format!("tokenizer memory: {error:?}"))?;
    let tokenizer_accounted = tokenizer_accounted
        .checked_add(tokenizer.artifact().len())
        .and_then(|bytes| {
            let record = tokenizer.record();
            let strings = record
                .dataset_id
                .len()
                .checked_add(record.training_manifest_sha256.len())?
                .checked_add(record.artifact_sha256.len())?;
            std::mem::size_of_val(record)
                .checked_add(strings.checked_mul(2)?)?
                .checked_add(bytes)
        })
        .ok_or("tokenizer memory accounting overflow")?;
    if tokenizer_accounted > memory.tokenizer_buffer_bytes()
        || dataset.retained_metadata_bytes() > memory.dataset_metadata_bytes()
    {
        return Err("source retained memory exceeds its reserved child budget".into());
    }
    Ok(Prepared {
        config_sha256: Sha256::digest(&raw_config).into(),
        experiment_id,
        source_repository_id,
        source_revision: verified_source_revision,
        dataset_id: dataset.manifest().dataset_id.clone(),
        identity,
        config,
        dataset,
        tokenizer,
        stream,
        dataset_root,
    })
}

fn execute(root: &Path, prepared: &mut Prepared, resume: bool) -> CliResult<()> {
    let train = prepared
        .dataset
        .fitted_batches(Split::Train, &prepared.tokenizer, prepared.stream)
        .map_err(|error| format!("train stream: {error:?}"))?;
    let mut validation = prepared
        .dataset
        .fitted_batches(Split::Validation, &prepared.tokenizer, prepared.stream)
        .map_err(|error| format!("validation stream: {error:?}"))?;
    let stop =
        StopToken::from_artifact_root(root).map_err(|error| format!("stop token: {error:?}"))?;
    let constructed = if resume {
        Trainer::resume(
            root,
            prepared.identity.clone(),
            prepared.config,
            prepared.tokenizer.model(),
            train,
            stop,
        )
    } else {
        Trainer::new(
            root,
            prepared.identity.clone(),
            prepared.config,
            prepared.tokenizer.model(),
            train,
            stop,
        )
    };
    let mut trainer = match constructed {
        Ok(trainer) => trainer,
        Err(error) => {
            let detail = format!("{error:?}");
            publish_startup_failure(root, prepared, &detail)
                .map_err(|publish| format!("trainer initialization failed: {detail}; failure-report publication failed: {publish}"))?;
            return Err(format!("trainer initialization failed: {detail}"));
        }
    };
    let process_identity = ProcessIdentityGuard::register(root)?;
    remove_failure_report(root)?;
    let run_result = trainer.run(Some(&mut validation));
    let cleanup_result = process_identity.remove();
    let report = run_result.map_err(|error| format!("training failed: {error:?}"))?;
    cleanup_result?;
    println!(
        "{}",
        json!({
            "experiment_id": prepared.experiment_id,
            "repository_id": prepared.source_repository_id,
            "run_config_sha256": hex(&prepared.config.digest(&prepared.identity)),
            "stop_reason": format!("{:?}", report.stop_reason),
            "completed_updates": report.counters.completed_updates,
            "committed_cursor": report.counters.committed_cursor,
            "attempted_consumed_tokens": report.counters.attempted_consumed_tokens,
            "committed_consumed_tokens": report.counters.committed_consumed_tokens,
            "committed_supervised_tokens": report.counters.committed_supervised_tokens,
            "cumulative_work": report.counters.cumulative_work,
            "charged_owned_peak_bytes": report.owned_memory_peak_bytes,
            "autograd_ledger_peak_bytes": report.autograd_ledger_peak_bytes,
            "process_vm_hwm_bytes": report.process_vm_hwm_bytes,
        })
    );
    Ok(())
}

fn publish_startup_failure(root: &Path, prepared: &Prepared, detail: &str) -> CliResult<()> {
    let error: String = detail.chars().take(4096).collect();
    let progress = RunJournal::inspect(
        root,
        prepared.config.digest(&prepared.identity),
        prepared.config.limits(),
    )
    .ok();
    let value = json!({
        "schema_version":1,
        "status":"failed",
        "phase":"trainer_initialization",
        "experiment_id":prepared.experiment_id,
        "repository_id":prepared.source_repository_id,
        "source_sha":prepared.source_revision,
        "config_sha256":hex(&prepared.config.digest(&prepared.identity)),
        "model_id":xynoos_training::MODEL_ID,
        "dataset_id":prepared.dataset_id,
        "error":error,
        "journal":progress.map(|p| json!({
            "attempted_consumed_tokens":p.attempted_consumed_tokens,
            "attempted_supervised_tokens":p.attempted_supervised_tokens,
            "cumulative_work":p.cumulative_work,
            "active_runtime_ns":p.active_runtime_ns,
            "reserved_runtime_ns":p.reserved_runtime_ns,
            "attempted_updates":p.attempted_updates,
            "checkpoint_sequence":p.checkpoint_sequence,
        })),
    });
    let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
    if bytes.len() > 64 * 1024 {
        return Err("startup failure report exceeds 64 KiB".into());
    }
    let path = root.join("failure-report.json");
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
        Ok(_) => return Err("failure-report path is not a regular file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("failure-report metadata: {error}")),
    }
    static NEXT_FAILURE_REPORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let temp = root.join(format!(
        ".startup-failure-{}-{}.tmp",
        std::process::id(),
        NEXT_FAILURE_REPORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|error| format!("failure-report temporary file: {error}"))?;
    let result = (|| {
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temp, &path)?;
        File::open(root)?.sync_all()?;
        Ok::<(), std::io::Error>(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temp);
        return Err(format!("failure-report publication: {error}"));
    }
    Ok(())
}

fn remove_failure_report(root: &Path) -> CliResult<()> {
    let path = root.join("failure-report.json");
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            fs::remove_file(path)
                .map_err(|error| format!("remove prior failure report: {error}"))?;
            File::open(root)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| format!("sync failure-report cleanup: {error}"))?;
            Ok(())
        }
        Ok(_) => Err("failure-report path is not a regular file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failure-report metadata: {error}")),
    }
}

fn detach(args: &Args, config: &Path, experiment_id: &str) -> Result<(), i32> {
    clear_stale_process_identity(&args.root).map_err(report)?;
    let stdout = File::options()
        .write(true)
        .open("/dev/null")
        .map_err(|error| report(format!("stdout redirection: {error}")))?;
    let stderr_path = args.root.join("process.stderr");
    match fs::symlink_metadata(&stderr_path) {
        Ok(metadata)
            if metadata.is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() <= 64 * 1024 => {}
        Ok(_) => return Err(report("runtime stderr path is invalid or exceeds 64 KiB")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(report(format!("runtime stderr metadata: {error}"))),
    }
    let stderr = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&stderr_path)
        .map_err(|error| report(format!("runtime stderr: {error}")))?;
    let executable =
        env::current_exe().map_err(|error| report(format!("current executable: {error}")))?;
    let mut child = Command::new("setsid")
        .arg(executable)
        .arg(if args.command == CommandKind::Resume {
            "resume"
        } else {
            "run"
        })
        .arg("--config")
        .arg(config)
        .arg("--artifact-root")
        .arg(&args.root)
        .env("XYNOOS_TRAIN_DETACHED_CHILD", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|error| report(format!("detached launch: {error}")))?;
    let started = std::time::Instant::now();
    let identity = loop {
        if let Ok(Some(identity)) = read_process_identity(&args.root)
            && identity.is_same_process_alive().unwrap_or(false)
            && live_run_lock(&args.root).unwrap_or(false)
        {
            break identity;
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| report(format!("detached startup status: {error}")))?
        {
            return Err(report(format!(
                "detached child exited before acquiring lineage ownership: {status}"
            )));
        }
        if started.elapsed() >= Duration::from_secs(5) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(report(
                "detached child did not acquire lineage ownership within five seconds",
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    // The new session survives controller exit and terminal hangup; all stdio
    // is redirected before the controller releases the child.
    drop(child);
    println!(
        "{}",
        json!({"status":"launched","pid":identity.pid,"experiment_id":experiment_id,"artifact_root":args.root})
    );
    Ok(())
}

fn acquire_launch_lock(root: &Path) -> CliResult<LaunchLock> {
    let path = root.join(".launch.lock");
    match AdvisoryLock::acquire(&path) {
        Ok(lock) => Ok(LaunchLock { _lock: lock }),
        Err(LockError::Contended) => Err("another run or detached launch owns admission".into()),
        Err(LockError::RecoveryFailure(kind)) => {
            Err(format!("detached launch lock recovery failed: {kind:?}"))
        }
    }
}

fn check_root_state(root: &Path, prepared: &Prepared, command: CommandKind) -> CliResult<()> {
    if command == CommandKind::Validate {
        if sidecar_exists(root)? {
            verify_sidecar(root, prepared)?;
        }
        return Ok(());
    }
    if matches!(command, CommandKind::Run | CommandKind::Resume) {
        recover_sidecar_staging(root)?;
    }
    if command == CommandKind::Run {
        if sidecar_exists(root)? {
            if initial_sidecar_recovery(root, prepared)? {
                return Ok(());
            }
            return Err("fresh run cannot reuse an initialized artifact root".into());
        }
        for entry in fs::read_dir(root).map_err(|error| format!("artifact root: {error}"))? {
            let entry = entry.map_err(|error| format!("artifact root entry: {error}"))?;
            if matches!(
                entry.file_name().to_str(),
                Some(".launch.lock" | ".run.lock")
            ) && entry
                .file_type()
                .map_err(|error| format!("artifact entry type: {error}"))?
                .is_file()
            {
                continue;
            }
            if env::var_os("XYNOOS_TRAIN_DETACHED_CHILD").is_some()
                && entry.file_name() == "process.stderr"
                && entry
                    .file_type()
                    .map_err(|error| format!("artifact entry type: {error}"))?
                    .is_file()
            {
                continue;
            }
            return Err(format!(
                "fresh run requires an empty artifact root (found {})",
                entry.file_name().to_string_lossy()
            ));
        }
        return Ok(());
    }
    verify_sidecar(root, prepared)
}

fn initial_sidecar_recovery(root: &Path, prepared: &Prepared) -> CliResult<bool> {
    if !sidecar_exists(root)? {
        return Ok(false);
    }
    verify_sidecar(root, prepared)?;
    for name in [
        "run.journal",
        "checkpoint.xyckpt",
        "metrics.jsonl",
        "handoff.json",
        "final-report.json",
        PROCESS_PID,
    ] {
        if regular_file_exists(root, name)? {
            return Ok(false);
        }
    }
    for entry in fs::read_dir(root).map_err(|error| format!("artifact root: {error}"))? {
        let entry = entry.map_err(|error| format!("artifact root entry: {error}"))?;
        if !matches!(
            entry.file_name().to_str(),
            Some(SIDECAR | ".launch.lock" | ".run.lock" | "process.stderr" | "failure-report.json")
        ) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn regular_file_exists(root: &Path, name: &str) -> CliResult<bool> {
    match fs::symlink_metadata(root.join(name)) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(format!("artifact path {name} is not a regular file")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("artifact metadata for {name}: {error}")),
    }
}

fn ensure_artifact_root_outside_checkout(root: &Path) -> CliResult<()> {
    let artifact_root = root
        .canonicalize()
        .map_err(|error| format!("artifact root canonicalization: {error}"))?;
    let output = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|error| format!("source checkout lookup: {error}"))?;
    if !output.status.success() {
        return Err("could not identify source checkout for artifact placement".into());
    }
    let checkout = PathBuf::from(
        String::from_utf8(output.stdout)
            .map_err(|_| "source checkout path is not valid UTF-8")?
            .trim(),
    )
    .canonicalize()
    .map_err(|error| format!("source checkout canonicalization: {error}"))?;
    if artifact_root.starts_with(checkout) {
        return Err("artifact root must be outside the source checkout".into());
    }
    Ok(())
}

fn process_start_time(pid: u32) -> CliResult<u64> {
    let stat_path = format!("/proc/{pid}/stat");
    let stat = fs::read_to_string(&stat_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            "process not found".to_owned()
        } else {
            format!("read process identity: {error}")
        }
    })?;
    let (_, fields) = stat
        .rsplit_once(") ")
        .ok_or("invalid Linux process stat record")?;
    fields
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| "Linux process stat is missing start time".to_owned())?
        .parse::<u64>()
        .map_err(|_| "invalid Linux process start time".into())
}

fn process_state(pid: u32) -> CliResult<String> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            "process not found".to_owned()
        } else {
            format!("read process identity: {error}")
        }
    })?;
    let (_, fields) = stat
        .rsplit_once(") ")
        .ok_or("invalid Linux process stat record")?;
    fields
        .split_whitespace()
        .next()
        .map(str::to_owned)
        .ok_or_else(|| "Linux process stat is missing state".into())
}

fn read_boot_id() -> CliResult<String> {
    let value = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|error| format!("read Linux boot identity: {error}"))?
        .trim()
        .to_owned();
    if !is_valid_boot_id(&value) {
        return Err("invalid Linux boot identity".into());
    }
    Ok(value)
}

fn is_valid_boot_id(value: &str) -> bool {
    value.len() == 36
        && value.chars().enumerate().all(|(index, character)| {
            if [8, 13, 18, 23].contains(&index) {
                character == '-'
            } else {
                character.is_ascii_hexdigit()
            }
        })
}

fn read_process_identity(root: &Path) -> CliResult<Option<ProcessIdentity>> {
    let path = root.join(PROCESS_PID);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("process identity metadata: {error}")),
    };
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 256 {
        return Err("invalid process identity file".into());
    }
    let text =
        fs::read_to_string(path).map_err(|error| format!("read process identity: {error}"))?;
    ProcessIdentity::parse(&text).map(Some)
}

fn write_process_identity(root: &Path, identity: &ProcessIdentity) -> CliResult<()> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let target = root.join(PROCESS_PID);
    match fs::symlink_metadata(&target) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
        Ok(_) => return Err("process identity target is not a regular file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("process identity target metadata: {error}")),
    }
    let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let temp = root.join(format!(
        ".process-identity-{}-{serial}.tmp",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp)
        .map_err(|error| format!("process identity staging file: {error}"))?;
    let result = (|| {
        file.write_all(identity.encode().as_bytes())
            .map_err(|error| format!("write process identity: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync process identity: {error}"))?;
        fs::rename(&temp, &target).map_err(|error| format!("publish process identity: {error}"))?;
        sync_directory(root, "process identity publication")
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn sync_directory(root: &Path, operation: &str) -> CliResult<()> {
    File::open(root)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync directory after {operation}: {error}"))
}

fn clear_stale_process_identity(root: &Path) -> CliResult<()> {
    if live_run_lock(root)? {
        return Err("another process owns this experiment lineage".into());
    }
    let Some(previous) = read_process_identity(root)? else {
        return Ok(());
    };
    if previous.is_same_process_alive()? {
        return Err("live process identity exists without the journal ownership lock".into());
    }
    fs::remove_file(root.join(PROCESS_PID))
        .map_err(|error| format!("remove stale process identity: {error}"))?;
    sync_directory(root, "stale process identity cleanup")
}

fn validation_json(prepared: &Prepared, root: &Path) -> Value {
    json!({
        "status":"valid",
        "experiment_id":prepared.experiment_id,
        "repository_id":prepared.source_repository_id,
        "source_revision":prepared.source_revision,
        "training_implementation_revision":prepared.source_revision,
        "model_id":xynoos_training::MODEL_ID,
        "parameters":914400,
        "tensors":74,
        "dataset_id":prepared.dataset_id,
        "dataset_manifest_sha256":hex(&prepared.identity.dataset_manifest_sha256()),
        "tokenizer_sha256":hex(&prepared.identity.tokenizer_sha256()),
        "train_order_sha256":hex(&prepared.identity.data_order_identity()),
        "validation_order_sha256":hex(&prepared.identity.validation_order_identity()),
        "dataset_root":prepared.dataset_root,
        "artifact_root":root,
        "run_config_sha256":hex(&prepared.config.digest(&prepared.identity)),
        "configuration_file_sha256":hex(&prepared.config_sha256),
        "limits":{
            "max_updates":prepared.config.limits().max_updates(),
            "consumed_tokens":prepared.config.limits().consumed_tokens(),
            "supervised_tokens":prepared.config.limits().supervised_tokens(),
            "cumulative_work":prepared.config.limits().cumulative_work(),
            "owned_memory_bytes":prepared.config.limits().owned_memory_bytes(),
            "max_runtime_seconds":prepared.config.limits().max_runtime().as_secs(),
            "max_attempt_runtime_seconds":prepared.config.limits().max_attempt_runtime().as_secs(),
        },
        "evaluation_limits":{
            "max_batches":prepared.config.validation_limits().max_batches(),
            "max_consumed_tokens":prepared.config.validation_limits().max_consumed_tokens(),
            "max_forward_work":prepared.config.validation_limits().max_forward_work(),
            "max_source_bytes":prepared.config.validation_limits().max_source_bytes(),
            "max_batch_positions":prepared.config.validation_limits().max_batch_positions(),
        },
        "source_kind":"manifest-v1-stream",
        "batch":{"B":1,"T":MODEL_T},
    })
}

fn write_sidecar(root: &Path, prepared: &Prepared) -> CliResult<()> {
    let value = json!({
        "schema_version":2,
        "experiment_id":prepared.experiment_id,
        "repository_id":prepared.source_repository_id,
        "source_revision":prepared.source_revision,
        "dataset_id":prepared.dataset_id,
        "configuration_file_sha256":hex(&prepared.config_sha256),
        "run_config_sha256":hex(&prepared.config.digest(&prepared.identity)),
    });
    let mut bytes = serde_json::to_vec(&value)
        .map_err(|error| format!("serialize experiment identity: {error}"))?;
    bytes.push(b'\n');
    publish_initial_sidecar(root, &bytes, |_| Ok(()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidecarBoundary {
    BeforeStaging,
    AfterStaging,
    AfterWrite,
    AfterFileSync,
    AfterPublish,
    BeforePublishDirectorySync,
    AfterPublishDirectorySync,
    AfterTemporaryRemoval,
    BeforeCleanupDirectorySync,
    AfterCleanupDirectorySync,
}

fn publish_initial_sidecar(
    root: &Path,
    bytes: &[u8],
    mut inject: impl FnMut(SidecarBoundary) -> CliResult<()>,
) -> CliResult<()> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let target = root.join(SIDECAR);
    let temp = root.join(format!(
        ".experiment.json-init-{}-{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let mut published = false;
    let result = (|| {
        inject(SidecarBoundary::BeforeStaging)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)
            .map_err(|error| format!("experiment identity staging: {error}"))?;
        inject(SidecarBoundary::AfterStaging)?;
        file.write_all(bytes)
            .map_err(|error| format!("experiment identity write: {error}"))?;
        inject(SidecarBoundary::AfterWrite)?;
        file.sync_all()
            .map_err(|error| format!("experiment identity file sync: {error}"))?;
        inject(SidecarBoundary::AfterFileSync)?;
        fs::hard_link(&temp, &target)
            .map_err(|error| format!("experiment identity atomic publication: {error}"))?;
        published = true;
        inject(SidecarBoundary::AfterPublish)?;
        inject(SidecarBoundary::BeforePublishDirectorySync)?;
        sync_directory(root, "experiment identity publication")?;
        inject(SidecarBoundary::AfterPublishDirectorySync)?;
        fs::remove_file(&temp)
            .map_err(|error| format!("experiment identity staging cleanup: {error}"))?;
        inject(SidecarBoundary::AfterTemporaryRemoval)?;
        inject(SidecarBoundary::BeforeCleanupDirectorySync)?;
        sync_directory(root, "experiment identity staging cleanup")?;
        inject(SidecarBoundary::AfterCleanupDirectorySync)?;
        Ok(())
    })();
    if result.is_err() {
        if fs::remove_file(&temp).is_ok() {
            let _ = sync_directory(root, "failed experiment identity staging cleanup");
        }
        if published {
            // The canonical sidecar is complete. Keep it so a restart can
            // verify and resume the initial, pre-journal state.
        }
    }
    result
}

fn recover_sidecar_staging(root: &Path) -> CliResult<()> {
    let mut removed = false;
    for entry in fs::read_dir(root).map_err(|error| format!("artifact root: {error}"))? {
        let entry = entry.map_err(|error| format!("artifact root entry: {error}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with(".experiment.json-init-") || !name.ends_with(".tmp") {
            continue;
        }
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|error| format!("sidecar staging metadata: {error}"))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err("invalid experiment identity staging entry".into());
        }
        fs::remove_file(entry.path())
            .map_err(|error| format!("remove interrupted sidecar staging: {error}"))?;
        removed = true;
    }
    if removed {
        sync_directory(root, "interrupted sidecar staging recovery")?;
    }
    Ok(())
}

fn verify_sidecar(root: &Path, prepared: &Prepared) -> CliResult<()> {
    let value = read_sidecar(root)?;
    if value["schema_version"] != 2
        || value["experiment_id"] != prepared.experiment_id
        || value["repository_id"] != prepared.source_repository_id
        || value["source_revision"] != prepared.source_revision
        || value["dataset_id"] != prepared.dataset_id
        || value["configuration_file_sha256"] != hex(&prepared.config_sha256)
        || value["run_config_sha256"] != hex(&prepared.config.digest(&prepared.identity))
    {
        return Err(
            "artifact root is bound to a different immutable experiment configuration".into(),
        );
    }
    Ok(())
}

fn read_sidecar(root: &Path) -> CliResult<Value> {
    let path = root.join(SIDECAR);
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("experiment identity missing: {error}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 4096 {
        return Err("invalid experiment identity file".into());
    }
    let bytes = read_bounded(&path, 4096)?;
    serde_json::from_slice(&bytes).map_err(|_| "corrupt experiment identity file".into())
}

fn live_run_lock(root: &Path) -> CliResult<bool> {
    match AdvisoryLock::is_held(&root.join(".run.lock")) {
        Ok(held) => Ok(held),
        Err(LockError::Contended) => Ok(true),
        Err(LockError::RecoveryFailure(kind)) => Err(format!("run lock recovery failed: {kind:?}")),
    }
}

fn sidecar_exists(root: &Path) -> CliResult<bool> {
    match fs::symlink_metadata(root.join(SIDECAR)) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err("invalid experiment identity path".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("experiment identity metadata: {error}")),
    }
}

fn read_optional_json(path: &Path) -> Result<Option<Value>, i32> {
    match fs::symlink_metadata(path) {
        Ok(metadata)
            if metadata.is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() <= 64 * 1024 =>
        {
            let data = read_bounded(path, 64 * 1024).map_err(report)?;
            serde_json::from_slice(&data)
                .map(Some)
                .map_err(|_| report("invalid final report JSON"))
        }
        Ok(_) => Err(report("invalid final report file")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(report(format!("final report metadata: {error}"))),
    }
}

fn resolve_config_path(base: &Path, path: &str) -> CliResult<PathBuf> {
    let given = Path::new(path);
    if given.as_os_str().is_empty()
        || given
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(
            "configuration paths must be nonempty and must not traverse parent directories".into(),
        );
    }
    let joined = if given.is_absolute() {
        given.to_path_buf()
    } else {
        base.join(given)
    };
    canonicalize_non_symlink(&joined).map_err(|error| format!("configured path: {error}"))
}

fn canonicalize_non_symlink(path: &Path) -> CliResult<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "symlink path component is not admitted: {}",
                current.display()
            ));
        }
    }
    fs::canonicalize(absolute).map_err(|error| error.to_string())
}

fn validate_root(root: &Path) -> CliResult<()> {
    if !root.is_absolute()
        || root
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err("artifact root must be absolute and must not use parent traversal".into());
    }
    let mut current = PathBuf::new();
    for component in root.components() {
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| format!("artifact root path: {error}"))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(
                "artifact root and every parent must be a real directory, not a symlink".into(),
            );
        }
    }
    Ok(())
}

fn read_bounded(path: &Path, limit: usize) -> CliResult<Vec<u8>> {
    let file = File::open(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    let reserve = file
        .metadata()
        .map_err(|error| format!("metadata: {error}"))?
        .len()
        .min(limit as u64) as usize;
    bytes
        .try_reserve_exact(reserve)
        .map_err(|_| "bounded allocation failed")?;
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() > limit {
        return Err(format!("{} exceeds its {limit}-byte cap", path.display()));
    }
    Ok(bytes)
}

fn object<'a>(value: &'a Value, allowed: &[&str]) -> CliResult<&'a Map<String, Value>> {
    let object = value.as_object().ok_or("configuration object expected")?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown configuration field: {key}"));
        }
    }
    for key in allowed {
        if !object.contains_key(*key) {
            return Err(format!("missing configuration field: {key}"));
        }
    }
    Ok(object)
}

fn object_with_optional<'a>(
    value: &'a Value,
    required: &[&str],
    optional: &[&str],
) -> CliResult<&'a Map<String, Value>> {
    let object = value.as_object().ok_or("configuration object expected")?;
    for key in object.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return Err(format!("unknown configuration field: {key}"));
        }
    }
    for key in required {
        if !object.contains_key(*key) {
            return Err(format!("missing configuration field: {key}"));
        }
    }
    Ok(object)
}

fn required<'a>(object: &'a Map<String, Value>, key: &str) -> CliResult<&'a Value> {
    object
        .get(key)
        .ok_or_else(|| format!("missing configuration field: {key}"))
}

fn string_value<'a>(object: &'a Map<String, Value>, key: &str) -> CliResult<&'a str> {
    required(object, key)?
        .as_str()
        .ok_or_else(|| format!("{key} must be a string"))
}

fn u64_value(object: &Map<String, Value>, key: &str) -> CliResult<u64> {
    required(object, key)?
        .as_u64()
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}

fn usize_value(object: &Map<String, Value>, key: &str) -> CliResult<usize> {
    usize::try_from(u64_value(object, key)?)
        .map_err(|_| format!("{key} does not fit this platform"))
}

fn f32_value(object: &Map<String, Value>, key: &str) -> CliResult<f32> {
    as_f32(required(object, key)?, key)
}

fn as_f32(value: &Value, key: &str) -> CliResult<f32> {
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{key} must be a finite number"))? as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{key} must be finite"))
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn is_lower_hex(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
}

fn report(error: impl std::fmt::Display) -> i32 {
    eprintln!("xynoos-train: {error}");
    2
}

fn print_help() {
    eprintln!(
        "xynoos-train <validate|run|resume|status|stop|inspect-checkpoint> --artifact-root ABSOLUTE_DIR [--config IMMUTABLE.json] [--detach]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = env::temp_dir().join(format!(
                "task006-cli-{label}-{}-{}",
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

    fn temp_entries(root: &Path) -> Vec<PathBuf> {
        fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect()
    }

    #[test]
    fn sidecar_failures_before_publication_clean_staging() {
        for boundary in [
            SidecarBoundary::BeforeStaging,
            SidecarBoundary::AfterStaging,
            SidecarBoundary::AfterWrite,
            SidecarBoundary::AfterFileSync,
        ] {
            let scratch = Scratch::new("sidecar-prepublish");
            let bytes = b"{\"schema_version\":2}\n";
            let result = publish_initial_sidecar(&scratch.0, bytes, |current| {
                if current == boundary {
                    Err("injected sidecar failure".into())
                } else {
                    Ok(())
                }
            });
            assert!(result.is_err(), "boundary {boundary:?} did not fail");
            assert!(!scratch.0.join(SIDECAR).exists());
            assert!(
                temp_entries(&scratch.0).is_empty(),
                "staging file remained after {boundary:?}"
            );
            recover_sidecar_staging(&scratch.0).unwrap();
            assert!(temp_entries(&scratch.0).is_empty());
        }
    }

    #[test]
    fn sidecar_failures_after_atomic_publish_leave_a_complete_recoverable_identity() {
        for boundary in [
            SidecarBoundary::AfterPublish,
            SidecarBoundary::BeforePublishDirectorySync,
            SidecarBoundary::AfterPublishDirectorySync,
            SidecarBoundary::AfterTemporaryRemoval,
            SidecarBoundary::BeforeCleanupDirectorySync,
            SidecarBoundary::AfterCleanupDirectorySync,
        ] {
            let scratch = Scratch::new("sidecar-postpublish");
            let bytes = b"{\"schema_version\":2,\"experiment_id\":\"stable\"}\n";
            let result = publish_initial_sidecar(&scratch.0, bytes, |current| {
                if current == boundary {
                    Err("injected sidecar failure".into())
                } else {
                    Ok(())
                }
            });
            assert!(result.is_err(), "boundary {boundary:?} did not fail");
            assert_eq!(fs::read(scratch.0.join(SIDECAR)).unwrap(), bytes);
            assert!(
                temp_entries(&scratch.0) == [scratch.0.join(SIDECAR)],
                "staging file remained after {boundary:?}"
            );
            assert!(read_sidecar(&scratch.0).unwrap()["experiment_id"] == "stable");
        }
    }

    #[test]
    fn sidecar_publication_preserves_existing_artifacts_and_recovers_orphan_staging() {
        let scratch = Scratch::new("sidecar-existing");
        let target = scratch.0.join(SIDECAR);
        fs::write(&target, b"existing artifact\n").unwrap();
        assert!(publish_initial_sidecar(&scratch.0, b"new artifact\n", |_| Ok(())).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"existing artifact\n");

        fs::remove_file(&target).unwrap();
        let orphan = scratch.0.join(".experiment.json-init-999-1.tmp");
        fs::write(&orphan, b"incomplete staging data").unwrap();
        recover_sidecar_staging(&scratch.0).unwrap();
        assert!(!orphan.exists());
        publish_initial_sidecar(&scratch.0, b"{\"schema_version\":2}\n", |_| Ok(())).unwrap();
        assert_eq!(fs::read(target).unwrap(), b"{\"schema_version\":2}\n");
    }

    #[test]
    fn process_identity_detects_pid_reuse_by_start_time_and_boot_id() {
        let current = ProcessIdentity::current().unwrap();
        let encoded = current.encode();
        assert_eq!(ProcessIdentity::parse(&encoded).unwrap(), current);
        assert!(current.is_same_process_alive().unwrap());

        let reused = ProcessIdentity {
            start_time_ticks: current.start_time_ticks + 1,
            ..current
        };
        assert!(!reused.is_same_process_alive().unwrap());

        let missing = ProcessIdentity {
            pid: u32::MAX,
            ..ProcessIdentity::current().unwrap()
        };
        assert!(!missing.is_same_process_alive().unwrap());
    }
}
