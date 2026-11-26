#![forbid(unsafe_code)]

use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
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
    CheckpointCodec, LinearWarmupDecay, RunIdentity, RunJournal, RunLimits, StopToken, Trainer,
    TrainingConfig, TrainingMemoryPlan,
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

struct LaunchLock(PathBuf);

impl Drop for LaunchLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn main() {
    let values: Vec<String> = env::args().skip(1).collect();
    if values.iter().any(|item| item == "--help" || item == "-h") {
        print_help();
        return;
    }
    let detached_child = env::var_os("XYNOOS_TRAIN_DETACHED_CHILD").is_some();
    let parsed = parse_args(values);
    let detached_root = parsed.as_ref().ok().map(|args| args.root.clone());
    let mut code = match parsed {
        Ok(args) => {
            if detached_child && let Err(error) = register_process_pid(&args.root) {
                eprintln!("xynoos-train: detached startup: {error}");
                2
            } else {
                dispatch(args).err().unwrap_or(0)
            }
        }
        Err(error) => {
            eprintln!("xynoos-train: {error}");
            2
        }
    };
    if detached_child
        && let Some(root) = detached_root
        && let Err(error) = remove_process_pid(&root)
    {
        eprintln!("xynoos-train: detached cleanup: {error}");
        if code == 0 {
            code = 2;
        }
    }
    if code != 0 {
        std::process::exit(code);
    }
}

fn dispatch(args: Args) -> Result<(), i32> {
    if let Err(error) = validate_root(&args.root) {
        return Err(report(error));
    }
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
            match args.command {
                CommandKind::Validate => {
                    let result = validation_json(&prepared, &args.root);
                    println!("{result}");
                    Ok(())
                }
                CommandKind::Run | CommandKind::Resume => {
                    if args.detached && env::var_os("XYNOOS_TRAIN_DETACHED_CHILD").is_none() {
                        let experiment_id = prepared.experiment_id.clone();
                        drop(prepared);
                        return detach(&args, config_path, &experiment_id);
                    }
                    if args.command == CommandKind::Run {
                        write_sidecar(&args.root, &prepared).map_err(report)?;
                    } else {
                        verify_sidecar(&args.root, &prepared).map_err(report)?;
                    }
                    execute(
                        &args.root,
                        &mut prepared,
                        args.command == CommandKind::Resume,
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
    let top = object(
        &value,
        &[
            "schema_version",
            "experiment_id",
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
    )?;
    if u64_value(top, "schema_version")? != 1 {
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
        source_revision.clone(),
        true,
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
        source_revision,
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
    remove_failure_report(root)?;
    let report = trainer
        .run(Some(&mut validation))
        .map_err(|error| format!("training failed: {error:?}"))?;
    println!(
        "{}",
        json!({
            "experiment_id": prepared.experiment_id,
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
    let _launch_lock = acquire_launch_lock(&args.root).map_err(report)?;
    match fs::symlink_metadata(args.root.join(PROCESS_PID)) {
        Ok(_) => {
            let pid = read_process_pid(&args.root).map_err(report)?;
            if Path::new("/proc").join(pid.to_string()).exists() {
                return Err(report("a detached training process is already active"));
            }
            fs::remove_file(args.root.join(PROCESS_PID))
                .map_err(|error| report(format!("remove stale process identity: {error}")))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(report(format!("process identity metadata: {error}"))),
    }
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
    let pid = loop {
        if let Ok(pid) = read_process_pid(&args.root)
            && Path::new("/proc").join(pid.to_string()).exists()
        {
            break pid;
        }
        if started.elapsed() >= Duration::from_secs(5) {
            let _ = child.kill();
            return Err(report(
                "detached child did not publish its process identity within five seconds",
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    // The new session survives controller exit and terminal hangup; all stdio
    // is redirected before the controller releases the child.
    drop(child);
    println!(
        "{}",
        json!({"status":"launched","pid":pid,"experiment_id":experiment_id,"artifact_root":args.root})
    );
    Ok(())
}

fn acquire_launch_lock(root: &Path) -> CliResult<LaunchLock> {
    let path = root.join(".launch.lock");
    for _ in 0..2 {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                if let Err(error) =
                    writeln!(file, "{}", std::process::id()).and_then(|()| file.sync_all())
                {
                    drop(file);
                    let _ = fs::remove_file(&path);
                    return Err(format!("write detached launch lock: {error}"));
                }
                return Ok(LaunchLock(path));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(&path)
                    .map_err(|error| format!("launch lock metadata: {error}"))?;
                if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 32 {
                    return Err("invalid detached launch lock".into());
                }
                let pid = fs::read_to_string(&path)
                    .map_err(|error| format!("read launch lock: {error}"))?
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| "invalid detached launch PID")?;
                if Path::new("/proc").join(pid.to_string()).exists() {
                    return Err("another detached launch is in progress".into());
                }
                fs::remove_file(&path)
                    .map_err(|error| format!("remove stale launch lock: {error}"))?;
            }
            Err(error) => return Err(format!("create detached launch lock: {error}")),
        }
    }
    Err("could not acquire detached launch lock".into())
}

fn check_root_state(root: &Path, prepared: &Prepared, command: CommandKind) -> CliResult<()> {
    if command == CommandKind::Validate {
        if sidecar_exists(root)? {
            verify_sidecar(root, prepared)?;
        }
        return Ok(());
    }
    if command == CommandKind::Run {
        for entry in fs::read_dir(root).map_err(|error| format!("artifact root: {error}"))? {
            let entry = entry.map_err(|error| format!("artifact root entry: {error}"))?;
            if env::var_os("XYNOOS_TRAIN_DETACHED_CHILD").is_some()
                && matches!(
                    entry.file_name().to_str(),
                    Some("process.stderr" | PROCESS_PID | ".launch.lock")
                )
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

fn register_process_pid(root: &Path) -> CliResult<()> {
    validate_root(root).map_err(|error| error.to_string())?;
    let path = root.join(PROCESS_PID);
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            let previous = read_process_pid(root)?;
            if Path::new("/proc").join(previous.to_string()).exists() {
                return Err("another detached process is already registered".into());
            }
            fs::remove_file(&path)
                .map_err(|error| format!("remove stale process identity: {error}"))?;
        }
        Ok(_) => return Err("process identity path is not a regular file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("process identity metadata: {error}")),
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("create process identity: {error}"))?;
    if let Err(error) = writeln!(file, "{}", std::process::id()).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(&path);
        return Err(format!("write process identity: {error}"));
    }
    Ok(())
}

fn read_process_pid(root: &Path) -> CliResult<u32> {
    let path = root.join(PROCESS_PID);
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("process identity metadata: {error}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 32 {
        return Err("invalid process identity file".into());
    }
    fs::read_to_string(path)
        .map_err(|error| format!("read process identity: {error}"))?
        .trim()
        .parse()
        .map_err(|_| "invalid process identity PID".into())
}

fn remove_process_pid(root: &Path) -> CliResult<()> {
    let path = root.join(PROCESS_PID);
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            if read_process_pid(root)? == std::process::id() {
                fs::remove_file(path)
                    .map_err(|error| format!("remove process identity: {error}"))?;
            }
            Ok(())
        }
        Ok(_) => Err("process identity path is not a regular file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("process identity metadata: {error}")),
    }
}

fn validation_json(prepared: &Prepared, root: &Path) -> Value {
    json!({
        "status":"valid",
        "experiment_id":prepared.experiment_id,
        "source_revision":prepared.source_revision,
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
        "source_kind":"manifest-v1-stream",
        "batch":{"B":1,"T":MODEL_T},
    })
}

fn write_sidecar(root: &Path, prepared: &Prepared) -> CliResult<()> {
    let path = root.join(SIDECAR);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("experiment identity publication: {error}"))?;
    let value = json!({
        "schema_version":1,
        "experiment_id":prepared.experiment_id,
        "source_revision":prepared.source_revision,
        "dataset_id":prepared.dataset_id,
        "configuration_file_sha256":hex(&prepared.config_sha256),
        "run_config_sha256":hex(&prepared.config.digest(&prepared.identity)),
    });
    serde_json::to_writer(&mut file, &value)
        .map_err(|error| format!("experiment identity write: {error}"))?;
    file.write_all(b"\n")
        .map_err(|error| format!("experiment identity write: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("experiment identity sync: {error}"))?;
    File::open(root)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("experiment identity directory sync: {error}"))?;
    Ok(())
}

fn verify_sidecar(root: &Path, prepared: &Prepared) -> CliResult<()> {
    let value = read_sidecar(root)?;
    if value["schema_version"] != 1
        || value["experiment_id"] != prepared.experiment_id
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
    let path = root.join(".run.lock");
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("run lock metadata: {error}")),
    };
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 32 {
        return Err("invalid run lock".into());
    }
    let pid = fs::read_to_string(&path)
        .map_err(|error| format!("read run lock: {error}"))?
        .trim()
        .parse::<u32>()
        .map_err(|_| "invalid run lock PID")?;
    Ok(Path::new("/proc").join(pid.to_string()).exists())
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
