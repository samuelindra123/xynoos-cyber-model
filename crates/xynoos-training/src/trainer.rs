use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use xynoos_autograd::{
    AdamW, CandidateTrainingPolicy, GradientClipMetrics, ResourceLimits,
    WeightedGradientAccumulator,
};
use xynoos_data::Batch;
use xynoos_tokenizer::Tokenizer;
use xynoos_transformer::{CandidateTrainingBatch, Config, Transformer};

use crate::{
    BatchSource, CheckpointCodec, CheckpointCounters, CheckpointSnapshot, MemoryPhase, Result,
    RunHandoff, RunIdentity, RunJournal, StopToken, TrainingConfig, TrainingError,
    TrainingRunBudget,
    checkpoint::next_batch_identity,
    process_vm_hwm_bytes,
    runtime::{RunLogWriter, hex, validate_root},
};

// Cumulative-work accounting uses one work unit per maximum checkpoint byte.
// This is a conservative, deterministic I/O allowance under the 16 MiB codec cap.
const CHECKPOINT_IO_WORK: u64 = (16 * 1024 * 1024) as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Requested,
    AttemptRuntimeLimit,
    LineageRuntimeLimit,
    ConsumedTokenLimit,
    SupervisedTokenLimit,
    CumulativeWorkLimit,
    UpdateLimit,
    ScheduleComplete,
    InputExhausted,
    NumericalFailure,
    Failure,
}

impl StopReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Requested => "requested",
            Self::AttemptRuntimeLimit => "attempt_runtime_limit",
            Self::LineageRuntimeLimit => "lineage_runtime_limit",
            Self::ConsumedTokenLimit => "consumed_token_limit",
            Self::SupervisedTokenLimit => "supervised_token_limit",
            Self::CumulativeWorkLimit => "cumulative_work_limit",
            Self::UpdateLimit => "update_limit",
            Self::ScheduleComplete => "schedule_complete",
            Self::InputExhausted => "input_exhausted",
            Self::NumericalFailure => "numerical_failure",
            Self::Failure => "failure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSummary {
    pub samples: u64,
    pub mean_ms: f64,
    pub sample_stddev_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl TimeSummary {
    fn json(self) -> Value {
        json!({
            "samples": self.samples,
            "mean_ms": self.mean_ms,
            "sample_stddev_ms": self.sample_stddev_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Samples {
    count: u64,
    mean_ns: f64,
    m2_ns: f64,
    min_ns: u64,
    max_ns: u64,
}

impl Samples {
    fn push(&mut self, value: Duration) {
        let ns = value.as_nanos().min(u64::MAX as u128) as u64;
        self.count += 1;
        self.min_ns = if self.count == 1 {
            ns
        } else {
            self.min_ns.min(ns)
        };
        self.max_ns = self.max_ns.max(ns);
        let delta = ns as f64 - self.mean_ns;
        self.mean_ns += delta / self.count as f64;
        self.m2_ns += delta * (ns as f64 - self.mean_ns);
    }

    fn summary(self) -> TimeSummary {
        TimeSummary {
            samples: self.count,
            mean_ms: self.mean_ns / 1_000_000.0,
            sample_stddev_ms: if self.count < 2 {
                0.0
            } else {
                (self.m2_ns / (self.count - 1) as f64).sqrt() / 1_000_000.0
            },
            min_ms: self.min_ns as f64 / 1_000_000.0,
            max_ms: self.max_ns as f64 / 1_000_000.0,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Timings {
    initialization: Samples,
    dataset_preparation: Samples,
    forward: Samples,
    backward: Samples,
    gradient_transfer: Samples,
    weighted_accumulation: Samples,
    normalize_clip: Samples,
    optimizer_commit: Samples,
    full_update: Samples,
    validation: Samples,
    checkpoint_write: Samples,
    checkpoint_load: Samples,
    shutdown: Samples,
}

impl Timings {
    fn json(self) -> Value {
        json!({
            "initialization": self.initialization.summary().json(),
            "dataset_preparation": self.dataset_preparation.summary().json(),
            "forward": self.forward.summary().json(),
            "backward": self.backward.summary().json(),
            "gradient_transfer": self.gradient_transfer.summary().json(),
            "weighted_accumulation": self.weighted_accumulation.summary().json(),
            "normalization_and_clipping": self.normalize_clip.summary().json(),
            "adamw_staging_and_commit": self.optimizer_commit.summary().json(),
            "full_optimizer_step": self.full_update.summary().json(),
            "validation": self.validation.summary().json(),
            "checkpoint_write": self.checkpoint_write.summary().json(),
            "checkpoint_load": self.checkpoint_load.summary().json(),
            "shutdown": self.shutdown.summary().json(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub experiment_id: String,
    pub stop_reason: StopReason,
    pub counters: CheckpointCounters,
    pub last_loss: Option<f64>,
    pub last_pre_clip_norm: Option<f64>,
    pub last_post_clip_norm: Option<f64>,
    pub last_update_norm: Option<f64>,
    pub supervised_tokens_per_second: f64,
    pub owned_memory_peak_bytes: usize,
    pub charged_owned_current_bytes: usize,
    pub autograd_ledger_peak_bytes: usize,
    pub autograd_ledger_current_bytes: usize,
    pub process_vm_hwm_bytes: Option<u64>,
    pub timings: Value,
}

impl TrainingReport {
    fn json(&self, repository_id: &str, source_sha: &str, config_digest: [u8; 32]) -> Value {
        json!({
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "repository_id": repository_id,
            "source_sha": source_sha,
            "config_sha256": hex(&config_digest),
            "model_id": crate::MODEL_ID,
            "stop_reason": self.stop_reason.as_str(),
            "counters": counter_json(self.counters),
            "last_loss": self.last_loss,
            "last_pre_clip_gradient_norm": self.last_pre_clip_norm,
            "last_post_clip_gradient_norm": self.last_post_clip_norm,
            "last_parameter_update_norm": self.last_update_norm,
            "supervised_tokens_per_second": self.supervised_tokens_per_second,
            "memory": {
                "charged_owned_peak_bytes": self.owned_memory_peak_bytes,
                "charged_owned_current_bytes": self.charged_owned_current_bytes,
                "autograd_ledger_peak_bytes": self.autograd_ledger_peak_bytes,
                "autograd_ledger_current_bytes": self.autograd_ledger_current_bytes,
                "process_vm_hwm_bytes": self.process_vm_hwm_bytes,
            },
            "timings": self.timings,
        })
    }
}

#[derive(Debug, Default)]
struct Window {
    microbatches: usize,
    loss_weighted_sum: f64,
    loss_supervised: u64,
}

pub struct Trainer<S: BatchSource> {
    root: PathBuf,
    identity: RunIdentity,
    config: TrainingConfig,
    source: S,
    model: Transformer,
    optimizer: AdamW,
    accumulator: Option<WeightedGradientAccumulator>,
    policy: CandidateTrainingPolicy,
    journal: RunJournal,
    counters: CheckpointCounters,
    run_budget: TrainingRunBudget,
    ledger_persistent_baseline: usize,
    stop: StopToken,
    log: RunLogWriter<Box<dyn Write>>,
    timings: Timings,
    pending_consumed: u64,
    pending_supervised: u64,
    last_loss: Option<f64>,
    last_clip: Option<GradientClipMetrics>,
    last_update_norm: Option<f64>,
    attempt_started: Option<Instant>,
    attempt_base_runtime_ns: u64,
    experiment_id: String,
    finalized: bool,
}

impl<S: BatchSource> Trainer<S> {
    pub fn new(
        root: &Path,
        identity: RunIdentity,
        config: TrainingConfig,
        tokenizer: &Tokenizer,
        source: S,
        stop: StopToken,
    ) -> Result<Self> {
        Self::construct(root, identity, config, tokenizer, source, stop, None)
    }

    pub fn resume(
        root: &Path,
        identity: RunIdentity,
        config: TrainingConfig,
        tokenizer: &Tokenizer,
        source: S,
        stop: StopToken,
    ) -> Result<Self> {
        Self::construct(root, identity, config, tokenizer, source, stop, Some(()))
    }

    fn construct(
        root: &Path,
        identity: RunIdentity,
        config: TrainingConfig,
        tokenizer: &Tokenizer,
        mut source: S,
        stop: StopToken,
        resume: Option<()>,
    ) -> Result<Self> {
        let started = Instant::now();
        validate_root(root)?;
        config.validate_static()?;
        if source.split() != xynoos_data::Split::Train
            || source.order_identity() != identity.data_order_identity()
        {
            return Err(TrainingError::InvalidIdentity);
        }
        validate_data_memory(&source, tokenizer, config)?;
        identity.validate(tokenizer)?;
        let checkpoint_path = root.join("checkpoint.xyckpt");
        let existing = fs::symlink_metadata(&checkpoint_path);
        match (resume.is_some(), existing) {
            (false, Ok(metadata)) if metadata.is_file() && !metadata.file_type().is_symlink() => {
                return Err(TrainingError::Checkpoint("checkpoint_exists_use_resume"));
            }
            (false, Ok(_)) => return Err(TrainingError::InvalidConfiguration),
            (true, Err(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(TrainingError::Checkpoint("resume_checkpoint_missing"));
            }
            (true, Ok(metadata)) if !metadata.is_file() || metadata.file_type().is_symlink() => {
                return Err(TrainingError::InvalidConfiguration);
            }
            (_, Err(error)) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(error.into());
            }
            _ => {}
        }

        let digest = config.digest(&identity);
        let mut journal = if resume.is_some() {
            RunJournal::open_existing(root, digest, CheckpointCounters::default(), config.limits())?
        } else {
            RunJournal::open(root, digest, CheckpointCounters::default(), config.limits())?
        };
        cleanup_temporary_artifacts(root)?;
        if resume.is_none() {
            let progress = journal.progress();
            if progress.attempted_consumed_tokens != 0
                || progress.attempted_supervised_tokens != 0
                || progress.cumulative_work != 0
                || progress.active_runtime_ns != 0
                || progress.attempted_updates != 0
                || progress.checkpoint_sequence != 0
            {
                return Err(TrainingError::Checkpoint("lineage_exists_use_resume"));
            }
        }
        journal.begin_attempt()?;
        let attempt_base_runtime_ns = journal.progress().active_runtime_ns;
        let attempt_started = Instant::now();
        let run_budget = TrainingRunBudget::new(config.memory())?;
        let mut timings = Timings::default();
        let (model, optimizer, counters, checkpoint_load_time) = if let Some(()) = resume {
            let _reservation = run_budget.reserve_phase(MemoryPhase::Resume)?;
            let checkpoint_metadata = fs::symlink_metadata(&checkpoint_path)?;
            if !checkpoint_metadata.is_file()
                || checkpoint_metadata.file_type().is_symlink()
                || checkpoint_metadata.len() == 0
                || checkpoint_metadata.len() > CHECKPOINT_IO_WORK
            {
                return Err(TrainingError::Checkpoint("invalid_file_size_or_type"));
            }
            journal.precharge_auxiliary_work(
                checkpoint_metadata
                    .len()
                    .checked_mul(2)
                    .ok_or(TrainingError::ArithmeticOverflow)?,
            )?;
            let load_start = Instant::now();
            let checkpoint = CheckpointCodec::load(root, &identity, config, tokenizer)?;
            if source.order_identity() != checkpoint.identity.data_order_identity()
                || checkpoint.next_batch_identity
                    != next_batch_identity(&source.order_identity(), checkpoint.cursor)
            {
                return Err(TrainingError::Checkpoint("next_batch_identity_mismatch"));
            }
            source.seek_to(checkpoint.cursor)?;
            journal.reconcile_checkpoint(checkpoint.counters)?;
            let mut model = Transformer::new(
                Config::RESEARCH_CANDIDATE,
                tokenizer,
                config.seed(),
                config.autograd_limits(),
            )?;
            model.restore_checkpoint_values(&checkpoint.parameters, checkpoint.optimizer_step)?;
            let optimizer = AdamW::restore(
                model.parameters(),
                config.optimizer(),
                checkpoint.optimizer_step,
                checkpoint.beta1_power,
                checkpoint.beta2_power,
                &checkpoint.first_moments,
                &checkpoint.second_moments,
            )?;
            (model, optimizer, checkpoint.counters, load_start.elapsed())
        } else {
            if source.cursor() != 0 {
                return Err(TrainingError::InvalidConfiguration);
            }
            let _reservation = run_budget.reserve_phase(MemoryPhase::Initialization)?;
            let model = Transformer::new(
                Config::RESEARCH_CANDIDATE,
                tokenizer,
                config.seed(),
                config.autograd_limits(),
            )?;
            let optimizer = AdamW::new(model.parameters(), config.optimizer())?;
            (
                model,
                optimizer,
                CheckpointCounters::default(),
                Duration::ZERO,
            )
        };
        timings.initialization.push(started.elapsed());
        if checkpoint_load_time != Duration::ZERO {
            timings.checkpoint_load.push(checkpoint_load_time);
        }
        let progress = journal.progress();
        let policy_limits = config.limits().candidate_policy()?;
        let prior_runtime = Duration::from_nanos(
            progress.active_runtime_ns.min(
                config
                    .limits()
                    .max_runtime()
                    .as_nanos()
                    .min(u64::MAX as u128) as u64,
            ),
        );
        let policy = CandidateTrainingPolicy::restore(
            policy_limits,
            progress.attempted_consumed_tokens,
            progress.attempted_supervised_tokens,
            progress.cumulative_work,
            optimizer.step_count(),
            progress.attempted_updates,
            prior_runtime,
        )?;
        let accumulator = WeightedGradientAccumulator::new(
            model.parameters(),
            config.limits().microbatches_per_update(),
        )?;
        let ledger_persistent_baseline = model.parameters().accounted_bytes();
        let mut counters = counters;
        counters.attempted_consumed_tokens = progress.attempted_consumed_tokens;
        counters.attempted_supervised_tokens = progress.attempted_supervised_tokens;
        counters.cumulative_work = progress.cumulative_work;
        counters.cumulative_runtime_ns = progress.active_runtime_ns;
        counters.attempted_updates = progress.attempted_updates;
        counters.checkpoint_sequence = progress.checkpoint_sequence;
        counters.completed_updates = optimizer.step_count();
        counters.discarded_consumed_tokens = progress
            .attempted_consumed_tokens
            .checked_sub(counters.committed_consumed_tokens)
            .ok_or(TrainingError::Checkpoint(
                "committed_tokens_exceed_attempts",
            ))?;
        if resume.is_none() {
            counters.committed_cursor = 0;
        }
        let metrics_path = root.join("metrics.jsonl");
        let metrics_file = open_append_file(&metrics_path)?;
        let writer: Box<dyn Write> = Box::new(BufWriter::new(metrics_file));
        let mut log = RunLogWriter::new(writer, &identity, config);
        let experiment_id = crate::config::experiment_id(&identity, config);
        RunHandoff::for_run(
            &identity,
            config,
            "checkpoint.xyckpt".into(),
            "metrics.jsonl".into(),
        )
        .write_atomic(root)?;
        log.write_event(
            if resume.is_some() { "resume" } else { "start" },
            json!({
                "completed_updates": counters.completed_updates,
                "committed_cursor": counters.committed_cursor,
                "attempted_consumed_tokens": counters.attempted_consumed_tokens,
                "cumulative_work": counters.cumulative_work,
                "owned_memory_cap_bytes": config.limits().owned_memory_bytes(),
            }),
        )
        .map_err(|_| TrainingError::Logging("start_or_resume"))?;
        Ok(Self {
            root: root.to_path_buf(),
            identity,
            config,
            source,
            model,
            optimizer,
            accumulator: Some(accumulator),
            policy,
            journal,
            counters,
            run_budget,
            ledger_persistent_baseline,
            stop,
            log,
            timings,
            pending_consumed: 0,
            pending_supervised: 0,
            last_loss: None,
            last_clip: None,
            last_update_norm: None,
            attempt_started: Some(attempt_started),
            attempt_base_runtime_ns,
            experiment_id,
            finalized: false,
        })
    }

    pub fn model(&self) -> &Transformer {
        &self.model
    }

    pub fn optimizer(&self) -> &AdamW {
        &self.optimizer
    }

    pub fn counters(&self) -> CheckpointCounters {
        self.counters
    }

    pub fn run(&mut self, validation: Option<&mut dyn BatchSource>) -> Result<TrainingReport> {
        if self.finalized || self.attempt_started.is_none() {
            return Err(TrainingError::InvalidConfiguration);
        }
        let mut primary_error = None;
        let final_checkpoint_reserved = match self.charge_auxiliary_work(CHECKPOINT_IO_WORK) {
            Ok(()) => true,
            Err(error) => {
                primary_error = Some(error);
                false
            }
        };
        let reason = if final_checkpoint_reserved {
            match self.run_updates(validation) {
                Ok(reason) => reason,
                Err(error) => {
                    let failed_reason = if is_numerical_failure(&error) {
                        StopReason::NumericalFailure
                    } else {
                        StopReason::Failure
                    };
                    primary_error = Some(error);
                    failed_reason
                }
            }
        } else {
            StopReason::Failure
        };

        if self.pending_consumed != 0
            && let Err(error) = self.discard_pending()
        {
            primary_error.get_or_insert(error);
        }
        let _ = self.source.seek_to(self.counters.committed_cursor);
        if self.attempt_started.is_some() {
            if let Err(error) = self.write_checkpoint(final_checkpoint_reserved) {
                primary_error.get_or_insert(error);
            }
            match self.journal.finish_attempt() {
                Ok(active) => self.counters.cumulative_runtime_ns = active,
                Err(error) => {
                    primary_error.get_or_insert(error);
                }
            }
            self.attempt_started = None;
        } else {
            self.counters.cumulative_runtime_ns = self.journal.progress().active_runtime_ns;
        }
        self.counters.attempted_consumed_tokens = self.journal.progress().attempted_consumed_tokens;
        self.counters.attempted_supervised_tokens =
            self.journal.progress().attempted_supervised_tokens;
        self.counters.cumulative_work = self.journal.progress().cumulative_work;
        self.counters.attempted_updates = self.journal.progress().attempted_updates;
        self.counters.checkpoint_sequence = self.journal.progress().checkpoint_sequence;
        if let Some(error) = primary_error.as_ref().map(|error| format!("{error:?}"))
            && let Err(logging_error) = self.log.write_event(
                "failure",
                json!({
                    "error": error,
                    "numerical_failure": primary_error.as_ref().is_some_and(is_numerical_failure),
                    "completed_updates": self.counters.completed_updates,
                }),
            )
        {
            primary_error.get_or_insert(logging_error);
        }
        if let Err(error) = self.log.write_event(
            "stop",
            json!({
                "reason": reason.as_str(),
                "completed_updates": self.counters.completed_updates,
                "attempted_consumed_tokens": self.counters.attempted_consumed_tokens,
                "committed_consumed_tokens": self.counters.committed_consumed_tokens,
            }),
        ) {
            primary_error.get_or_insert(error);
        }
        let report_path = self.root.join("final-report.json");
        if self.log.flush().is_err() {
            primary_error.get_or_insert(TrainingError::Logging("final_flush"));
        }
        let first_report = self.build_report(reason);
        let mut report_json = first_report.json(
            self.identity.source_repository_id(),
            self.identity.source_revision(),
            self.config.digest(&self.identity),
        );
        if let Some(error) = &primary_error {
            report_json["error"] = json!(format!("{error:?}"));
        }
        let shutdown_start = Instant::now();
        if let Err(_error) = write_json_atomic(&self.root, &report_path, &report_json) {
            primary_error.get_or_insert(TrainingError::RuntimeArtifact("final-report"));
        }
        self.timings.shutdown.push(shutdown_start.elapsed());
        let report = self.build_report(reason);
        let mut final_json = report.json(
            self.identity.source_repository_id(),
            self.identity.source_revision(),
            self.config.digest(&self.identity),
        );
        if let Some(error) = &primary_error {
            final_json["error"] = json!(format!("{error:?}"));
        }
        if let Err(_error) = write_json_atomic(&self.root, &report_path, &final_json) {
            primary_error.get_or_insert(TrainingError::RuntimeArtifact("final-report"));
        }
        if let Err(error) = cleanup_temporary_artifacts(&self.root) {
            primary_error.get_or_insert(error);
        }
        self.finalized = true;
        match primary_error {
            Some(error) => Err(error),
            None => Ok(report),
        }
    }

    fn run_updates(&mut self, mut validation: Option<&mut dyn BatchSource>) -> Result<StopReason> {
        if let Some(source) = validation.as_deref() {
            if source.split() != xynoos_data::Split::Validation
                || source.order_identity() != self.identity.validation_order_identity()
            {
                return Err(TrainingError::InvalidIdentity);
            }
            validate_data_source_memory(source, self.config)?;
        }
        loop {
            if let Some(reason) = self.safe_stop()? {
                return Ok(reason);
            }
            let mut window = Window::default();
            let window_start = Instant::now();
            let max_microbatches = self.config.limits().microbatches_per_update();
            while window.microbatches < max_microbatches {
                if let Some(reason) = self.safe_stop()? {
                    return Ok(reason);
                }
                const BATCH_READER_WORK: u64 = 32;
                if let Err(error) = self.journal.precharge_auxiliary_work(BATCH_READER_WORK) {
                    if let Some(reason) = budget_stop_reason(&error) {
                        return Ok(reason);
                    }
                    return Err(error);
                }
                if let Err(error) = self
                    .policy
                    .charge_auxiliary_work(self.model.parameters(), BATCH_READER_WORK)
                {
                    if matches!(
                        error,
                        xynoos_autograd::AutogradError::TrainingPolicyExhausted
                    ) {
                        return Ok(StopReason::CumulativeWorkLimit);
                    }
                    return Err(error.into());
                }
                let data_start = Instant::now();
                let Some(batch) = self.source.next_batch()? else {
                    self.timings.dataset_preparation.push(data_start.elapsed());
                    return Ok(StopReason::InputExhausted);
                };
                self.timings.dataset_preparation.push(data_start.elapsed());
                let (consumed, supervised) = count_batch(&batch)?;
                let _backward_reservation = if supervised == 0 {
                    None
                } else {
                    Some(self.run_budget.reserve_phase(MemoryPhase::Backward)?)
                };
                let micro_work = if supervised == 0 {
                    0
                } else {
                    (ResourceLimits::MAX_WORK as u64)
                        .checked_add(160_000_000)
                        .and_then(|work| work.checked_add(74 * 4))
                        .ok_or(TrainingError::ArithmeticOverflow)?
                };
                if let Err(error) = self
                    .journal
                    .precharge_microbatch(consumed, supervised, micro_work)
                {
                    if let Some(reason) = budget_stop_reason(&error) {
                        return Ok(reason);
                    }
                    return Err(error);
                }
                self.pending_consumed = self
                    .pending_consumed
                    .checked_add(consumed)
                    .ok_or(TrainingError::ArithmeticOverflow)?;
                self.pending_supervised = self
                    .pending_supervised
                    .checked_add(supervised)
                    .ok_or(TrainingError::ArithmeticOverflow)?;
                window.microbatches += 1;
                let forward_start = Instant::now();
                let evaluated = self
                    .model
                    .evaluate_candidate_training(&batch, &mut self.policy);
                self.timings.forward.push(forward_start.elapsed());
                let evaluated = match evaluated {
                    Ok(value) => value,
                    Err(error) => {
                        if matches!(
                            error,
                            xynoos_transformer::Error::Autograd(
                                xynoos_autograd::AutogradError::TrainingPolicyExhausted
                            )
                        ) {
                            if let Some(reason) = self.safe_stop()? {
                                return Ok(reason);
                            }
                            return Ok(StopReason::CumulativeWorkLimit);
                        }
                        return Err(error.into());
                    }
                };
                match evaluated {
                    CandidateTrainingBatch::Skipped { .. } => {
                        let accumulation_start = Instant::now();
                        self.accumulator
                            .as_mut()
                            .ok_or(TrainingError::InvalidConfiguration)?
                            .add_empty(self.model.parameters())?;
                        self.timings
                            .weighted_accumulation
                            .push(accumulation_start.elapsed());
                    }
                    CandidateTrainingBatch::Evaluated {
                        evaluation,
                        supervised_tokens,
                        ..
                    } => {
                        let loss = f64::from(evaluation.loss());
                        let forward_metrics = evaluation.metrics()?;
                        let backward_start = Instant::now();
                        let reverse = evaluation.backward()?;
                        self.timings.backward.push(backward_start.elapsed());
                        let transfer_start = Instant::now();
                        let (gradients, tape_metrics) = reverse.into_gradients()?;
                        self.timings
                            .gradient_transfer
                            .push(transfer_start.elapsed());
                        let accumulation_start = Instant::now();
                        self.accumulator
                            .as_mut()
                            .ok_or(TrainingError::InvalidConfiguration)?
                            .add(self.model.parameters(), gradients, supervised_tokens)?;
                        self.timings
                            .weighted_accumulation
                            .push(accumulation_start.elapsed());
                        window.loss_weighted_sum += loss * supervised_tokens as f64;
                        window.loss_supervised = window
                            .loss_supervised
                            .checked_add(supervised_tokens)
                            .ok_or(TrainingError::ArithmeticOverflow)?;
                        self.log.write_event(
                            "microbatch",
                            json!({
                                "loss": loss,
                                "consumed_tokens": consumed,
                                "supervised_tokens": supervised_tokens,
                            "attempted_consumed_tokens": self.journal.progress().attempted_consumed_tokens,
                            "committed_consumed_tokens": self.counters.committed_consumed_tokens,
                                "forward_work": forward_metrics.forward_work,
                                "backward_work": tape_metrics.backward_work,
                                "gradient_transfer_work": tape_metrics.gradient_transfer_work,
                            }),
                        ).map_err(|_| TrainingError::Logging("microbatch"))?;
                    }
                }
                drop(_backward_reservation);
            }

            if self
                .accumulator
                .as_ref()
                .ok_or(TrainingError::InvalidConfiguration)?
                .total_supervised_tokens()
                == 0
            {
                self.policy.discard_window();
                self.accumulator = Some(WeightedGradientAccumulator::new(
                    self.model.parameters(),
                    max_microbatches,
                )?);
                if self.model.parameters().accounted_bytes() != self.ledger_persistent_baseline {
                    return Err(TrainingError::InvalidConfiguration);
                }
                self.log
                    .write_event(
                        "empty_window",
                        json!({
                            "consumed_tokens": self.pending_consumed,
                            "committed_cursor": self.counters.committed_cursor,
                        }),
                    )
                    .map_err(|_| TrainingError::Logging("empty_window"))?;
                continue;
            }

            let update_work = (914_400_u64 + 74)
                .checked_mul(30)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            if let Err(error) = self.journal.precharge_update(update_work) {
                if let Some(reason) = budget_stop_reason(&error) {
                    return Ok(reason);
                }
                return Err(error);
            }
            if let Err(error) = self.policy.authorize_update(self.model.parameters()) {
                if matches!(
                    error,
                    xynoos_autograd::AutogradError::TrainingPolicyExhausted
                ) {
                    if let Some(reason) = self.safe_stop()? {
                        return Ok(reason);
                    }
                    return Ok(StopReason::CumulativeWorkLimit);
                }
                return Err(error.into());
            }
            let next_committed_consumed = self
                .counters
                .committed_consumed_tokens
                .checked_add(self.pending_consumed)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            let next_committed_supervised = self
                .counters
                .committed_supervised_tokens
                .checked_add(self.pending_supervised)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            let next_metric_sequence = self
                .counters
                .last_metric_sequence
                .checked_add(1)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            let _update_reservation = self.run_budget.reserve_phase(MemoryPhase::Update)?;
            let full_update_start = window_start;
            let normalization_start = Instant::now();
            let accumulator = self
                .accumulator
                .take()
                .ok_or(TrainingError::InvalidConfiguration)?;
            let (gradient_batch, clip_metrics) =
                accumulator.finish(self.model.parameters(), self.config.gradient_clip())?;
            self.timings
                .normalize_clip
                .push(normalization_start.elapsed());
            self.model.accumulate_gradients_with_work(gradient_batch)?;
            let learning_rate = self
                .config
                .schedule()
                .next_rate(self.optimizer.step_count())?;
            let optimizer_start = Instant::now();
            let update_metrics = match self
                .model
                .optimizer_step(&mut self.optimizer, learning_rate)
            {
                Ok(metrics) => metrics,
                Err(error) => {
                    self.model.clear_gradients()?;
                    self.policy.discard_window();
                    return Err(error.into());
                }
            };
            drop(_update_reservation);
            if self.run_budget.charged_bytes() != self.run_budget.persistent_baseline_bytes() {
                return Err(TrainingError::InvalidConfiguration);
            }
            self.timings
                .optimizer_commit
                .push(optimizer_start.elapsed());
            self.timings.full_update.push(full_update_start.elapsed());
            self.counters.completed_updates = self.optimizer.step_count();
            self.counters.committed_cursor = self.source.cursor();
            self.counters.committed_consumed_tokens = next_committed_consumed;
            self.counters.committed_supervised_tokens = next_committed_supervised;
            self.pending_consumed = 0;
            self.pending_supervised = 0;
            self.counters.last_metric_sequence = next_metric_sequence;
            if let Err(error) = self.policy.commit_update() {
                return Err(error.into());
            }
            self.last_loss = if window.loss_supervised == 0 {
                None
            } else {
                Some(window.loss_weighted_sum / window.loss_supervised as f64)
            };
            self.last_clip = Some(clip_metrics);
            self.last_update_norm = Some(update_metrics.parameter_update_norm);
            self.accumulator = Some(WeightedGradientAccumulator::new(
                self.model.parameters(),
                max_microbatches,
            )?);
            if self.model.parameters().accounted_bytes() != self.ledger_persistent_baseline {
                return Err(TrainingError::InvalidConfiguration);
            }
            self.log.write_event(
                "optimizer_update",
                json!({
                    "completed_updates": self.counters.completed_updates,
                    "attempted_consumed_tokens": self.journal.progress().attempted_consumed_tokens,
                    "attempted_supervised_tokens": self.journal.progress().attempted_supervised_tokens,
                    "cumulative_work": self.journal.progress().cumulative_work,
                    "learning_rate": learning_rate,
                    "loss": self.last_loss,
                    "supervised_tokens": self.counters.committed_supervised_tokens,
                    "gradient_norm_before_clip": clip_metrics.pre_clip_norm,
                    "gradient_norm_after_clip": clip_metrics.post_clip_norm,
                    "gradient_clipped": clip_metrics.clipped,
                    "parameter_update_norm": update_metrics.parameter_update_norm,
                    "window_microbatches": window.microbatches,
                }),
            )
            .map_err(|_| TrainingError::Logging("optimizer_update"))?;

            if self
                .counters
                .completed_updates
                .is_multiple_of(self.config.limits().checkpoint_every_updates())
                && let Err(error) = self.write_checkpoint(false)
            {
                if let Some(reason) = budget_stop_reason(&error) {
                    return Ok(reason);
                }
                return Err(error);
            }
            if let Some(source) = validation.as_deref_mut()
                && self
                    .counters
                    .completed_updates
                    .is_multiple_of(self.config.limits().validation_every_updates())
                && let Err(error) = self.validate_once(source)
            {
                if let Some(reason) = budget_stop_reason(&error) {
                    return Ok(reason);
                }
                if matches!(
                    error,
                    TrainingError::Autograd(
                        xynoos_autograd::AutogradError::TrainingPolicyExhausted
                    )
                ) {
                    return Ok(StopReason::CumulativeWorkLimit);
                }
                return Err(error);
            }
            if self.counters.completed_updates >= self.config.schedule().total_updates() {
                return Ok(StopReason::ScheduleComplete);
            }
        }
    }

    fn safe_stop(&self) -> Result<Option<StopReason>> {
        if self.stop.is_requested()? {
            return Ok(Some(StopReason::Requested));
        }
        let elapsed = self
            .attempt_started
            .map_or(Duration::ZERO, |started| started.elapsed());
        if elapsed >= self.config.limits().max_attempt_runtime() {
            return Ok(Some(StopReason::AttemptRuntimeLimit));
        }
        let active = (self.attempt_base_runtime_ns as u128 + elapsed.as_nanos())
            .max(self.journal.progress().active_runtime_ns as u128);
        if active >= self.config.limits().max_runtime().as_nanos() {
            return Ok(Some(StopReason::LineageRuntimeLimit));
        }
        let progress = self.journal.progress();
        if progress.attempted_consumed_tokens >= self.config.limits().consumed_tokens() {
            return Ok(Some(StopReason::ConsumedTokenLimit));
        }
        if progress.attempted_supervised_tokens >= self.config.limits().supervised_tokens() {
            return Ok(Some(StopReason::SupervisedTokenLimit));
        }
        if progress.cumulative_work >= self.config.limits().cumulative_work() {
            return Ok(Some(StopReason::CumulativeWorkLimit));
        }
        if progress.attempted_updates >= self.config.limits().max_updates() {
            return Ok(Some(StopReason::UpdateLimit));
        }
        if self.optimizer.step_count() >= self.config.schedule().total_updates() {
            return Ok(Some(StopReason::ScheduleComplete));
        }
        Ok(None)
    }

    fn validate_once(&mut self, validation: &mut dyn BatchSource) -> Result<()> {
        if validation.split() != xynoos_data::Split::Validation
            || validation.order_identity() != self.identity.validation_order_identity()
        {
            return Err(TrainingError::InvalidIdentity);
        }
        validate_data_source_memory(validation, self.config)?;
        validation.seek_to(0)?;
        let _reservation = self.run_budget.reserve_phase(MemoryPhase::Validation)?;
        let start = Instant::now();
        let model = &self.model;
        let parameters = model.parameters();
        let identity = &self.identity;
        let limits = self.config.validation_limits();
        let report = crate::evaluation::evaluate_batches_with_charge(
            model,
            identity,
            crate::EvaluationStage::PeriodicValidation,
            validation,
            limits,
            |charge| {
                let work = match charge {
                    crate::evaluation::EvaluationCharge::BatchRead => 32,
                    crate::evaluation::EvaluationCharge::ForwardPass => {
                        ResourceLimits::MAX_WORK as u64
                    }
                };
                self.journal.precharge_auxiliary_work(work)?;
                self.policy.charge_auxiliary_work(parameters, work)?;
                Ok(())
            },
        )?;
        self.timings.validation.push(start.elapsed());
        self.counters.validation_supervised_tokens = self
            .counters
            .validation_supervised_tokens
            .checked_add(report.supervised_tokens)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        self.log
            .write_event(
                "validation",
                json!({
                    "loss": report.mean_loss,
                    "perplexity": report.perplexity,
                    "perplexity_overflow": report.perplexity_overflow,
                    "batch_count": report.batch_count,
                    "coverage_complete": report.coverage_complete,
                    "consumed_tokens": report.consumed_tokens,
                    "supervised_tokens": report.supervised_tokens,
                    "validation_supervised_tokens": self.counters.validation_supervised_tokens,
                    "forward_work": report.total_forward_work,
                    "evaluation_config_sha256": report.evaluation_config_sha256,
                    "test_access": report.test_access,
                }),
            )
            .map_err(|_| TrainingError::Logging("validation"))?;
        Ok(())
    }

    fn discard_pending(&mut self) -> Result<()> {
        let discarded = self
            .journal
            .progress()
            .attempted_consumed_tokens
            .checked_sub(self.counters.committed_consumed_tokens)
            .ok_or(TrainingError::Checkpoint(
                "committed_tokens_exceed_attempts",
            ))?;
        self.counters.discarded_consumed_tokens = discarded;
        self.model.clear_gradients()?;
        self.policy.discard_window();
        self.accumulator = None;
        self.pending_consumed = 0;
        self.pending_supervised = 0;
        self.source.seek_to(self.counters.committed_cursor)?;
        Ok(())
    }

    fn write_checkpoint(&mut self, final_checkpoint_reserved: bool) -> Result<()> {
        if !final_checkpoint_reserved {
            self.charge_auxiliary_work(CHECKPOINT_IO_WORK)?;
        }
        self.write_checkpoint_inner()
    }

    fn write_checkpoint_inner(&mut self) -> Result<()> {
        let active = self.journal.sync_runtime()?;
        let progress = self.journal.progress();
        self.counters.attempted_consumed_tokens = progress.attempted_consumed_tokens;
        self.counters.attempted_supervised_tokens = progress.attempted_supervised_tokens;
        self.counters.cumulative_work = progress.cumulative_work;
        self.counters.cumulative_runtime_ns = active;
        self.counters.attempted_updates = progress.attempted_updates;
        self.counters.checkpoint_sequence = self.journal.next_checkpoint_sequence()?;
        let snapshot = CheckpointSnapshot::new(
            &self.identity,
            self.config,
            &self.model,
            &self.optimizer,
            self.counters,
        )?;
        let _reservation = self
            .run_budget
            .reserve_phase(MemoryPhase::CheckpointWrite)?;
        let start = Instant::now();
        let publication = CheckpointCodec::write_atomic(&self.root, &snapshot)?;
        self.timings.checkpoint_write.push(start.elapsed());
        self.log
            .write_event(
                "checkpoint",
                json!({
                    "path": publication.path,
                    "bytes": publication.bytes,
                    "sha256": hex(&publication.sha256),
                    "durable": publication.durable,
                    "sequence": self.counters.checkpoint_sequence,
                }),
            )
            .map_err(|_| TrainingError::Logging("checkpoint"))?;
        Ok(())
    }

    fn charge_auxiliary_work(&mut self, work: u64) -> Result<()> {
        self.journal.precharge_auxiliary_work(work)?;
        self.policy
            .charge_auxiliary_work(self.model.parameters(), work)?;
        Ok(())
    }

    fn build_report(&self, reason: StopReason) -> TrainingReport {
        let active = self.counters.cumulative_runtime_ns as f64 / 1_000_000_000.0;
        TrainingReport {
            experiment_id: self.experiment_id.clone(),
            stop_reason: reason,
            counters: self.counters,
            last_loss: self.last_loss,
            last_pre_clip_norm: self.last_clip.map(|m| m.pre_clip_norm),
            last_post_clip_norm: self.last_clip.map(|m| m.post_clip_norm),
            last_update_norm: self.last_update_norm,
            supervised_tokens_per_second: if active == 0.0 {
                0.0
            } else {
                self.counters.committed_supervised_tokens as f64 / active
            },
            owned_memory_peak_bytes: self.run_budget.peak_charged_bytes(),
            charged_owned_current_bytes: self.run_budget.charged_bytes(),
            autograd_ledger_peak_bytes: self.model.parameters().peak_accounted_bytes(),
            autograd_ledger_current_bytes: self.model.parameters().accounted_bytes(),
            process_vm_hwm_bytes: process_vm_hwm_bytes(),
            timings: self.timings.json(),
        }
    }
}

fn count_batch(batch: &Batch) -> Result<(u64, u64)> {
    if batch.shape() != [1, 32] {
        return Err(TrainingError::InvalidConfiguration);
    }
    let positions = batch.input().len();
    if batch.target().len() != positions
        || batch
            .valid_mask()
            .is_some_and(|mask| mask.len() != positions)
        || batch
            .loss_mask()
            .is_some_and(|mask| mask.len() != positions)
    {
        return Err(TrainingError::InvalidConfiguration);
    }
    let mut consumed = 0_u64;
    let mut supervised = 0_u64;
    for index in 0..positions {
        let valid = batch.valid_mask().is_none_or(|mask| mask[index] == 1);
        let loss = batch.loss_mask().is_none_or(|mask| mask[index] == 1);
        if batch.valid_mask().is_some_and(|mask| mask[index] > 1)
            || batch.loss_mask().is_some_and(|mask| mask[index] > 1)
            || loss && !valid
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        consumed += u64::from(valid);
        supervised += u64::from(loss);
    }
    Ok((consumed, supervised))
}

fn validate_data_memory(
    source: &dyn BatchSource,
    tokenizer: &Tokenizer,
    config: TrainingConfig,
) -> Result<()> {
    let tokenizer_bytes = tokenizer
        .retained_bytes()
        .map_err(|_| TrainingError::InvalidIdentity)?;
    if tokenizer_bytes > config.memory().tokenizer_buffer_bytes() {
        return Err(TrainingError::ResourceLimit {
            requested: tokenizer_bytes,
            limit: config.memory().tokenizer_buffer_bytes(),
        });
    }
    validate_data_source_memory(source, config)
}

fn validate_data_source_memory(source: &dyn BatchSource, config: TrainingConfig) -> Result<()> {
    let plan = config.memory();
    let limit = plan
        .dataset_metadata_bytes()
        .checked_add(plan.dataset_buffer_bytes())
        .and_then(|bytes| bytes.checked_add(plan.reader_buffer_bytes()))
        .ok_or(TrainingError::ArithmeticOverflow)?;
    let requested = source.reserved_memory_bytes()?;
    if requested > limit {
        return Err(TrainingError::ResourceLimit { requested, limit });
    }
    Ok(())
}

fn counter_json(c: CheckpointCounters) -> Value {
    json!({
        "committed_cursor": c.committed_cursor,
        "committed_consumed_tokens": c.committed_consumed_tokens,
        "committed_supervised_tokens": c.committed_supervised_tokens,
        "validation_supervised_tokens": c.validation_supervised_tokens,
        "attempted_consumed_tokens": c.attempted_consumed_tokens,
        "attempted_supervised_tokens": c.attempted_supervised_tokens,
        "discarded_consumed_tokens": c.discarded_consumed_tokens,
        "cumulative_work": c.cumulative_work,
        "cumulative_runtime_ns": c.cumulative_runtime_ns,
        "completed_updates": c.completed_updates,
        "attempted_updates": c.attempted_updates,
        "last_metric_sequence": c.last_metric_sequence,
        "checkpoint_sequence": c.checkpoint_sequence,
    })
}

fn budget_stop_reason(error: &TrainingError) -> Option<StopReason> {
    match error {
        TrainingError::BudgetExhausted("consumed_tokens") => Some(StopReason::ConsumedTokenLimit),
        TrainingError::BudgetExhausted("supervised_tokens") => {
            Some(StopReason::SupervisedTokenLimit)
        }
        TrainingError::BudgetExhausted("cumulative_work") => Some(StopReason::CumulativeWorkLimit),
        TrainingError::BudgetExhausted("optimizer_updates") => Some(StopReason::UpdateLimit),
        _ => None,
    }
}

fn is_numerical_failure(error: &TrainingError) -> bool {
    use xynoos_autograd::AutogradError as A;
    let numerical = |error: &A| {
        matches!(
            error,
            A::NonFinite { .. }
                | A::InvalidOptimizerState
                | A::InvalidBiasPower
                | A::InvalidSecondMoment { .. }
        )
    };
    match error {
        TrainingError::Autograd(error) => numerical(error),
        TrainingError::Transformer(xynoos_transformer::Error::Autograd(error)) => numerical(error),
        _ => false,
    }
}

fn open_append_file(path: &Path) -> Result<File> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
        Ok(_) => return Err(TrainingError::InvalidConfiguration),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(Into::into)
}

fn cleanup_temporary_artifacts(root: &Path) -> Result<()> {
    let mut scanned = 0_usize;
    for entry in fs::read_dir(root)? {
        scanned = scanned
            .checked_add(1)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if scanned > 4096 {
            return Err(TrainingError::ResourceLimit {
                requested: scanned,
                limit: 4096,
            });
        }
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if ![
            ".xyckpt-",
            ".run-journal-",
            ".handoff-",
            ".final-report.json-",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
        {
            continue;
        }
        let metadata = fs::symlink_metadata(entry.path())?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(TrainingError::InvalidConfiguration);
        }
        fs::remove_file(entry.path())?;
    }
    Ok(())
}

fn write_json_atomic(root: &Path, path: &Path, value: &Value) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
        Ok(_) => return Err(TrainingError::InvalidConfiguration),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let name = path
        .file_name()
        .ok_or(TrainingError::InvalidConfiguration)?;
    let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp = root.join(format!(
        ".{}-{}-{serial}.tmp",
        name.to_string_lossy(),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| {
        serde_json::to_writer(&mut file, value).map_err(|_| TrainingError::AllocationFailed)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        File::open(root)?
            .sync_all()
            .map_err(|_| TrainingError::PublicationDurabilityUncertain)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearWarmupDecay, RunLimits, TrainingMemoryPlan, VecBatchSource};
    use sha2::{Digest, Sha256};
    use std::io;
    use xynoos_autograd::{AdamWConfig, ResourceLimits};
    use xynoos_data::Batch;
    use xynoos_tokenizer::{TokenizerArtifact, TokenizerLimits};

    const DESIGN_HEAD: &str = "26777fa95383600c003ff094216d3d1c310b051c";
    const FIXTURE: &str = include_str!("../tests/fixtures/objective_a.json");

    fn root(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "xytrainer-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn digest(value: &str) -> [u8; 32] {
        value
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).unwrap()
    }

    fn tokenizer() -> Tokenizer {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../xynoos-data/tests/fixtures/task004.json"
        ))
        .unwrap();
        TokenizerArtifact::read(
            fixture["artifact"].as_str().unwrap().as_bytes(),
            TokenizerLimits::MAX,
        )
        .unwrap()
    }

    fn identity(tokenizer: &Tokenizer, fixture: &Value) -> RunIdentity {
        let manifest: [u8; 32] = Sha256::digest(FIXTURE.as_bytes()).into();
        let splits = &fixture["split_sha256"];
        RunIdentity::new(
            DESIGN_HEAD.to_owned(),
            true,
            fixture["fixture_id"].as_str().unwrap().to_owned(),
            manifest,
            digest(splits["train"].as_str().unwrap()),
            digest(splits["validation"].as_str().unwrap()),
            digest(splits["test"].as_str().unwrap()),
            "task006-objective-a-v1".to_owned(),
            tokenizer.algorithm().to_owned(),
            digest(fixture["tokenizer_artifact_sha256"].as_str().unwrap()),
            292,
            digest(fixture["data_order_identity_sha256"].as_str().unwrap()),
            digest(
                fixture["validation_order_identity_sha256"]
                    .as_str()
                    .unwrap(),
            ),
        )
    }

    fn config(fixture: &Value, microbatches: usize, total_updates: u64) -> TrainingConfig {
        let limits = RunLimits::new(
            microbatches,
            total_updates,
            4096,
            4096,
            2_000_000_000,
            ResourceLimits::MAX_BYTES,
            Duration::from_secs(600),
            Duration::from_secs(300),
            1,
            1,
        )
        .unwrap();
        TrainingConfig::new(
            AdamWConfig::new(0.9, 0.999, 1e-8, 0.01).unwrap(),
            LinearWarmupDecay::new(
                total_updates,
                0,
                fixture["fixed_integration_peak_rate"].as_f64().unwrap() as f32,
                fixture["fixed_integration_final_rate"].as_f64().unwrap() as f32,
            )
            .unwrap(),
            limits,
            TrainingMemoryPlan::candidate(4 * 1024 * 1024, ResourceLimits::MAX_BYTES).unwrap(),
            ResourceLimits::default(),
            Some(1.0),
            fixture["model_seed"].as_u64().unwrap(),
        )
        .unwrap()
    }

    struct OverBudgetSource {
        order: [u8; 32],
        bytes: usize,
    }

    impl BatchSource for OverBudgetSource {
        fn next_batch(&mut self) -> Result<Option<Batch>> {
            Ok(None)
        }
        fn cursor(&self) -> u64 {
            0
        }
        fn seek_to(&mut self, cursor: u64) -> Result<()> {
            if cursor == 0 {
                Ok(())
            } else {
                Err(TrainingError::InvalidConfiguration)
            }
        }
        fn split(&self) -> xynoos_data::Split {
            xynoos_data::Split::Train
        }
        fn order_identity(&self) -> [u8; 32] {
            self.order
        }
        fn reserved_memory_bytes(&self) -> Result<usize> {
            Ok(self.bytes)
        }
    }

    fn batch(tokenizer: &Tokenizer, text: &str) -> Batch {
        let mut sequence = tokenizer.encode_text(text, TokenizerLimits::MAX).unwrap();
        assert!(!sequence.is_empty());
        sequence.push(xynoos_tokenizer::SpecialToken::Eos.id());
        let source = sequence.clone();
        while sequence.len() < 33 {
            sequence.extend_from_slice(&source);
        }
        sequence.truncate(33);
        assert!(sequence.iter().all(|token| *token < 292));
        let input = &sequence[..32];
        let target = &sequence[1..33];
        Batch::from_parts(1, 32, input, target, Some(&[1; 32]), Some(&[1; 32])).unwrap()
    }

    fn batches(fixture: &Value, split: &str, tokenizer: &Tokenizer) -> Vec<Batch> {
        fixture["split_rows"][split]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| batch(tokenizer, row["text"].as_str().unwrap()))
            .collect()
    }

    fn mean_loss(model: &Transformer, batches: &[Batch]) -> f64 {
        let mut sum = 0.0_f64;
        for batch in batches {
            sum += f64::from(model.evaluate_batch(batch).unwrap().loss());
        }
        sum / batches.len() as f64
    }

    fn copy_batches(batches: &[Batch]) -> Vec<Batch> {
        batches
            .iter()
            .map(|batch| {
                Batch::from_parts(
                    batch.shape()[0],
                    batch.shape()[1],
                    batch.input(),
                    batch.target(),
                    batch.valid_mask(),
                    batch.loss_mask(),
                )
                .unwrap()
            })
            .collect()
    }

    struct FailWriter {
        marker: &'static [u8],
        fail_write: bool,
        failed: bool,
        pending: Vec<u8>,
    }

    impl Write for FailWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.pending.extend_from_slice(bytes);
            if self.fail_write
                && !self.failed
                && self
                    .pending
                    .windows(self.marker.len())
                    .any(|part| part == self.marker)
            {
                self.failed = true;
                return Err(io::Error::other("injected event write failure"));
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            if !self.fail_write
                && !self.failed
                && self
                    .pending
                    .windows(self.marker.len())
                    .any(|part| part == self.marker)
            {
                self.failed = true;
                return Err(io::Error::other("injected event flush failure"));
            }
            self.pending.clear();
            Ok(())
        }
    }

    fn with_failure_writer(
        label: &str,
        marker: &'static [u8],
        fail_write: bool,
        stop: StopToken,
    ) -> (Trainer<VecBatchSource>, PathBuf) {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let config = config(&fixture, 1, 2);
        let source = VecBatchSource::new(
            batches(&fixture, "train", &tokenizer),
            identity.data_order_identity(),
        );
        let root = root(label);
        let mut trainer =
            Trainer::new(&root, identity.clone(), config, &tokenizer, source, stop).unwrap();
        trainer.log = RunLogWriter::new(
            Box::new(FailWriter {
                marker,
                fail_write,
                failed: false,
                pending: Vec::new(),
            }),
            &identity,
            config,
        );
        (trainer, root)
    }

    fn state_bits(
        trainer: &Trainer<VecBatchSource>,
    ) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, u64, u64, u64) {
        let parameters = trainer
            .model()
            .inventory()
            .map(|(_, id)| {
                trainer
                    .model()
                    .parameters()
                    .value(id)
                    .unwrap()
                    .data()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        let first: Vec<Vec<u32>> = (0..74)
            .map(|index| {
                trainer
                    .optimizer()
                    .first_moment(index)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            })
            .collect();
        let second: Vec<Vec<u32>> = (0..74)
            .map(|index| {
                trainer
                    .optimizer()
                    .second_moment(index)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            })
            .collect();
        let step = trainer.optimizer().step_count();
        let q1 = trainer.optimizer().beta1_power().to_bits();
        let q2 = trainer.optimizer().beta2_power().to_bits();
        let mut joined = first;
        joined.extend(second);
        (parameters, joined, step, q1, q2)
    }

    #[test]
    fn masks_exclude_padding_from_consumed_and_supervised_token_counts() {
        let input = [258_u32; 32];
        let target = [258_u32; 32];
        let mut valid = [0_u8; 32];
        let mut loss = [0_u8; 32];
        valid[..4].fill(1);
        loss[..2].fill(1);
        let batch = Batch::from_parts(1, 32, &input, &target, Some(&valid), Some(&loss)).unwrap();
        assert_eq!(count_batch(&batch).unwrap(), (4, 2));

        let empty = [0_u8; 32];
        let padded = Batch::from_parts(1, 32, &input, &target, Some(&empty), Some(&empty)).unwrap();
        assert_eq!(count_batch(&padded).unwrap(), (0, 0));

        let mut invalid_loss = [0_u8; 32];
        invalid_loss[4] = 1;
        let invalid =
            Batch::from_parts(1, 32, &input, &target, Some(&valid), Some(&invalid_loss)).unwrap();
        assert_eq!(
            count_batch(&invalid),
            Err(TrainingError::InvalidConfiguration)
        );
    }

    #[test]
    fn incompatible_source_budget_rejects_before_model_allocation() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let config = config(&fixture, 1, 2);
        let limit = config
            .memory()
            .dataset_metadata_bytes()
            .checked_add(config.memory().dataset_buffer_bytes())
            .and_then(|value| value.checked_add(config.memory().reader_buffer_bytes()))
            .unwrap();
        let source = OverBudgetSource {
            order: identity.data_order_identity(),
            bytes: limit + 1,
        };
        let root = root("source-budget");
        assert!(matches!(
            Trainer::new(&root, identity, config, &tokenizer, source, StopToken::new()),
            Err(TrainingError::ResourceLimit {
                requested,
                limit: expected
            }) if requested == limit + 1 && expected == limit
        ));
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn all_empty_supervision_windows_never_commit_an_optimizer_update() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let config = config(&fixture, 1, 2);
        let input = [0_u32; 32];
        let target = [0_u32; 32];
        let empty = [0_u8; 32];
        let empty_batch =
            || Batch::from_parts(1, 32, &input, &target, Some(&empty), Some(&empty)).unwrap();
        let source = VecBatchSource::new(
            vec![empty_batch(), empty_batch()],
            identity.data_order_identity(),
        );
        let root = root("empty-supervision");
        let mut trainer = Trainer::new(
            &root,
            identity,
            config,
            &tokenizer,
            source,
            StopToken::new(),
        )
        .unwrap();
        let report = trainer.run(None).unwrap();
        assert_eq!(report.stop_reason, StopReason::InputExhausted);
        assert_eq!(report.counters.completed_updates, 0);
        assert_eq!(report.counters.committed_cursor, 0);
        assert_eq!(report.counters.committed_consumed_tokens, 0);
        assert_eq!(report.counters.committed_supervised_tokens, 0);
        assert_eq!(trainer.optimizer().step_count(), 0);
        assert!(
            trainer.model().inventory().all(|(_, id)| trainer
                .model()
                .parameters()
                .gradient(id)
                .unwrap()
                .is_none())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn microbatch_log_write_failure_discards_partial_window_and_preserves_precharge() {
        let (mut trainer, root) = with_failure_writer(
            "micro-log-write",
            b"\"type\":\"microbatch\"",
            true,
            StopToken::new(),
        );
        assert_eq!(
            trainer.run(None).unwrap_err(),
            TrainingError::Logging("microbatch")
        );
        assert_eq!(trainer.counters().completed_updates, 0);
        assert_eq!(trainer.counters().committed_consumed_tokens, 0);
        assert_eq!(trainer.counters().committed_cursor, 0);
        assert_eq!(trainer.counters().attempted_consumed_tokens, 32);
        assert!(root.join("checkpoint.xyckpt").is_file());
        let report: Value =
            serde_json::from_slice(&fs::read(root.join("final-report.json")).unwrap()).unwrap();
        assert_eq!(report["error"], "Logging(\"microbatch\")");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn microbatch_log_flush_failure_stops_before_optimizer_commit() {
        let (mut trainer, root) = with_failure_writer(
            "micro-log-flush",
            b"\"type\":\"microbatch\"",
            false,
            StopToken::new(),
        );
        assert_eq!(
            trainer.run(None).unwrap_err(),
            TrainingError::Logging("microbatch")
        );
        assert_eq!(trainer.optimizer().step_count(), 0);
        assert_eq!(trainer.counters().attempted_consumed_tokens, 32);
        assert_eq!(trainer.counters().committed_consumed_tokens, 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn postcommit_metric_log_failure_keeps_committed_update_and_checkpoint() {
        let (mut trainer, root) = with_failure_writer(
            "postcommit-log",
            b"\"type\":\"optimizer_update\"",
            false,
            StopToken::new(),
        );
        assert_eq!(
            trainer.run(None).unwrap_err(),
            TrainingError::Logging("optimizer_update")
        );
        assert_eq!(trainer.optimizer().step_count(), 1);
        assert_eq!(trainer.counters().completed_updates, 1);
        assert_eq!(trainer.counters().committed_consumed_tokens, 32);
        assert_eq!(trainer.counters().committed_cursor, 1);
        assert!(root.join("checkpoint.xyckpt").is_file());
        let report: Value =
            serde_json::from_slice(&fs::read(root.join("final-report.json")).unwrap()).unwrap();
        assert_eq!(report["counters"]["completed_updates"], 1);
        assert_eq!(report["error"], "Logging(\"optimizer_update\")");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn final_report_publication_failure_returns_nonzero_runtime_error() {
        let stop = StopToken::new();
        stop.request();
        let (mut trainer, root) =
            with_failure_writer("final-report-fail", b"no match", false, stop);
        fs::create_dir(root.join("final-report.json")).unwrap();
        assert_eq!(
            trainer.run(None).unwrap_err(),
            TrainingError::RuntimeArtifact("final-report")
        );
        assert!(root.join("checkpoint.xyckpt").is_file());
        assert!(root.join("final-report.json").is_dir());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn objective_a_runs_bounded_updates_and_reports_heldout_learning_metrics() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let config = config(&fixture, 1, 2)
            .with_validation_limits(
                crate::EvaluationLimits::new(2, 4_096, 250_000_000, 4 << 20, 32).unwrap(),
            )
            .unwrap();
        let train_order = identity.data_order_identity();
        let validation_order = identity.validation_order_identity();
        let preparation_start = Instant::now();
        let train_batches = batches(&fixture, "train", &tokenizer);
        let heldout = batches(&fixture, "validation", &tokenizer);
        let fixture_preparation = preparation_start.elapsed();
        let train_source = VecBatchSource::new(train_batches, train_order);
        let mut validation_source = VecBatchSource::for_split(
            copy_batches(&heldout),
            validation_order,
            xynoos_data::Split::Validation,
        );
        let root = root("objective-a");
        let orphan = root.join(".xyckpt-orphan.tmp");
        fs::write(&orphan, b"interrupted checkpoint").unwrap();
        let mut trainer = Trainer::new(
            &root,
            identity.clone(),
            config,
            &tokenizer,
            train_source,
            StopToken::new(),
        )
        .unwrap();
        assert!(!orphan.exists());
        let ledger_baseline = trainer.model().parameters().accounted_bytes();
        let before = mean_loss(trainer.model(), &heldout);
        let parameter_id = trainer.model().inventory().next().unwrap().1;
        let before_parameter: Vec<u32> = trainer
            .model()
            .parameters()
            .value(parameter_id)
            .unwrap()
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let report = trainer.run(Some(&mut validation_source)).unwrap();
        let after = mean_loss(trainer.model(), &heldout);
        let after_parameter: Vec<u32> = trainer
            .model()
            .parameters()
            .value(parameter_id)
            .unwrap()
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(report.stop_reason, StopReason::ScheduleComplete);
        assert_eq!(report.counters.completed_updates, 2);
        assert_eq!(report.counters.committed_cursor, 2);
        assert_eq!(report.counters.committed_consumed_tokens, 64);
        assert_eq!(report.counters.committed_supervised_tokens, 64);
        assert_eq!(report.counters.validation_supervised_tokens, 128);
        assert_ne!(
            before_parameter, after_parameter,
            "the tied embedding Transformer parameter must update"
        );
        assert!(
            trainer
                .optimizer()
                .first_moment(0)
                .unwrap()
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(
            trainer
                .optimizer()
                .second_moment(0)
                .unwrap()
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(before.is_finite() && after.is_finite());
        assert!(report.owned_memory_peak_bytes <= config.limits().owned_memory_bytes());
        assert!(report.autograd_ledger_peak_bytes <= config.autograd_limits().max_bytes());
        assert_eq!(
            report.charged_owned_current_bytes,
            config.memory().persistent_bytes()
        );
        assert_eq!(report.autograd_ledger_current_bytes, ledger_baseline);
        println!(
            "TASK006 ObjectiveA fixture={} sha256={} train_rows=2 validation_rows=2 fixture_preparation_ms={:.6} fixture_preparation_samples=4 updates={} heldout_mean_loss_before={before:.9} after={after:.9} delta={:.9} supervised_tokens={} tokens_per_second={:.3} memory_owned_peak={} memory_owned_current={} autograd_ledger_peak={} autograd_ledger_current={} process_vm_hwm={:?} timings={}",
            fixture["fixture_id"],
            hex(&Sha256::digest(FIXTURE.as_bytes())),
            fixture_preparation.as_secs_f64() * 1000.0,
            report.counters.completed_updates,
            after - before,
            report.counters.committed_supervised_tokens,
            report.supervised_tokens_per_second,
            report.owned_memory_peak_bytes,
            report.charged_owned_current_bytes,
            report.autograd_ledger_peak_bytes,
            report.autograd_ledger_current_bytes,
            report.process_vm_hwm_bytes,
            report.timings,
        );
        let final_report: Value =
            serde_json::from_slice(&fs::read(root.join("final-report.json")).unwrap()).unwrap();
        assert_eq!(final_report["stop_reason"], "schedule_complete");
        assert_eq!(final_report["counters"]["completed_updates"], 2);
        let handoff: Value =
            serde_json::from_slice(&fs::read(root.join("handoff.json")).unwrap()).unwrap();
        assert_eq!(handoff["budget"]["max_updates"], 2);
        assert_eq!(
            handoff["budget"]["active_runtime_cap_ns"],
            config.limits().max_runtime().as_nanos() as u64
        );
        let logs = fs::read_to_string(root.join("metrics.jsonl")).unwrap();
        assert!(logs.lines().any(|line| {
            serde_json::from_str::<Value>(line)
                .is_ok_and(|record| record["type"] == "optimizer_update")
        }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn objective_a_initial_baseline_covers_train_validation_and_locked_test() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let source_identity = crate::attest_source_checkout(
            Path::new(env!("CARGO_MANIFEST_DIR")),
            crate::SOURCE_REPOSITORY_ID,
            env!("XYNOOS_BUILD_SOURCE_REVISION"),
        )
        .unwrap();
        let splits = &fixture["split_sha256"];
        let identity = RunIdentity::new(
            source_identity.revision().to_owned(),
            true,
            fixture["fixture_id"].as_str().unwrap().to_owned(),
            Sha256::digest(FIXTURE.as_bytes()).into(),
            digest(splits["train"].as_str().unwrap()),
            digest(splits["validation"].as_str().unwrap()),
            digest(splits["test"].as_str().unwrap()),
            "task006-objective-a-v1".to_owned(),
            tokenizer.algorithm().to_owned(),
            digest(fixture["tokenizer_artifact_sha256"].as_str().unwrap()),
            292,
            digest(fixture["data_order_identity_sha256"].as_str().unwrap()),
            digest(
                fixture["validation_order_identity_sha256"]
                    .as_str()
                    .unwrap(),
            ),
        );
        let model = Transformer::new(
            Config::RESEARCH_CANDIDATE,
            &tokenizer,
            fixture["model_seed"].as_u64().unwrap(),
            ResourceLimits::default(),
        )
        .unwrap();
        let limits = |batches| {
            crate::EvaluationLimits::new(batches, batches * 32, 250_000_000, 4 << 20, 32).unwrap()
        };
        let train = crate::evaluate_batches(
            &model,
            &tokenizer,
            &identity,
            crate::EvaluationStage::InitialBaseline,
            VecBatchSource::for_split(
                batches(&fixture, "train", &tokenizer),
                identity.data_order_identity(),
                xynoos_data::Split::Train,
            ),
            limits(2),
        )
        .unwrap();
        let validation = crate::evaluate_batches(
            &model,
            &tokenizer,
            &identity,
            crate::EvaluationStage::InitialBaseline,
            VecBatchSource::for_split(
                batches(&fixture, "validation", &tokenizer),
                identity.validation_order_identity(),
                xynoos_data::Split::Validation,
            ),
            limits(2),
        )
        .unwrap();
        let mut test_order = Sha256::new();
        test_order.update(b"XYNOOS-TASK006-OBJECTIVE-A-TEST-ORDER-V1\0");
        for row in fixture["sample_order"]["test"].as_array().unwrap() {
            test_order.update(row.as_str().unwrap().as_bytes());
        }
        let test = crate::evaluate_batches(
            &model,
            &tokenizer,
            &identity,
            crate::EvaluationStage::InitialBaseline,
            VecBatchSource::for_split(
                batches(&fixture, "test", &tokenizer),
                test_order.finalize().into(),
                xynoos_data::Split::Test,
            ),
            limits(2),
        )
        .unwrap();

        assert_eq!(train.batch_count, 2);
        assert_eq!(validation.batch_count, 2);
        assert_eq!(test.batch_count, 2);
        assert_eq!(train.supervised_tokens, 64);
        assert_eq!(validation.supervised_tokens, 64);
        assert_eq!(test.supervised_tokens, 64);
        assert!(train.mean_loss.unwrap().is_finite());
        assert!(validation.mean_loss.unwrap().is_finite());
        assert!(test.mean_loss.unwrap().is_finite());
        assert!(!train.test_access && !validation.test_access && test.test_access);
        assert_eq!(train.model_state_sha256, validation.model_state_sha256);
        assert_eq!(validation.model_state_sha256, test.model_state_sha256);
        println!("TASK006_D1_BASELINE train={}", train.to_json());
        println!("TASK006_D1_BASELINE validation={}", validation.to_json());
        println!("TASK006_D1_BASELINE test_access={}", test.to_json());
    }

    #[test]
    fn trainer_rejects_test_split_before_model_allocation() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let source = VecBatchSource::for_split(
            batches(&fixture, "test", &tokenizer),
            identity.data_order_identity(),
            xynoos_data::Split::Test,
        );
        let root = root("test-split-training-rejection");
        assert_eq!(
            Trainer::new(
                &root,
                identity,
                config(&fixture, 1, 2),
                &tokenizer,
                source,
                StopToken::new(),
            )
            .err(),
            Some(TrainingError::InvalidIdentity)
        );
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn uninterrupted_and_checkpoint_resumed_updates_are_bitwise_equivalent() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let config = config(&fixture, 1, 2);
        let order = identity.data_order_identity();

        let full_root = root("resume-full");
        let full_source = VecBatchSource::new(batches(&fixture, "train", &tokenizer), order);
        let mut full = Trainer::new(
            &full_root,
            identity.clone(),
            config,
            &tokenizer,
            full_source,
            StopToken::new(),
        )
        .unwrap();
        let full_report = full.run(None).unwrap();
        let full_state = state_bits(&full);
        drop(full);

        let resumed_root = root("resume-split");
        let partial_source = VecBatchSource::new(
            batches(&fixture, "train", &tokenizer)
                .into_iter()
                .take(1)
                .collect(),
            order,
        );
        let mut partial = Trainer::new(
            &resumed_root,
            identity.clone(),
            config,
            &tokenizer,
            partial_source,
            StopToken::new(),
        )
        .unwrap();
        let partial_report = partial.run(None).unwrap();
        assert_eq!(partial_report.stop_reason, StopReason::InputExhausted);
        assert_eq!(partial_report.counters.completed_updates, 1);
        let mut codec_load_samples = Samples::default();
        let codec_load_start = Instant::now();
        let mid_checkpoint =
            CheckpointCodec::load(&resumed_root, &identity, config, &tokenizer).unwrap();
        codec_load_samples.push(codec_load_start.elapsed());
        for _ in 1..3 {
            let codec_load_start = Instant::now();
            let repeated =
                CheckpointCodec::load(&resumed_root, &identity, config, &tokenizer).unwrap();
            codec_load_samples.push(codec_load_start.elapsed());
            assert_eq!(repeated.cursor, mid_checkpoint.cursor);
            assert_eq!(repeated.digest, mid_checkpoint.digest);
        }
        println!(
            "TASK006 CheckpointCodecLoad samples={}",
            codec_load_samples.summary().json()
        );
        assert_eq!(mid_checkpoint.cursor, 1);
        assert_eq!(
            config
                .schedule()
                .next_rate(mid_checkpoint.optimizer_step)
                .unwrap()
                .to_bits(),
            config.schedule().next_rate(1).unwrap().to_bits()
        );
        assert_eq!(
            mid_checkpoint.next_batch_identity,
            next_batch_identity(&order, 1)
        );
        drop(partial);

        let resumed_source = VecBatchSource::new(batches(&fixture, "train", &tokenizer), order);
        let mut resumed = Trainer::resume(
            &resumed_root,
            identity.clone(),
            config,
            &tokenizer,
            resumed_source,
            StopToken::new(),
        )
        .unwrap();
        let resumed_report = resumed.run(None).unwrap();
        let resumed_state = state_bits(&resumed);
        assert_eq!(
            next_batch_identity(&order, resumed_report.counters.committed_cursor),
            next_batch_identity(&order, full_report.counters.committed_cursor)
        );
        println!(
            "TASK006 ResumeLoad checkpoint_load={} shutdown={} memory_owned_current={} autograd_ledger_current={} process_vm_hwm={:?}",
            resumed_report.timings["checkpoint_load"],
            resumed_report.timings["shutdown"],
            resumed_report.charged_owned_current_bytes,
            resumed_report.autograd_ledger_current_bytes,
            resumed_report.process_vm_hwm_bytes,
        );
        assert_eq!(resumed_state, full_state);
        assert_eq!(
            resumed_report.counters.completed_updates,
            full_report.counters.completed_updates
        );
        assert_eq!(
            resumed_report.counters.committed_cursor,
            full_report.counters.committed_cursor
        );
        assert_eq!(
            resumed_report.counters.committed_consumed_tokens,
            full_report.counters.committed_consumed_tokens
        );
        assert_eq!(
            resumed_report.counters.committed_supervised_tokens,
            full_report.counters.committed_supervised_tokens
        );
        assert_eq!(
            resumed_report.counters.attempted_consumed_tokens,
            full_report.counters.attempted_consumed_tokens
        );
        assert_eq!(
            resumed_report.counters.attempted_supervised_tokens,
            full_report.counters.attempted_supervised_tokens
        );
        assert_eq!(
            resumed_report.counters.attempted_updates,
            full_report.counters.attempted_updates
        );
        assert_eq!(
            resumed_report.counters.last_metric_sequence,
            full_report.counters.last_metric_sequence
        );
        assert_eq!(
            resumed_report.last_loss.unwrap().to_bits(),
            full_report.last_loss.unwrap().to_bits()
        );
        assert_eq!(
            resumed_report.last_pre_clip_norm.unwrap().to_bits(),
            full_report.last_pre_clip_norm.unwrap().to_bits()
        );
        assert_eq!(
            resumed_report.last_post_clip_norm.unwrap().to_bits(),
            full_report.last_post_clip_norm.unwrap().to_bits()
        );
        assert_eq!(
            resumed_report.last_update_norm.unwrap().to_bits(),
            full_report.last_update_norm.unwrap().to_bits()
        );
        assert!(
            resumed_report.counters.cumulative_work > full_report.counters.cumulative_work + 32,
            "resume read and checkpoint I/O are conservatively charged in addition to replay work"
        );
        fs::remove_dir_all(full_root).unwrap();
        fs::remove_dir_all(resumed_root).unwrap();
    }

    #[test]
    fn partial_window_is_discarded_and_recomputed_without_scientific_double_counting() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let config = config(&fixture, 2, 2);
        let order = identity.data_order_identity();
        let mut all = batches(&fixture, "train", &tokenizer);
        let second_pass = copy_batches(&all);
        all.extend(second_pass);

        let full_root = root("partial-full");
        let full_source = VecBatchSource::new(copy_batches(&all), order);
        let mut full = Trainer::new(
            &full_root,
            identity.clone(),
            config,
            &tokenizer,
            full_source,
            StopToken::new(),
        )
        .unwrap();
        let full_report = full.run(None).unwrap();
        let full_state = state_bits(&full);
        drop(full);

        let resumed_root = root("partial-resumed");
        let partial_source = VecBatchSource::new(copy_batches(&all[..1]), order);
        let mut partial = Trainer::new(
            &resumed_root,
            identity.clone(),
            config,
            &tokenizer,
            partial_source,
            StopToken::new(),
        )
        .unwrap();
        let partial_report = partial.run(None).unwrap();
        assert_eq!(partial_report.stop_reason, StopReason::InputExhausted);
        assert_eq!(partial_report.counters.completed_updates, 0);
        assert_eq!(partial_report.counters.committed_cursor, 0);
        assert_eq!(partial_report.counters.committed_consumed_tokens, 0);
        assert_eq!(partial_report.counters.discarded_consumed_tokens, 32);
        assert_eq!(partial_report.counters.attempted_consumed_tokens, 32);
        let checkpoint =
            CheckpointCodec::load(&resumed_root, &identity, config, &tokenizer).unwrap();
        assert_eq!(checkpoint.cursor, 0);
        drop(partial);

        let mut resumed = Trainer::resume(
            &resumed_root,
            identity.clone(),
            config,
            &tokenizer,
            VecBatchSource::new(copy_batches(&all), order),
            StopToken::new(),
        )
        .unwrap();
        let resumed_report = resumed.run(None).unwrap();
        assert_eq!(state_bits(&resumed), full_state);
        assert_eq!(
            resumed_report.counters.committed_consumed_tokens,
            full_report.counters.committed_consumed_tokens
        );
        assert_eq!(
            resumed_report.counters.committed_supervised_tokens,
            full_report.counters.committed_supervised_tokens
        );
        assert_eq!(resumed_report.counters.committed_cursor, 4);
        assert_eq!(resumed_report.counters.discarded_consumed_tokens, 32);
        assert_eq!(resumed_report.counters.attempted_consumed_tokens, 160);
        assert_eq!(resumed_report.counters.committed_consumed_tokens, 128);
        assert!(resumed_report.counters.cumulative_work > full_report.counters.cumulative_work);
        fs::remove_dir_all(full_root).unwrap();
        fs::remove_dir_all(resumed_root).unwrap();
    }

    #[test]
    fn resume_with_missing_journal_preserves_checkpoint_and_does_not_recreate_lineage() {
        let fixture = fixture();
        let tokenizer = tokenizer();
        let identity = identity(&tokenizer, &fixture);
        let config = config(&fixture, 1, 2);
        let order = identity.data_order_identity();
        let root = root("missing-journal-resume");
        let source = VecBatchSource::new(batches(&fixture, "train", &tokenizer), order);
        let mut trainer = Trainer::new(
            &root,
            identity.clone(),
            config,
            &tokenizer,
            source,
            StopToken::new(),
        )
        .unwrap();
        trainer.run(None).unwrap();
        drop(trainer);
        let checkpoint_before = fs::read(root.join("checkpoint.xyckpt")).unwrap();
        fs::remove_file(root.join("run.journal")).unwrap();

        assert!(matches!(
            Trainer::resume(
                &root,
                identity,
                config,
                &tokenizer,
                VecBatchSource::new(batches(&fixture, "train", &tokenizer), order),
                StopToken::new(),
            ),
            Err(TrainingError::Checkpoint("run_journal_missing_for_resume"))
        ));
        assert_eq!(
            fs::read(root.join("checkpoint.xyckpt")).unwrap(),
            checkpoint_before
        );
        assert!(!root.join("run.journal").exists());
        assert!(root.join(".run.lock").is_file());
        assert!(!crate::AdvisoryLock::is_held(&root.join(".run.lock")).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}
