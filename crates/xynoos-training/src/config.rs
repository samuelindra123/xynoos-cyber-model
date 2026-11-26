use std::time::Duration;

use sha2::{Digest, Sha256};
use xynoos_autograd::{
    AdamWConfig, CandidateTrainingLimits, CandidateTrainingPolicy, ResourceLimits,
};
use xynoos_data::{Dataset, Split};
use xynoos_tokenizer::{Tokenizer, TokenizerArtifact, TokenizerLimits};
use xynoos_transformer::Config;

use crate::provenance::{SOURCE_REPOSITORY_ID, SourceIdentity};
use crate::{EvaluationLimits, LinearWarmupDecay, Result, TrainingError, TrainingMemoryPlan};

pub const MODEL_ID: &str = "xynoos-cyber-0.1";
const SPECIAL_REGISTRY: &[u8] = b"XYNOOS-SPECIALS-V1|BOS=256|EOS=257|PAD=258|DOC_SEP=259";
const MIB: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunLimits {
    microbatches_per_update: usize,
    max_updates: u64,
    consumed_tokens: u64,
    supervised_tokens: u64,
    cumulative_work: u64,
    owned_memory_bytes: usize,
    max_runtime: Duration,
    max_attempt_runtime: Duration,
    checkpoint_every_updates: u64,
    validation_every_updates: u64,
}

impl RunLimits {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        microbatches_per_update: usize,
        max_updates: u64,
        consumed_tokens: u64,
        supervised_tokens: u64,
        cumulative_work: u64,
        owned_memory_bytes: usize,
        max_runtime: Duration,
        max_attempt_runtime: Duration,
        checkpoint_every_updates: u64,
        validation_every_updates: u64,
    ) -> Result<Self> {
        if microbatches_per_update == 0
            || microbatches_per_update > ResourceLimits::MAX_EXPERIMENT_STEPS
            || max_updates == 0
            || max_updates > ResourceLimits::MAX_EXPERIMENT_STEPS as u64
            || consumed_tokens == 0
            || supervised_tokens == 0
            || supervised_tokens > consumed_tokens
            || cumulative_work == 0
            || owned_memory_bytes == 0
            || owned_memory_bytes > ResourceLimits::MAX_BYTES
            || max_runtime.is_zero()
            || max_attempt_runtime.is_zero()
            || max_attempt_runtime > max_runtime
            || max_runtime.as_nanos() > u64::MAX as u128
            || max_attempt_runtime.as_nanos() > u64::MAX as u128
            || checkpoint_every_updates == 0
            || validation_every_updates == 0
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        Ok(Self {
            microbatches_per_update,
            max_updates,
            consumed_tokens,
            supervised_tokens,
            cumulative_work,
            owned_memory_bytes,
            max_runtime,
            max_attempt_runtime,
            checkpoint_every_updates,
            validation_every_updates,
        })
    }

    pub fn microbatches_per_update(self) -> usize {
        self.microbatches_per_update
    }
    pub fn max_updates(self) -> u64 {
        self.max_updates
    }
    pub fn consumed_tokens(self) -> u64 {
        self.consumed_tokens
    }
    pub fn supervised_tokens(self) -> u64 {
        self.supervised_tokens
    }
    pub fn cumulative_work(self) -> u64 {
        self.cumulative_work
    }
    pub fn owned_memory_bytes(self) -> usize {
        self.owned_memory_bytes
    }
    pub fn max_runtime(self) -> Duration {
        self.max_runtime
    }
    pub fn max_attempt_runtime(self) -> Duration {
        self.max_attempt_runtime
    }
    pub fn checkpoint_every_updates(self) -> u64 {
        self.checkpoint_every_updates
    }
    pub fn validation_every_updates(self) -> u64 {
        self.validation_every_updates
    }

    pub(crate) fn candidate_policy(self) -> Result<CandidateTrainingLimits> {
        CandidateTrainingLimits::new(
            self.microbatches_per_update,
            self.max_updates,
            self.consumed_tokens,
            self.supervised_tokens,
            self.cumulative_work,
            self.owned_memory_bytes,
            self.max_runtime,
        )
        .map_err(TrainingError::Autograd)
    }
}

/// Immutable data, source, and tokenizer identities associated with one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunIdentity {
    source_repository_id: String,
    source_revision: String,
    source_clean: bool,
    dataset_id: String,
    dataset_manifest_sha256: [u8; 32],
    train_split_sha256: [u8; 32],
    validation_split_sha256: [u8; 32],
    test_split_sha256: [u8; 32],
    provenance_revision: String,
    tokenizer_algorithm: String,
    tokenizer_sha256: [u8; 32],
    vocabulary_size: u32,
    special_registry_sha256: [u8; 32],
    data_order_identity: [u8; 32],
    validation_order_identity: [u8; 32],
}

impl RunIdentity {
    /// Bind an official run identity to a validated manifest-backed dataset,
    /// tokenizer artifact, and the exact deterministic train/validation
    /// stream orders admitted by the native runtime.
    pub fn from_standard_dataset(
        source_identity: SourceIdentity,
        dataset: &Dataset,
        tokenizer: &Tokenizer,
        data_order_identity: [u8; 32],
        validation_order_identity: [u8; 32],
    ) -> Result<Self> {
        let artifact = TokenizerArtifact::serialize(tokenizer, TokenizerLimits::MAX)
            .map_err(|_| TrainingError::InvalidIdentity)?;
        let mut identity = Self::new(
            source_identity.revision().to_owned(),
            true,
            dataset.manifest().dataset_id.clone(),
            dataset.manifest_sha256(),
            split_identity(dataset, Split::Train),
            split_identity(dataset, Split::Validation),
            split_identity(dataset, Split::Test),
            provenance_identity(dataset),
            tokenizer.algorithm().to_owned(),
            Sha256::digest(artifact).into(),
            u32::try_from(tokenizer.vocabulary_size())
                .map_err(|_| TrainingError::InvalidIdentity)?,
            data_order_identity,
            validation_order_identity,
        );
        identity.source_repository_id = source_identity.repository_id().to_owned();
        identity.validate(tokenizer)?;
        identity.validate_dataset(dataset)?;
        Ok(identity)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        source_revision: String,
        source_clean: bool,
        dataset_id: String,
        dataset_manifest_sha256: [u8; 32],
        train_split_sha256: [u8; 32],
        validation_split_sha256: [u8; 32],
        test_split_sha256: [u8; 32],
        provenance_revision: String,
        tokenizer_algorithm: String,
        tokenizer_sha256: [u8; 32],
        vocabulary_size: u32,
        data_order_identity: [u8; 32],
        validation_order_identity: [u8; 32],
    ) -> Self {
        Self {
            source_repository_id: SOURCE_REPOSITORY_ID.to_owned(),
            source_revision,
            source_clean,
            dataset_id,
            dataset_manifest_sha256,
            train_split_sha256,
            validation_split_sha256,
            test_split_sha256,
            provenance_revision,
            tokenizer_algorithm,
            tokenizer_sha256,
            vocabulary_size,
            special_registry_sha256: Sha256::digest(SPECIAL_REGISTRY).into(),
            data_order_identity,
            validation_order_identity,
        }
    }

    pub fn validate(&self, tokenizer: &Tokenizer) -> Result<()> {
        self.validate_static()?;
        if usize::try_from(self.vocabulary_size).ok() != Some(tokenizer.vocabulary_size())
            || self.tokenizer_algorithm != tokenizer.algorithm()
        {
            return Err(TrainingError::InvalidIdentity);
        }
        let artifact = TokenizerArtifact::serialize(tokenizer, TokenizerLimits::MAX)
            .map_err(|_| TrainingError::InvalidIdentity)?;
        let actual: [u8; 32] = Sha256::digest(&artifact).into();
        if actual != self.tokenizer_sha256 {
            return Err(TrainingError::InvalidIdentity);
        }
        Ok(())
    }

    pub(crate) fn validate_static(&self) -> Result<()> {
        let expected_special_registry: [u8; 32] = Sha256::digest(SPECIAL_REGISTRY).into();
        if self.source_repository_id != SOURCE_REPOSITORY_ID
            || !self.source_clean
            || self.source_revision.len() != 40
            || !self
                .source_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || self.dataset_id.is_empty()
            || self.dataset_id.len() > 128
            || self.provenance_revision.is_empty()
            || self.provenance_revision.len() > 128
            || self.tokenizer_algorithm.is_empty()
            || self.tokenizer_algorithm.len() > 32
            || self.vocabulary_size != 292
            || self.special_registry_sha256 != expected_special_registry
        {
            return Err(TrainingError::InvalidIdentity);
        }
        Ok(())
    }

    pub fn validate_dataset(&self, dataset: &Dataset) -> Result<()> {
        if self.dataset_id != dataset.manifest().dataset_id
            || self.dataset_manifest_sha256 != dataset.manifest_sha256()
            || self.train_split_sha256 != split_identity(dataset, Split::Train)
            || self.validation_split_sha256 != split_identity(dataset, Split::Validation)
            || self.test_split_sha256 != split_identity(dataset, Split::Test)
            || self.provenance_revision != provenance_identity(dataset)
        {
            return Err(TrainingError::InvalidIdentity);
        }
        Ok(())
    }

    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }
    pub fn source_repository_id(&self) -> &str {
        &self.source_repository_id
    }
    pub fn source_clean(&self) -> bool {
        self.source_clean
    }
    pub fn dataset_id(&self) -> &str {
        &self.dataset_id
    }
    pub fn dataset_manifest_sha256(&self) -> [u8; 32] {
        self.dataset_manifest_sha256
    }
    pub fn train_split_sha256(&self) -> [u8; 32] {
        self.train_split_sha256
    }
    pub fn validation_split_sha256(&self) -> [u8; 32] {
        self.validation_split_sha256
    }
    pub fn test_split_sha256(&self) -> [u8; 32] {
        self.test_split_sha256
    }
    pub fn provenance_revision(&self) -> &str {
        &self.provenance_revision
    }
    pub fn tokenizer_algorithm(&self) -> &str {
        &self.tokenizer_algorithm
    }
    pub fn tokenizer_sha256(&self) -> [u8; 32] {
        self.tokenizer_sha256
    }
    pub fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }
    pub fn special_registry_sha256(&self) -> [u8; 32] {
        self.special_registry_sha256
    }
    pub fn data_order_identity(&self) -> [u8; 32] {
        self.data_order_identity
    }
    pub fn validation_order_identity(&self) -> [u8; 32] {
        self.validation_order_identity
    }
}

fn split_identity(dataset: &Dataset, split: Split) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-SPLIT-IDENTITY-V1\0");
    for record in dataset
        .records()
        .iter()
        .filter(|record| record.split == split)
    {
        hasher.update((record.document_id.len() as u64).to_le_bytes());
        hasher.update(record.document_id.as_bytes());
        hasher.update(record.content_sha256.as_bytes());
        hasher.update((record.byte_length as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

fn provenance_identity(dataset: &Dataset) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-PROVENANCE-IDENTITY-V1\0");
    for source in dataset.sources() {
        hasher.update((source.source_id.len() as u64).to_le_bytes());
        hasher.update(source.source_id.as_bytes());
        hasher.update((source.source_revision.len() as u64).to_le_bytes());
        hasher.update(source.source_revision.as_bytes());
    }
    let digest = hasher.finalize();
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(result, "{byte:02x}");
    }
    result
}

/// Validated fixed-model run configuration. All budgets are finite and bounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingConfig {
    model: Config,
    optimizer: AdamWConfig,
    schedule: LinearWarmupDecay,
    limits: RunLimits,
    memory: TrainingMemoryPlan,
    autograd_limits: ResourceLimits,
    gradient_clip: Option<f32>,
    seed: u64,
    validation_limits: EvaluationLimits,
}

impl TrainingConfig {
    pub fn new(
        optimizer: AdamWConfig,
        schedule: LinearWarmupDecay,
        limits: RunLimits,
        memory: TrainingMemoryPlan,
        autograd_limits: ResourceLimits,
        gradient_clip: Option<f32>,
        seed: u64,
    ) -> Result<Self> {
        let validation_limits = EvaluationLimits::default_for_source(limits.owned_memory_bytes());
        let config = Self {
            model: Config::RESEARCH_CANDIDATE,
            optimizer,
            schedule,
            limits,
            memory,
            autograd_limits,
            gradient_clip,
            seed,
            validation_limits,
        };
        config.validate_static()?;
        Ok(config)
    }

    /// Set the deterministic, bounded multi-batch validation policy.
    pub fn with_validation_limits(mut self, limits: EvaluationLimits) -> Result<Self> {
        self.validation_limits = limits;
        self.validate_static()?;
        Ok(self)
    }

    pub(crate) fn validate_static(self) -> Result<()> {
        if self.model != Config::RESEARCH_CANDIDATE
            || self.schedule.total_updates() > self.limits.max_updates
            || self.memory.owned_cap() != self.limits.owned_memory_bytes
            || self.memory.data_allowance() > 64 * MIB
            || self.autograd_limits.max_bytes() > self.limits.owned_memory_bytes
            || self.validation_limits.max_source_bytes() > self.limits.owned_memory_bytes
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        self.memory.validate()?;
        self.schedule
            .validate_decay(self.optimizer.weight_decay())?;
        if let Some(threshold) = self.gradient_clip
            && (!threshold.is_finite() || threshold <= 0.0)
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        self.limits.candidate_policy()?;
        Ok(())
    }

    pub fn validate(self, identity: &RunIdentity, tokenizer: &Tokenizer) -> Result<()> {
        self.validate_static()?;
        identity.validate(tokenizer)
    }

    pub fn model(self) -> Config {
        self.model
    }
    pub fn optimizer(self) -> AdamWConfig {
        self.optimizer
    }
    pub fn schedule(self) -> LinearWarmupDecay {
        self.schedule
    }
    pub fn limits(self) -> RunLimits {
        self.limits
    }
    pub fn memory(self) -> TrainingMemoryPlan {
        self.memory
    }
    pub fn autograd_limits(self) -> ResourceLimits {
        self.autograd_limits
    }
    pub fn gradient_clip(self) -> Option<f32> {
        self.gradient_clip
    }
    pub fn seed(self) -> u64 {
        self.seed
    }
    pub fn validation_limits(self) -> EvaluationLimits {
        self.validation_limits
    }

    pub fn candidate_policy(self) -> Result<CandidateTrainingPolicy> {
        Ok(CandidateTrainingPolicy::new(
            self.limits.candidate_policy()?,
        ))
    }

    /// Canonical digest over the typed configuration fields persisted by v1.
    pub fn digest(self, identity: &RunIdentity) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(MODEL_ID.as_bytes());
        bytes.extend_from_slice(&(identity.source_repository_id.len() as u64).to_le_bytes());
        bytes.extend_from_slice(identity.source_repository_id.as_bytes());
        bytes.extend_from_slice(identity.source_revision.as_bytes());
        bytes.push(u8::from(identity.source_clean));
        bytes.extend_from_slice(&(identity.dataset_id.len() as u64).to_le_bytes());
        bytes.extend_from_slice(identity.dataset_id.as_bytes());
        bytes.extend_from_slice(&identity.dataset_manifest_sha256);
        bytes.extend_from_slice(&identity.train_split_sha256);
        bytes.extend_from_slice(&identity.validation_split_sha256);
        bytes.extend_from_slice(&identity.test_split_sha256);
        bytes.extend_from_slice(&(identity.provenance_revision.len() as u64).to_le_bytes());
        bytes.extend_from_slice(identity.provenance_revision.as_bytes());
        bytes.extend_from_slice(&(identity.tokenizer_algorithm.len() as u64).to_le_bytes());
        bytes.extend_from_slice(identity.tokenizer_algorithm.as_bytes());
        bytes.extend_from_slice(&identity.tokenizer_sha256);
        bytes.extend_from_slice(&identity.vocabulary_size.to_le_bytes());
        bytes.extend_from_slice(&identity.special_registry_sha256);
        bytes.extend_from_slice(&identity.data_order_identity);
        bytes.extend_from_slice(&identity.validation_order_identity);
        bytes.extend_from_slice(&(self.memory.data_allowance() as u64).to_le_bytes());
        for value in [
            self.model.vocabulary as u64,
            self.model.width as u64,
            self.model.layers as u64,
            self.model.heads as u64,
            self.model.ffn as u64,
            self.model.context as u64,
            self.limits.microbatches_per_update as u64,
            self.limits.max_updates,
            self.limits.consumed_tokens,
            self.limits.supervised_tokens,
            self.limits.cumulative_work,
            self.limits.owned_memory_bytes as u64,
            self.limits.max_runtime.as_nanos() as u64,
            self.limits.max_attempt_runtime.as_nanos() as u64,
            self.limits.checkpoint_every_updates,
            self.limits.validation_every_updates,
            self.schedule.total_updates(),
            self.schedule.warmup_updates(),
            self.seed,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            self.optimizer.beta1(),
            self.optimizer.beta2(),
            self.optimizer.epsilon(),
            self.optimizer.weight_decay(),
            self.schedule.peak_rate(),
            self.schedule.final_rate(),
        ] {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        match self.gradient_clip {
            Some(value) => {
                bytes.push(1);
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            None => {
                bytes.push(0);
                bytes.extend_from_slice(&0_u32.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&(self.autograd_limits.max_bytes() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.autograd_limits.max_values() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.autograd_limits.max_operations() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.autograd_limits.max_work() as u64).to_le_bytes());
        // The v1 checkpoint/run identity had no evaluation limits. Preserve
        // its digest for the default policy while binding any configured
        // multi-batch policy into new experiment identities.
        if self.validation_limits
            != EvaluationLimits::default_for_source(self.limits.owned_memory_bytes())
        {
            bytes.extend_from_slice(&self.validation_limits.max_batches().to_le_bytes());
            bytes.extend_from_slice(&self.validation_limits.max_consumed_tokens().to_le_bytes());
            bytes.extend_from_slice(&self.validation_limits.max_forward_work().to_le_bytes());
            bytes.extend_from_slice(
                &(self.validation_limits.max_source_bytes() as u64).to_le_bytes(),
            );
            bytes.extend_from_slice(
                &(self.validation_limits.max_batch_positions() as u64).to_le_bytes(),
            );
        }
        Sha256::digest(bytes).into()
    }
}

pub(crate) fn experiment_id(identity: &RunIdentity, config: TrainingConfig) -> String {
    let digest = config.digest(identity);
    let mut text = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(text, "{byte:02x}");
    }
    text
}
