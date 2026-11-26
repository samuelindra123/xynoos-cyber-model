use sha2::{Digest, Sha256};
use xynoos_autograd::ResourceLimits;
use xynoos_data::{Batch, Dataset, FittedTokenizer, Split, StreamConfig};
use xynoos_tokenizer::Tokenizer;
use xynoos_transformer::{Config, Transformer};

use crate::{BatchSource, MODEL_ID, Result, RunIdentity, TrainingError};

const MAX_EVALUATION_BATCHES: u64 = 4096;
const MAX_EVALUATION_TOKENS: u64 = 16_777_216;

/// A hard ceiling for one deterministic evaluation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationLimits {
    max_batches: u64,
    max_consumed_tokens: u64,
    max_forward_work: u64,
    max_source_bytes: usize,
    max_batch_positions: usize,
}

impl EvaluationLimits {
    pub fn new(
        max_batches: u64,
        max_consumed_tokens: u64,
        max_forward_work: u64,
        max_source_bytes: usize,
        max_batch_positions: usize,
    ) -> Result<Self> {
        if max_batches == 0
            || max_batches > MAX_EVALUATION_BATCHES
            || max_consumed_tokens == 0
            || max_consumed_tokens > MAX_EVALUATION_TOKENS
            || max_forward_work == 0
            || max_forward_work
                > u64::try_from(ResourceLimits::MAX_WORK)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(MAX_EVALUATION_BATCHES)
            || max_source_bytes == 0
            || max_source_bytes > ResourceLimits::MAX_BYTES
            || max_batch_positions == 0
            || max_batch_positions > 65_536
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        Ok(Self {
            max_batches,
            max_consumed_tokens,
            max_forward_work,
            max_source_bytes,
            max_batch_positions,
        })
    }

    pub fn max_batches(self) -> u64 {
        self.max_batches
    }

    pub fn max_consumed_tokens(self) -> u64 {
        self.max_consumed_tokens
    }

    pub fn max_forward_work(self) -> u64 {
        self.max_forward_work
    }

    pub fn max_source_bytes(self) -> usize {
        self.max_source_bytes
    }

    pub fn max_batch_positions(self) -> usize {
        self.max_batch_positions
    }
}

impl Default for EvaluationLimits {
    fn default() -> Self {
        Self {
            max_batches: 1,
            max_consumed_tokens: 65_536,
            max_forward_work: ResourceLimits::MAX_WORK as u64,
            max_source_bytes: 64 * 1024 * 1024,
            max_batch_positions: 65_536,
        }
    }
}

impl EvaluationLimits {
    pub(crate) fn default_for_source(max_source_bytes: usize) -> Self {
        Self {
            max_source_bytes: max_source_bytes.min(64 * 1024 * 1024),
            ..Self::default()
        }
    }
}

/// Names why a split was evaluated. A locked-test record is clearly tagged
/// and the evaluator exposes only an immutable model reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationStage {
    InitialBaseline,
    PeriodicValidation,
    LockedTest,
}

impl EvaluationStage {
    const fn label(self) -> &'static str {
        match self {
            Self::InitialBaseline => "initial-baseline",
            Self::PeriodicValidation => "periodic-validation",
            Self::LockedTest => "locked-test",
        }
    }

    fn permits(self, split: Split) -> bool {
        match self {
            Self::InitialBaseline => true,
            Self::PeriodicValidation => split == Split::Validation,
            Self::LockedTest => split == Split::Test,
        }
    }
}

/// Reproducible aggregate for one split and one exact model state.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationReport {
    pub stage: EvaluationStage,
    pub split: Split,
    pub dataset_id: String,
    pub dataset_manifest_sha256: String,
    pub split_sha256: String,
    pub data_order_identity_sha256: String,
    pub source_repository_id: String,
    pub source_revision: String,
    pub source_clean: bool,
    pub model_id: &'static str,
    pub model_config: Config,
    pub model_seed: Option<u64>,
    pub model_parameter_count: usize,
    pub model_state_sha256: String,
    pub tokenizer_algorithm: String,
    pub tokenizer_artifact_sha256: String,
    pub evaluation_config_sha256: String,
    pub evaluation_limits: EvaluationLimits,
    pub test_access: bool,
    /// `false` means the ordered source had more batches than this pass cap.
    pub coverage_complete: bool,
    pub batch_count: u64,
    pub consumed_tokens: u64,
    pub supervised_tokens: u64,
    pub mean_loss: Option<f64>,
    pub perplexity: Option<f64>,
    pub perplexity_overflow: bool,
    pub max_batch_loss: Option<f64>,
    pub total_forward_work: u64,
    pub max_batch_forward_work: u64,
    pub max_batch_positions: usize,
    pub source_reserved_bytes: usize,
    pub peak_reserved_bytes: usize,
    pub all_evaluated_losses_finite: bool,
}

impl EvaluationReport {
    /// Return a stable JSON representation suitable for a baseline/test-access log.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "stage": self.stage.label(),
            "split": split_label(self.split),
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "split_sha256": self.split_sha256,
            "data_order_identity_sha256": self.data_order_identity_sha256,
            "source_repository_id": self.source_repository_id,
            "source_revision": self.source_revision,
            "source_clean": self.source_clean,
            "model_id": self.model_id,
            "model_config": {
                "vocabulary": self.model_config.vocabulary,
                "width": self.model_config.width,
                "layers": self.model_config.layers,
                "heads": self.model_config.heads,
                "ffn": self.model_config.ffn,
                "context": self.model_config.context,
            },
            "model_seed": self.model_seed,
            "model_parameter_count": self.model_parameter_count,
            "model_state_sha256": self.model_state_sha256,
            "tokenizer_algorithm": self.tokenizer_algorithm,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "evaluation_limits": {
                "max_batches": self.evaluation_limits.max_batches,
                "max_consumed_tokens": self.evaluation_limits.max_consumed_tokens,
                "max_forward_work": self.evaluation_limits.max_forward_work,
                "max_source_bytes": self.evaluation_limits.max_source_bytes,
                "max_batch_positions": self.evaluation_limits.max_batch_positions,
            },
            "test_access": self.test_access,
            "coverage_complete": self.coverage_complete,
            "batch_count": self.batch_count,
            "consumed_tokens": self.consumed_tokens,
            "supervised_tokens": self.supervised_tokens,
            "mean_loss": self.mean_loss,
            "perplexity": self.perplexity,
            "perplexity_overflow": self.perplexity_overflow,
            "empty_supervision": self.supervised_tokens == 0,
            "max_batch_loss": self.max_batch_loss,
            "total_forward_work": self.total_forward_work,
            "max_batch_forward_work": self.max_batch_forward_work,
            "max_batch_positions": self.max_batch_positions,
            "source_reserved_bytes": self.source_reserved_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "all_evaluated_losses_finite": self.all_evaluated_losses_finite,
        })
    }
}

/// Evaluate a complete, stable batch source with a token-weighted mean loss.
/// `source` is consumed so the caller cannot accidentally continue from a
/// partially evaluated cursor. Standard `BatchStream` sources carry their
/// split tag and deterministic order identity from the validated dataset.
pub fn evaluate_batches<S: BatchSource>(
    model: &Transformer,
    tokenizer: &Tokenizer,
    identity: &RunIdentity,
    stage: EvaluationStage,
    source: S,
    limits: EvaluationLimits,
) -> Result<EvaluationReport> {
    identity.validate(tokenizer)?;
    evaluate_batches_inner(model, identity, stage, source, limits, |_| Ok(()))
}

/// Evaluate a split from a manifest-validated dataset and its fitted tokenizer.
/// This is the preferred entry point for baselines and locked-test reports: it
/// verifies the dataset identity and constructs the deterministic split stream
/// itself rather than accepting caller-provided batch contents.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_standard_dataset_split(
    model: &Transformer,
    dataset: &Dataset,
    tokenizer: &FittedTokenizer,
    identity: &RunIdentity,
    stage: EvaluationStage,
    split: Split,
    stream: StreamConfig,
    limits: EvaluationLimits,
) -> Result<EvaluationReport> {
    identity.validate(tokenizer.model())?;
    identity.validate_dataset(dataset)?;
    let source = dataset.fitted_batches(split, tokenizer, stream)?;
    evaluate_batches(model, tokenizer.model(), identity, stage, source, limits)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvaluationCharge {
    BatchRead,
    ForwardPass,
}

pub(crate) fn evaluate_batches_with_charge<S, F>(
    model: &Transformer,
    identity: &RunIdentity,
    stage: EvaluationStage,
    source: S,
    limits: EvaluationLimits,
    charge: F,
) -> Result<EvaluationReport>
where
    S: BatchSource,
    F: FnMut(EvaluationCharge) -> Result<()>,
{
    identity.validate_static()?;
    evaluate_batches_inner(model, identity, stage, source, limits, charge)
}

fn evaluate_batches_inner<S, F>(
    model: &Transformer,
    identity: &RunIdentity,
    stage: EvaluationStage,
    mut source: S,
    limits: EvaluationLimits,
    mut charge: F,
) -> Result<EvaluationReport>
where
    S: BatchSource,
    F: FnMut(EvaluationCharge) -> Result<()>,
{
    if source.cursor() != 0 || !stage.permits(source.split()) {
        return Err(TrainingError::EvaluationSplitMismatch);
    }
    let split = source.split();
    match split {
        Split::Train if source.order_identity() != identity.data_order_identity() => {
            return Err(TrainingError::InvalidIdentity);
        }
        Split::Validation if source.order_identity() != identity.validation_order_identity() => {
            return Err(TrainingError::InvalidIdentity);
        }
        _ => {}
    }
    let source_bytes = source.reserved_memory_bytes()?;
    if source_bytes > limits.max_source_bytes {
        return Err(TrainingError::ResourceLimit {
            requested: source_bytes,
            limit: limits.max_source_bytes,
        });
    }

    let mut batch_count = 0_u64;
    let mut consumed_tokens = 0_u64;
    let mut supervised_tokens = 0_u64;
    let mut weighted_loss = 0.0_f64;
    let mut max_batch_loss: Option<f64> = None;
    let mut total_forward_work = 0_u64;
    let mut max_batch_forward_work = 0_u64;
    let mut max_batch_positions = 0_usize;
    let mut peak_reserved_bytes = source_bytes;
    let mut source_exhausted = false;

    while batch_count < limits.max_batches {
        charge(EvaluationCharge::BatchRead)?;
        let Some(batch) = source.next_batch()? else {
            source_exhausted = true;
            break;
        };
        let (positions, consumed, supervised) = count_batch(&batch)?;
        if positions > limits.max_batch_positions {
            return Err(TrainingError::ResourceLimit {
                requested: positions,
                limit: limits.max_batch_positions,
            });
        }
        let next_consumed = consumed_tokens
            .checked_add(consumed)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if next_consumed > limits.max_consumed_tokens {
            return Err(TrainingError::EvaluationTokenLimit);
        }
        let batch_bytes = source.evaluation_reserved_bytes(batch.retained_bytes()?)?;
        if batch_bytes > limits.max_source_bytes {
            return Err(TrainingError::ResourceLimit {
                requested: batch_bytes,
                limit: limits.max_source_bytes,
            });
        }
        peak_reserved_bytes = peak_reserved_bytes.max(batch_bytes);
        max_batch_positions = max_batch_positions.max(positions);
        batch_count = batch_count
            .checked_add(1)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        consumed_tokens = next_consumed;

        if supervised != 0 {
            charge(EvaluationCharge::ForwardPass)?;
            let evaluation = model.evaluate_batch(&batch)?;
            let loss = f64::from(evaluation.loss());
            if !loss.is_finite() {
                return Err(TrainingError::NonFiniteEvaluation);
            }
            let metrics = evaluation.metrics()?;
            let work = u64::try_from(metrics.forward_work)
                .map_err(|_| TrainingError::ArithmeticOverflow)?;
            total_forward_work = total_forward_work
                .checked_add(work)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            if total_forward_work > limits.max_forward_work {
                return Err(TrainingError::EvaluationWorkLimit);
            }
            max_batch_forward_work = max_batch_forward_work.max(work);
            weighted_loss = add_weighted_loss(weighted_loss, loss, supervised)?;
            supervised_tokens = supervised_tokens
                .checked_add(supervised)
                .ok_or(TrainingError::ArithmeticOverflow)?;
            max_batch_loss = Some(max_batch_loss.map_or(loss, |prior| prior.max(loss)));
        }
    }

    let coverage_complete = if source_exhausted {
        true
    } else if batch_count == limits.max_batches {
        charge(EvaluationCharge::BatchRead)?;
        let lookahead_bytes = limits
            .max_batch_positions
            .checked_mul(2 * std::mem::size_of::<u32>() + 2 * std::mem::size_of::<u8>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Batch>()))
            .ok_or(TrainingError::ArithmeticOverflow)?;
        let reserved = source.evaluation_reserved_bytes(lookahead_bytes)?;
        if reserved > limits.max_source_bytes {
            return Err(TrainingError::ResourceLimit {
                requested: reserved,
                limit: limits.max_source_bytes,
            });
        }
        peak_reserved_bytes = peak_reserved_bytes.max(reserved);
        source.next_batch()?.is_none()
    } else {
        false
    };

    let mean_loss = if supervised_tokens == 0 {
        None
    } else {
        Some(weighted_loss / supervised_tokens as f64)
    };
    let perplexity_candidate = mean_loss.map(f64::exp);
    let perplexity_overflow = perplexity_candidate.is_some_and(|value| !value.is_finite());
    let perplexity = perplexity_candidate.filter(|value| value.is_finite());
    let split_sha256 = match split {
        Split::Train => identity.train_split_sha256(),
        Split::Validation => identity.validation_split_sha256(),
        Split::Test => identity.test_split_sha256(),
    };
    let order_identity = source.order_identity();
    let state_sha256 = model_state_sha256(model)?;
    let evaluation_config_sha256 = evaluation_config_sha256(
        stage,
        split,
        limits,
        identity.dataset_manifest_sha256(),
        split_sha256,
        order_identity,
        identity.tokenizer_sha256(),
        state_sha256,
    );

    Ok(EvaluationReport {
        stage,
        split,
        dataset_id: identity.dataset_id().to_owned(),
        dataset_manifest_sha256: hex(&identity.dataset_manifest_sha256()),
        split_sha256: hex(&split_sha256),
        data_order_identity_sha256: hex(&order_identity),
        source_repository_id: identity.source_repository_id().to_owned(),
        source_revision: identity.source_revision().to_owned(),
        source_clean: identity.source_clean(),
        model_id: MODEL_ID,
        model_config: model.config(),
        model_seed: model.seed(),
        model_parameter_count: model.parameter_count(),
        model_state_sha256: hex(&state_sha256),
        tokenizer_algorithm: identity.tokenizer_algorithm().to_owned(),
        tokenizer_artifact_sha256: hex(&identity.tokenizer_sha256()),
        evaluation_config_sha256: hex(&evaluation_config_sha256),
        evaluation_limits: limits,
        test_access: split == Split::Test,
        coverage_complete,
        batch_count,
        consumed_tokens,
        supervised_tokens,
        mean_loss,
        perplexity,
        perplexity_overflow,
        max_batch_loss,
        total_forward_work,
        max_batch_forward_work,
        max_batch_positions,
        source_reserved_bytes: source_bytes,
        peak_reserved_bytes,
        all_evaluated_losses_finite: true,
    })
}

fn count_batch(batch: &Batch) -> Result<(usize, u64, u64)> {
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
        consumed = consumed
            .checked_add(u64::from(valid))
            .ok_or(TrainingError::ArithmeticOverflow)?;
        supervised = supervised
            .checked_add(u64::from(loss))
            .ok_or(TrainingError::ArithmeticOverflow)?;
    }
    Ok((positions, consumed, supervised))
}

fn add_weighted_loss(total: f64, batch_mean: f64, supervised_tokens: u64) -> Result<f64> {
    let next = total + batch_mean * supervised_tokens as f64;
    if !next.is_finite() {
        return Err(TrainingError::NonFiniteEvaluation);
    }
    Ok(next)
}

fn model_state_sha256(model: &Transformer) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-MODEL-STATE-V1\0");
    update_config(&mut hasher, model.config());
    hasher.update(model.parameter_count().to_le_bytes());
    match model.seed() {
        Some(seed) => {
            hasher.update([1]);
            hasher.update(seed.to_le_bytes());
        }
        None => hasher.update([0]),
    }
    for (spec, id) in model.inventory() {
        hasher.update((spec.index() as u64).to_le_bytes());
        for dimension in spec.shape() {
            hasher.update((*dimension as u64).to_le_bytes());
        }
        let tensor = model.parameters().value(id)?;
        for value in tensor.data() {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(hasher.finalize().into())
}

#[allow(clippy::too_many_arguments)]
fn evaluation_config_sha256(
    stage: EvaluationStage,
    split: Split,
    limits: EvaluationLimits,
    manifest: [u8; 32],
    split_sha256: [u8; 32],
    order_identity: [u8; 32],
    tokenizer_sha256: [u8; 32],
    state_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-EVALUATION-CONFIG-V1\0");
    hasher.update([stage as u8, split as u8]);
    hasher.update(limits.max_batches.to_le_bytes());
    hasher.update(limits.max_consumed_tokens.to_le_bytes());
    hasher.update(limits.max_forward_work.to_le_bytes());
    hasher.update((limits.max_source_bytes as u64).to_le_bytes());
    hasher.update((limits.max_batch_positions as u64).to_le_bytes());
    hasher.update(manifest);
    hasher.update(split_sha256);
    hasher.update(order_identity);
    hasher.update(tokenizer_sha256);
    hasher.update(state_sha256);
    hasher.finalize().into()
}

fn update_config(hasher: &mut Sha256, config: Config) {
    for value in [
        config.vocabulary,
        config.width,
        config.layers,
        config.heads,
        config.ffn,
        config.context,
    ] {
        hasher.update((value as u64).to_le_bytes());
    }
}

fn split_label(split: Split) -> &'static str {
    match split {
        Split::Train => "train",
        Split::Validation => "validation",
        Split::Test => "test",
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VecBatchSource;
    use sha2::Sha256;
    use xynoos_autograd::ResourceLimits;
    use xynoos_data::Split;
    use xynoos_tokenizer::{TokenizerArtifact, TokenizerLimits};
    use xynoos_transformer::Config;

    fn tokenizer() -> Tokenizer {
        let task004: serde_json::Value = serde_json::from_str(include_str!(
            "../../xynoos-data/tests/fixtures/task004.json"
        ))
        .unwrap();
        TokenizerArtifact::read(
            task004["artifact"].as_str().unwrap().as_bytes(),
            TokenizerLimits::MAX,
        )
        .unwrap()
    }

    fn identity(tokenizer: &Tokenizer, train_order: [u8; 32], val_order: [u8; 32]) -> RunIdentity {
        let artifact = TokenizerArtifact::serialize(tokenizer, TokenizerLimits::MAX).unwrap();
        RunIdentity::new(
            "0123456789abcdef0123456789abcdef01234567".to_owned(),
            true,
            "d1-evaluation-fixture".to_owned(),
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            "d1-provenance".to_owned(),
            tokenizer.algorithm().to_owned(),
            Sha256::digest(artifact).into(),
            292,
            train_order,
            val_order,
        )
    }

    fn batch(offset: u32, loss: &[u8]) -> Batch {
        let input = [offset, offset + 1, offset + 2, offset + 3];
        let target = [offset + 1, offset + 2, offset + 3, offset + 4];
        Batch::from_parts(1, 4, &input, &target, Some(&[1; 4]), Some(loss)).unwrap()
    }

    fn limits(max_batches: u64, max_tokens: u64, max_work: u64) -> EvaluationLimits {
        EvaluationLimits::new(max_batches, max_tokens, max_work, 1 << 20, 4).unwrap()
    }

    fn model(tokenizer: &Tokenizer) -> Transformer {
        Transformer::new(
            Config::INTEGRATION,
            tokenizer,
            0xD1,
            ResourceLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn evaluation_uses_token_weighted_loss_and_records_exact_identities() {
        let tokenizer = tokenizer();
        let model = model(&tokenizer);
        let first = batch(1, &[1, 0, 0, 0]);
        let second = batch(20, &[1, 1, 1, 0]);
        let first_loss = f64::from(model.evaluate_batch(&first).unwrap().loss());
        let second_loss = f64::from(model.evaluate_batch(&second).unwrap().loss());
        let expected = (first_loss + 3.0 * second_loss) / 4.0;
        let train_order = [8; 32];
        let identity = identity(&tokenizer, train_order, [9; 32]);
        let report = evaluate_batches(
            &model,
            &tokenizer,
            &identity,
            EvaluationStage::InitialBaseline,
            VecBatchSource::for_split(vec![first, second], train_order, Split::Train),
            limits(2, 8, 200_000_000),
        )
        .unwrap();

        assert_eq!(report.split, Split::Train);
        assert_eq!(report.batch_count, 2);
        assert_eq!(report.consumed_tokens, 8);
        assert_eq!(report.supervised_tokens, 4);
        assert_eq!(report.mean_loss, Some(expected));
        assert_eq!(
            report.model_parameter_count,
            Config::INTEGRATION.parameter_count().unwrap()
        );
        assert!(!report.test_access);
        assert_eq!(report.model_state_sha256.len(), 64);
        assert_eq!(report.tokenizer_artifact_sha256.len(), 64);
        assert_eq!(report.evaluation_config_sha256.len(), 64);
        assert_eq!(report.to_json()["supervised_tokens"], 4);
        assert_eq!(report.to_json()["empty_supervision"], false);
        assert!(report.coverage_complete);
    }

    #[test]
    fn independent_weighted_loss_fixture_matches_token_weighted_aggregation() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tools/fixtures/task006-evaluation-math.json"
        ))
        .unwrap();
        let losses = fixture["batch_mean_losses"].as_array().unwrap();
        let tokens = fixture["supervised_tokens"].as_array().unwrap();
        assert_eq!(losses.len(), tokens.len());
        let mut weighted = 0.0;
        let mut count = 0_u64;
        for (loss, tokens) in losses.iter().zip(tokens) {
            let loss = loss.as_f64().unwrap();
            let tokens = tokens.as_u64().unwrap();
            weighted = add_weighted_loss(weighted, loss, tokens).unwrap();
            count += tokens;
        }
        assert_eq!(
            weighted / count as f64,
            fixture["token_weighted_mean"].as_f64().unwrap()
        );
    }

    #[test]
    fn locked_test_is_explicit_and_periodic_validation_rejects_test_data() {
        let tokenizer = tokenizer();
        let model = model(&tokenizer);
        let identity = identity(&tokenizer, [8; 32], [9; 32]);
        let test_source =
            VecBatchSource::for_split(vec![batch(40, &[1; 4])], [10; 32], Split::Test);
        let report = evaluate_batches(
            &model,
            &tokenizer,
            &identity,
            EvaluationStage::LockedTest,
            test_source,
            limits(1, 4, 100_000_000),
        )
        .unwrap();
        assert!(report.test_access);
        assert_eq!(report.to_json()["stage"], "locked-test");
        assert_eq!(report.to_json()["split"], "test");
        assert_eq!(report.test_access, report.split == Split::Test);

        let invalid = VecBatchSource::for_split(vec![batch(40, &[1; 4])], [10; 32], Split::Test);
        assert_eq!(
            evaluate_batches(
                &model,
                &tokenizer,
                &identity,
                EvaluationStage::PeriodicValidation,
                invalid,
                limits(1, 4, 100_000_000),
            )
            .err(),
            Some(TrainingError::EvaluationSplitMismatch)
        );
    }

    #[test]
    fn evaluation_reports_empty_supervision_without_nan() {
        let tokenizer = tokenizer();
        let model = model(&tokenizer);
        let train_order = [8; 32];
        let identity = identity(&tokenizer, train_order, [9; 32]);
        let report = evaluate_batches(
            &model,
            &tokenizer,
            &identity,
            EvaluationStage::InitialBaseline,
            VecBatchSource::for_split(vec![batch(1, &[0; 4])], train_order, Split::Train),
            limits(1, 4, 100_000_000),
        )
        .unwrap();
        assert_eq!(report.batch_count, 1);
        assert_eq!(report.supervised_tokens, 0);
        assert_eq!(report.mean_loss, None);
        assert_eq!(report.perplexity, None);
        assert!(!report.perplexity_overflow);
        assert_eq!(report.total_forward_work, 0);
        assert_eq!(report.to_json()["empty_supervision"], true);
    }

    #[test]
    fn evaluation_enforces_batch_token_and_work_budgets() {
        let tokenizer = tokenizer();
        let model = model(&tokenizer);
        let order = [8; 32];
        let identity = identity(&tokenizer, order, [9; 32]);
        let make_source = || {
            VecBatchSource::for_split(
                vec![batch(1, &[1; 4]), batch(20, &[1; 4])],
                order,
                Split::Train,
            )
        };
        let limited = evaluate_batches(
            &model,
            &tokenizer,
            &identity,
            EvaluationStage::InitialBaseline,
            make_source(),
            limits(1, 8, 100_000_000),
        )
        .unwrap();
        assert_eq!(limited.batch_count, 1);
        assert!(!limited.coverage_complete);
        assert_eq!(
            evaluate_batches(
                &model,
                &tokenizer,
                &identity,
                EvaluationStage::InitialBaseline,
                make_source(),
                limits(2, 3, 200_000_000),
            )
            .err(),
            Some(TrainingError::EvaluationTokenLimit)
        );
        assert_eq!(
            evaluate_batches(
                &model,
                &tokenizer,
                &identity,
                EvaluationStage::InitialBaseline,
                make_source(),
                limits(2, 8, 1),
            )
            .err(),
            Some(TrainingError::EvaluationWorkLimit)
        );
    }

    #[test]
    fn evaluation_rejects_wrong_train_order_and_invalid_split_stage() {
        let tokenizer = tokenizer();
        let model = model(&tokenizer);
        let identity = identity(&tokenizer, [8; 32], [9; 32]);
        let wrong_order = VecBatchSource::for_split(vec![], [10; 32], Split::Train);
        assert_eq!(
            evaluate_batches(
                &model,
                &tokenizer,
                &identity,
                EvaluationStage::InitialBaseline,
                wrong_order,
                limits(1, 4, 100_000_000),
            )
            .err(),
            Some(TrainingError::InvalidIdentity)
        );
        let wrong_stage = VecBatchSource::for_split(vec![], [9; 32], Split::Validation);
        assert_eq!(
            evaluate_batches(
                &model,
                &tokenizer,
                &identity,
                EvaluationStage::LockedTest,
                wrong_stage,
                limits(1, 4, 100_000_000),
            )
            .err(),
            Some(TrainingError::EvaluationSplitMismatch)
        );
    }
}
