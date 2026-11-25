//! Native, bounded decoder-only Transformer reference.
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod config;
mod init;

use std::fmt;
use xynoos_autograd::{
    GradientBatch, ParameterId, ParameterSet, ReferenceTransformerBackward,
    ReferenceTransformerEvaluation, ResourceLimits, TapeMetrics,
};
use xynoos_data::Batch;
use xynoos_tensor::Tensor;
use xynoos_tokenizer::Tokenizer;

pub use config::{Config, Error};
pub use xynoos_autograd::{CandidateReferenceExperimentPolicy, ReferenceBatch};

use config::{add, check_init_work};
use init::SplitMix;

const BLOCK_NAMES: [&str; 9] = [
    "attention_gain",
    "q",
    "k",
    "v",
    "o",
    "ffn_gain",
    "gate",
    "value",
    "down",
];

/// One entry in the model's deterministic, row-major parameter inventory.
#[derive(Debug, Clone, Copy)]
pub struct ParameterSpec {
    index: usize,
    group: Group,
    dimensions: [usize; 2],
    rank: usize,
}

#[derive(Debug, Clone, Copy)]
enum Group {
    Embedding,
    Block(usize, usize),
    Final,
}

impl ParameterSpec {
    pub fn index(self) -> usize {
        self.index
    }
    pub fn shape(&self) -> &[usize] {
        &self.dimensions[..self.rank]
    }

    fn residual_output(self) -> bool {
        matches!(self.group, Group::Block(_, 4 | 8))
    }
}

impl fmt::Display for ParameterSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.group {
            Group::Embedding => f.write_str("embedding"),
            Group::Block(layer, family) => write!(f, "block.{layer}.{}", BLOCK_NAMES[family]),
            Group::Final => f.write_str("final_gain"),
        }
    }
}

/// Parameters and layout of one Transformer. Evaluation borrows the parameter set.
#[derive(Debug)]
pub struct Transformer {
    config: Config,
    parameters: ParameterSet,
    count: usize,
    seed: Option<u64>,
    init_work: usize,
}

impl Transformer {
    pub fn new(
        config: Config,
        tokenizer: &Tokenizer,
        seed: u64,
        limits: ResourceLimits,
    ) -> Result<Self, Error> {
        let mut rng = SplitMix::new(seed);
        Self::build(config, tokenizer, Some(seed), limits, |set, spec| {
            let mut create = rng.parameter(spec, config.layers);
            set.insert_generated(spec.shape(), |i| Ok(create(i)))?;
            Ok(())
        })
    }

    /// Construct from explicit finite coordinates for small mathematical fixtures.
    /// The callback is called in registered row-major parameter order.
    pub fn from_coordinates(
        config: Config,
        tokenizer: &Tokenizer,
        limits: ResourceLimits,
        mut coordinate: impl FnMut(ParameterSpec, usize) -> f32,
    ) -> Result<Self, Error> {
        Self::build(config, tokenizer, None, limits, |set, spec| {
            set.insert_generated(spec.shape(), |i| Ok(coordinate(spec, i)))?;
            Ok(())
        })
    }

    fn build(
        config: Config,
        tokenizer: &Tokenizer,
        seed: Option<u64>,
        limits: ResourceLimits,
        mut insert: impl FnMut(&mut ParameterSet, ParameterSpec) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let vocabulary = tokenizer.vocabulary_size();
        if vocabulary != config.vocabulary {
            return Err(Error::VocabularyMismatch {
                model: config.vocabulary,
                tokenizer: vocabulary,
            });
        }
        let count = config.parameter_count()?;
        let entries = config.inventory_len()?;
        let init_work = config.initialization_work(count)?;
        check_init_work(limits, init_work)?;
        let mut parameters = if config == Config::RESEARCH_CANDIDATE {
            ParameterSet::with_candidate_reference_inventory(limits)?
        } else {
            ParameterSet::with_inventory(limits, entries, count, config.inventory_ranks()?)?
        };
        let mut actual = 0;
        for index in 0..entries {
            insert(&mut parameters, spec_at(config, entries, index))?;
            let id = parameters.id_at(index)?;
            actual = add(actual, parameters.value(id)?.numel())?;
        }
        if actual != count || parameters.len() != entries {
            return Err(Error::InvalidInventory);
        }
        Ok(Self {
            config,
            parameters,
            count,
            seed,
            init_work,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }
    pub fn parameter_count(&self) -> usize {
        self.count
    }
    pub fn initialization_work(&self) -> usize {
        self.init_work
    }
    pub fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }

    pub fn inventory(&self) -> impl ExactSizeIterator<Item = (ParameterSpec, ParameterId)> + '_ {
        let entries = self.parameters.len();
        (0..entries).map(move |index| {
            (
                spec_at(self.config, entries, index),
                self.parameters.id_at(index).expect("registered parameter"),
            )
        })
    }

    /// Consume a gradient batch with the native transactional accumulation rules.
    pub fn accumulate_gradients(&mut self, batch: GradientBatch) -> Result<(), Error> {
        Ok(self.parameters.accumulate_gradients(batch)?)
    }

    /// Accumulate transactionally and return the work charged on success.
    pub fn accumulate_gradients_with_work(&mut self, batch: GradientBatch) -> Result<usize, Error> {
        Ok(self.parameters.accumulate_gradients_with_work(batch)?)
    }

    pub fn evaluate_batch(&self, batch: &Batch) -> Result<Evaluation<'_>, Error> {
        self.evaluate(ReferenceBatch {
            shape: batch.shape(),
            input: batch.input(),
            target: batch.target(),
            valid: batch.valid_mask(),
            loss: batch.loss_mask(),
        })
    }

    pub fn evaluate(&self, batch: ReferenceBatch<'_>) -> Result<Evaluation<'_>, Error> {
        self.preflight_batch(batch.shape)?;
        Ok(Evaluation(ReferenceTransformerEvaluation::record(
            &self.parameters,
            batch,
            self.config.vocabulary,
            self.config.heads,
            self.config.layers,
        )?))
    }

    /// Evaluate exactly the authorized candidate at B=1 and T=32.
    /// Only this evaluation's complete backward pass receives the scoped ceiling.
    pub fn evaluate_candidate(
        &self,
        batch: ReferenceBatch<'_>,
        policy: CandidateReferenceExperimentPolicy,
    ) -> Result<Evaluation<'_>, Error> {
        if self.config != Config::RESEARCH_CANDIDATE {
            return Err(Error::InvalidCandidateConfiguration {
                actual: self.config,
            });
        }
        if batch.shape != CandidateReferenceExperimentPolicy::BATCH_SHAPE {
            return Err(Error::InvalidCandidateBatchShape {
                actual: batch.shape,
            });
        }
        self.preflight_batch(batch.shape)?;
        Ok(Evaluation(
            ReferenceTransformerEvaluation::record_candidate(&self.parameters, batch, policy)?,
        ))
    }

    fn preflight_batch(&self, shape: [usize; 2]) -> Result<(), Error> {
        self.config.preflight_batch(
            shape,
            self.parameters.limits(),
            self.count,
            self.parameters.accounted_bytes(),
        )
    }
}

fn spec_at(config: Config, entries: usize, index: usize) -> ParameterSpec {
    let (d, f) = (config.width, config.ffn);
    let (group, dimensions, rank) = if index == 0 {
        (Group::Embedding, [config.vocabulary, d], 2)
    } else if index == entries - 1 {
        (Group::Final, [d, 0], 1)
    } else {
        let layer = (index - 1) / 9;
        let family = (index - 1) % 9;
        let (dims, rank) = match family {
            0 | 5 => ([d, 0], 1),
            1..=4 => ([d, d], 2),
            6 | 7 => ([d, f], 2),
            8 => ([f, d], 2),
            _ => unreachable!(),
        };
        (Group::Block(layer, family), dims, rank)
    };
    ParameterSpec {
        index,
        group,
        dimensions,
        rank,
    }
}

/// A complete reference forward pass. Dropping it releases its tape charges.
#[derive(Debug)]
pub struct Evaluation<'a>(ReferenceTransformerEvaluation<'a>);

impl<'a> Evaluation<'a> {
    pub fn logits(&self) -> &Tensor {
        self.0.logits()
    }
    pub fn loss(&self) -> f32 {
        self.0.loss()
    }
    pub fn metrics(&self) -> Result<TapeMetrics, Error> {
        Ok(self.0.metrics()?)
    }

    /// Finish reverse mode before consuming the tape for gradient transfer.
    pub fn backward(self) -> Result<ReferenceTransformerBackward<'a>, Error> {
        Ok(self.0.backward()?)
    }

    /// Consume the evaluation and publish a gradient batch only after successful reverse and transfer.
    pub fn gradients(self) -> Result<(GradientBatch, TapeMetrics), Error> {
        Ok(self.backward()?.into_gradients()?)
    }
}
