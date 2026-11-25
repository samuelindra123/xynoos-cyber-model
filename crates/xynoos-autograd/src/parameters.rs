use std::time::{Duration, Instant};
use xynoos_tensor::{Tensor, TensorError, TensorLimits};

use crate::budget::{add, mul, Budget, OwnedF64Table, OwnedTensor, Table};
use crate::error::{check_finite, finite, same_shape};
use crate::{identity, AutogradError as Error, Pass, ResourceLimits, Stage};

/// Stable handle to a parameter in its originating set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterId {
    owner: u64,
    index: usize,
}

/// Exclusive ownership of parameter values and their persistent gradients.
#[derive(Debug)]
pub struct ParameterSet {
    pub(crate) budget: Budget,
    owner: u64,
    revision: u64,
    candidate_inventory: bool,
    parameters: Table<Parameter>,
}

#[derive(Debug)]
struct Parameter {
    value: OwnedTensor,
    gradient: Option<OwnedTensor>,
}

/// Single-use owned parameter gradients from a successful reverse pass.
///
/// ```compile_fail
/// fn apply_twice(set: &mut xynoos_autograd::ParameterSet, batch: xynoos_autograd::GradientBatch) {
///     set.accumulate_gradients(batch).unwrap();
///     set.accumulate_gradients(batch).unwrap();
/// }
/// ```
#[derive(Debug)]
pub struct GradientBatch {
    owner: u64,
    revision: u64,
    pub(crate) gradients: Table<Option<OwnedTensor>>,
}

impl GradientBatch {
    /// Parameter-set revision captured when the batch was transferred.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Read the transferred gradient for an ID from the originating set.
    pub fn gradient(&self, id: ParameterId) -> Result<Option<&Tensor>, Error> {
        if id.owner != self.owner {
            return Err(Error::InvalidParameterId);
        }
        let slot = self
            .gradients
            .entries
            .get(id.index)
            .ok_or(Error::InvalidParameterId)?;
        Ok(slot.as_ref().map(|value| &value.tensor))
    }
}

impl ParameterSet {
    pub fn new(limits: ResourceLimits) -> Result<Self, Error> {
        Ok(Self {
            owner: identity()?,
            revision: 0,
            candidate_inventory: false,
            budget: Budget::new(limits),
            parameters: Table::empty(),
        })
    }

    pub fn limits(&self) -> ResourceLimits {
        self.budget.limits
    }

    /// Accounted bytes across this set, its live tapes, and all outstanding batches.
    pub fn accounted_bytes(&self) -> usize {
        self.budget.used()
    }

    /// Lifetime high-water mark of application-level charges.
    pub fn peak_accounted_bytes(&self) -> usize {
        self.budget.peak()
    }

    /// Reserve a complete parameter inventory before running its generators.
    pub fn with_inventory(
        limits: ResourceLimits,
        entries: usize,
        elements: usize,
        sum_ranks: usize,
    ) -> Result<Self, Error> {
        let mut set = Self::new(limits)?;
        let payload = add(
            mul(elements, size_of::<f32>())?,
            mul(sum_ranks, size_of::<usize>())?,
        )?;
        let requested = add(payload, mul(entries, size_of::<Parameter>())?)?;
        if requested > limits.max_bytes() {
            return Err(Error::MemoryLimit {
                requested,
                limit: limits.max_bytes(),
            });
        }
        set.parameters = Table::allocate(&set.budget, entries)?;
        Ok(set)
    }

    /// Reserve an inventory for the fixed TASK 005 candidate, whose context
    /// cap is 128. The complete inventory is validated when recording an
    /// opted-in candidate evaluation. Ordinary tapes retain ordinary limits.
    pub fn with_candidate_reference_inventory(limits: ResourceLimits) -> Result<Self, Error> {
        let mut set = Self::with_inventory(limits, 74, 914_400, 131)?;
        set.candidate_inventory = true;
        Ok(set)
    }

    pub(crate) fn has_candidate_inventory(&self) -> bool {
        self.candidate_inventory
    }

    /// Generate a finite parameter inside the accounting ledger.
    pub fn insert_generated(
        &mut self,
        shape: &[usize],
        coordinate: impl FnMut(usize) -> Result<f32, Error>,
    ) -> Result<ParameterId, Error> {
        let revision = self.next_revision()?;
        let index = self.len();
        if shape.len() > TensorLimits::MAX_RANK {
            return Err(TensorError::RankLimitExceeded {
                rank: shape.len(),
                limit: TensorLimits::MAX_RANK,
            }
            .into());
        }
        let len = shape
            .iter()
            .rev()
            .enumerate()
            .try_fold(1_usize, |a, (i, &b)| {
                a.checked_mul(b).ok_or(TensorError::ShapeOverflow {
                    axis: shape.len() - 1 - i,
                })
            })?;
        self.limits().check_work(
            Pass::ParameterInsertion,
            add(self.parameters.move_work(), add(len, 1)?)?,
        )?;
        let max = self.limits().max_bytes() / size_of::<Parameter>();
        let replacement = self.parameters.growth(&self.budget, max)?;
        let value = self.budget.generate(shape, len, coordinate)?;
        check_finite(value.tensor.data(), Stage::Parameter)?;
        self.parameters.push(
            replacement,
            Parameter {
                value,
                gradient: None,
            },
        );
        self.revision = revision;
        Ok(ParameterId {
            owner: self.owner,
            index,
        })
    }

    /// Resolve an inventory coordinate to a handle for this set.
    pub fn id_at(&self, index: usize) -> Result<ParameterId, Error> {
        if index >= self.len() {
            return Err(Error::InvalidParameterId);
        }
        Ok(ParameterId {
            owner: self.owner,
            index,
        })
    }

    pub fn len(&self) -> usize {
        self.parameters.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Deterministic structural fingerprint of the ordered rank/shape inventory.
    /// Runtime owner identity is deliberately separate from this value.
    pub fn inventory_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        for (index, parameter) in self.parameters.entries.iter().enumerate() {
            for value in [index as u64, parameter.value.tensor.rank() as u64] {
                for byte in value.to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                }
            }
            for &dimension in parameter.value.tensor.shape() {
                for byte in (dimension as u64).to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                }
            }
        }
        hash
    }

    /// Restore fixed-candidate parameter coordinates into a freshly seeded
    /// private staging model. This is the only raw-value restore path and
    /// derives the new revision from the inventory size and committed steps.
    pub fn restore_candidate_values(
        &mut self,
        values: &[Vec<f32>],
        committed_steps: u64,
    ) -> Result<(), Error> {
        if values.len() != self.len() {
            return Err(Error::InvalidCandidateInventory);
        }
        crate::CandidateReferenceExperimentPolicy::validate_inventory(self)?;
        if self
            .parameters
            .entries
            .iter()
            .any(|parameter| parameter.gradient.is_some())
        {
            return Err(Error::InvalidOptimizerState);
        }
        let base_revision = u64::try_from(self.len()).map_err(|_| Error::ArithmeticOverflow)?;
        let revision = base_revision
            .checked_add(committed_steps)
            .ok_or(Error::RevisionExhausted)?;
        let mut work = mul(self.len(), 2)?;
        for (parameter, source) in self.parameters.entries.iter().zip(values) {
            if source.len() != parameter.value.tensor.numel() {
                return Err(Error::InvalidGradientInventory);
            }
            work = add(work, source.len())?;
            check_finite(source, Stage::Parameter)?;
        }
        self.limits().check_work(Pass::ParameterInsertion, work)?;
        let mut staged = Table::allocate(&self.budget, self.len())?;
        for (parameter, source) in self.parameters.entries.iter().zip(values) {
            staged.entries.push(self.budget.generate(
                parameter.value.tensor.shape(),
                source.len(),
                |i| finite(source[i], Stage::Parameter, i),
            )?);
        }
        for (parameter, value) in self.parameters.entries.iter_mut().zip(&mut staged.entries) {
            std::mem::swap(&mut parameter.value, value);
        }
        self.revision = revision;
        Ok(())
    }

    /// Adopt a finite tensor and advance the inventory revision.
    pub fn insert(&mut self, initial: Tensor) -> Result<ParameterId, Error> {
        let revision = self.next_revision()?;
        let index = self.len();
        let metadata = self.parameters.move_work();
        self.limits().check_work(
            Pass::ParameterInsertion,
            add(metadata, add(initial.numel(), 1)?)?,
        )?;
        let value = self.budget.adopt(initial)?;
        let max = self.limits().max_bytes() / std::mem::size_of::<Parameter>();
        let replacement = self.parameters.growth(&self.budget, max)?;
        check_finite(value.tensor.data(), Stage::Parameter)?;
        self.parameters.push(
            replacement,
            Parameter {
                value,
                gradient: None,
            },
        );
        self.revision = revision;
        Ok(ParameterId {
            owner: self.owner,
            index,
        })
    }

    pub fn value(&self, id: ParameterId) -> Result<&Tensor, Error> {
        Ok(self.snapshot_source(id)?.1)
    }

    pub fn gradient(&self, id: ParameterId) -> Result<Option<&Tensor>, Error> {
        Ok(self
            .lookup(id)?
            .gradient
            .as_ref()
            .map(|value| &value.tensor))
    }

    /// Install explicit zero gradients as one transaction.
    pub fn zero_grad(&mut self) -> Result<(), Error> {
        let mut work = self.len();
        for parameter in &self.parameters.entries {
            work = add(work, parameter.value.tensor.numel())?;
        }
        self.limits().check_work(Pass::ZeroGrad, work)?;
        let mut staged = Table::allocate(&self.budget, self.len())?;
        for parameter in &self.parameters.entries {
            let value = &parameter.value.tensor;
            let zero = self.budget.tensor(value.rank(), value.numel(), || {
                Tensor::zeros(value.shape(), Default::default())
            })?;
            staged.entries.push(Some(zero));
        }
        for (parameter, gradient) in self.parameters.entries.iter_mut().zip(&mut staged.entries) {
            std::mem::swap(&mut parameter.gradient, gradient);
        }
        Ok(())
    }

    /// Consume a batch from this set at the current revision and commit all
    /// gradient additions together. An error also consumes the batch.
    pub fn accumulate_gradients(&mut self, batch: GradientBatch) -> Result<(), Error> {
        self.accumulate_gradients_with_work(batch).map(|_| ())
    }

    /// Accumulate transactionally and return the work charged on success.
    pub fn accumulate_gradients_with_work(
        &mut self,
        mut batch: GradientBatch,
    ) -> Result<usize, Error> {
        if batch.owner != self.owner {
            return Err(Error::WrongParameterSet);
        }
        if batch.revision != self.revision {
            return Err(Error::StaleGradientBatch {
                batch: batch.revision,
                current: self.revision,
            });
        }
        if batch.gradients.entries.len() != self.len() {
            return Err(Error::InvalidGradientInventory);
        }
        let mut work = mul(self.len(), 2)?;
        for (parameter, incoming) in self.parameters.entries.iter().zip(&batch.gradients.entries) {
            if let Some(incoming) = incoming {
                let tensor = &incoming.tensor;
                same_shape(parameter.value.tensor.shape(), tensor.shape())?;
                work = add(work, tensor.numel())?;
                if let Some(old) = &parameter.gradient {
                    same_shape(old.tensor.shape(), tensor.shape())?;
                    work = add(work, mul(tensor.numel(), 5)?)?;
                }
            }
        }
        self.limits().check_work(Pass::Accumulation, work)?;
        for (parameter, incoming) in self
            .parameters
            .entries
            .iter()
            .zip(&mut batch.gradients.entries)
        {
            if let Some(incoming) = incoming {
                let tensor = &incoming.tensor;
                check_finite(tensor.data(), Stage::Gradient)?;
                if let Some(old) = &parameter.gradient {
                    check_finite(old.tensor.data(), Stage::Gradient)?;
                    *incoming = self.budget.generate(tensor.shape(), tensor.numel(), |i| {
                        finite(
                            (f64::from(old.tensor.data()[i]) + f64::from(tensor.data()[i])) as f32,
                            Stage::Gradient,
                            i,
                        )
                    })?;
                }
            }
        }
        for (parameter, slot) in self
            .parameters
            .entries
            .iter_mut()
            .zip(&mut batch.gradients.entries)
        {
            if let Some(incoming) = slot.take() {
                parameter.gradient = Some(incoming);
            }
        }
        Ok(work)
    }

    pub(crate) fn batch(&self) -> Result<GradientBatch, Error> {
        let mut gradients = Table::allocate(&self.budget, self.len())?;
        gradients.entries.resize_with(self.len(), || None);
        Ok(GradientBatch {
            owner: self.owner,
            revision: self.revision,
            gradients,
        })
    }

    pub(crate) fn snapshot_source(&self, id: ParameterId) -> Result<(usize, &Tensor), Error> {
        Ok((id.index, &self.lookup(id)?.value.tensor))
    }

    fn lookup(&self, id: ParameterId) -> Result<&Parameter, Error> {
        if id.owner != self.owner {
            return Err(Error::InvalidParameterId);
        }
        self.parameters
            .entries
            .get(id.index)
            .ok_or(Error::InvalidParameterId)
    }

    fn next_revision(&self) -> Result<u64, Error> {
        self.revision.checked_add(1).ok_or(Error::RevisionExhausted)
    }
}

/// Stateless SGD with finite strictly positive binary32 learning rate.
#[derive(Debug)]
pub struct Sgd {
    learning_rate: f32,
}

impl Sgd {
    pub fn new(learning_rate: f32) -> Result<Self, Error> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(Error::InvalidLearningRate);
        }
        Ok(Self { learning_rate })
    }

    /// Stage every candidate before committing. A successful nonempty step
    /// advances the revision and clears all accumulated gradients.
    pub fn step(&self, parameters: &mut ParameterSet) -> Result<(), Error> {
        if parameters.is_empty() {
            return Ok(());
        }
        let revision = parameters.next_revision()?;
        let mut work = mul(parameters.len(), 3)?;
        for (index, parameter) in parameters.parameters.entries.iter().enumerate() {
            let (value, gradient) = Self::pair(index, parameter)?;
            same_shape(value.shape(), gradient.shape())?;
            work = add(work, mul(value.numel(), 6)?)?;
        }
        parameters.limits().check_work(Pass::Sgd, work)?;
        for (index, parameter) in parameters.parameters.entries.iter().enumerate() {
            let (value, gradient) = Self::pair(index, parameter)?;
            check_finite(value.data(), Stage::Parameter)?;
            check_finite(gradient.data(), Stage::Gradient)?;
        }
        let mut staged = Table::allocate(&parameters.budget, parameters.len())?;
        for (index, parameter) in parameters.parameters.entries.iter().enumerate() {
            let (value, gradient) = Self::pair(index, parameter)?;
            let next = parameters
                .budget
                .generate(value.shape(), value.numel(), |i| {
                    let delta = f64::from(self.learning_rate) * f64::from(gradient.data()[i]);
                    finite(
                        (f64::from(value.data()[i]) - delta) as f32,
                        Stage::Candidate,
                        i,
                    )
                })?;
            staged.entries.push(next);
        }
        for (parameter, slot) in parameters
            .parameters
            .entries
            .iter_mut()
            .zip(&mut staged.entries)
        {
            std::mem::swap(&mut parameter.value, slot);
            parameter.gradient = None;
        }
        parameters.revision = revision;
        Ok(())
    }

    fn pair(index: usize, parameter: &Parameter) -> Result<(&Tensor, &Tensor), Error> {
        let gradient = parameter
            .gradient
            .as_ref()
            .ok_or(Error::MissingGradient { parameter: index })?;
        Ok((&parameter.value.tensor, &gradient.tensor))
    }
}

/// Immutable binary32 AdamW hyperparameters. Arithmetic is widened to binary64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWConfig {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
}

impl AdamWConfig {
    pub fn new(beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Result<Self, Error> {
        let config = Self {
            beta1,
            beta2,
            epsilon,
            weight_decay,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<(), Error> {
        if !self.beta1.is_finite()
            || !self.beta2.is_finite()
            || !self.epsilon.is_finite()
            || !self.weight_decay.is_finite()
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
            || self.epsilon <= 0.0
            || self.weight_decay < 0.0
        {
            return Err(Error::InvalidOptimizerConfig);
        }
        Ok(())
    }

    pub fn beta1(self) -> f32 {
        self.beta1
    }
    pub fn beta2(self) -> f32 {
        self.beta2
    }
    pub fn epsilon(self) -> f32 {
        self.epsilon
    }
    pub fn weight_decay(self) -> f32 {
        self.weight_decay
    }
}

/// AdamW's persistent state is bound to one ParameterSet owner, revision, and
/// ordered rank/shape inventory. Moment tensors share the parameter ledger.
#[derive(Debug)]
pub struct AdamW {
    owner: u64,
    inventory: u64,
    expected_revision: u64,
    config: AdamWConfig,
    first: Table<OwnedTensor>,
    second: Table<OwnedTensor>,
    beta1_power: f64,
    beta2_power: f64,
    step: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWStepMetrics {
    pub step: u64,
    pub parameter_update_norm: f64,
}

impl AdamW {
    pub fn new(parameters: &ParameterSet, config: AdamWConfig) -> Result<Self, Error> {
        Self::build(parameters, config, 0, 1.0, 1.0, None, None)
    }

    /// Attach validated checkpoint moments to a fresh runtime owner.
    pub fn restore(
        parameters: &ParameterSet,
        config: AdamWConfig,
        step: u64,
        beta1_power: f64,
        beta2_power: f64,
        first: &[Vec<f32>],
        second: &[Vec<f32>],
    ) -> Result<Self, Error> {
        Self::build(
            parameters,
            config,
            step,
            beta1_power,
            beta2_power,
            Some(first),
            Some(second),
        )
    }

    fn build(
        parameters: &ParameterSet,
        config: AdamWConfig,
        step: u64,
        beta1_power: f64,
        beta2_power: f64,
        first_values: Option<&[Vec<f32>]>,
        second_values: Option<&[Vec<f32>]>,
    ) -> Result<Self, Error> {
        config.validate()?;
        if parameters.is_empty() {
            return Err(Error::EmptyParameterSet);
        }
        if !beta1_power.is_finite()
            || !beta2_power.is_finite()
            || !(0.0..=1.0).contains(&beta1_power)
            || !(0.0..=1.0).contains(&beta2_power)
        {
            return Err(Error::InvalidBiasPower);
        }
        if let (Some(first), Some(second)) = (first_values, second_values) {
            if first.len() != parameters.len() || second.len() != parameters.len() {
                return Err(Error::InvalidOptimizerState);
            }
        } else if first_values.is_some() || second_values.is_some() {
            return Err(Error::InvalidOptimizerState);
        }
        let mut first = Table::allocate(&parameters.budget, parameters.len())?;
        let mut second = Table::allocate(&parameters.budget, parameters.len())?;
        for (index, parameter) in parameters.parameters.entries.iter().enumerate() {
            let shape = parameter.value.tensor.shape();
            let len = parameter.value.tensor.numel();
            let first_source = first_values.map(|values| values[index].as_slice());
            let second_source = second_values.map(|values| values[index].as_slice());
            for (source, parameter_index) in [(first_source, 0), (second_source, 1)] {
                if let Some(values) = source {
                    if values.len() != len {
                        return Err(Error::InvalidOptimizerState);
                    }
                    check_finite(values, Stage::Candidate)?;
                    if parameter_index == 1 {
                        for (coordinate, &value) in values.iter().enumerate() {
                            if value < 0.0 {
                                return Err(Error::InvalidSecondMoment {
                                    parameter: index,
                                    index: coordinate,
                                });
                            }
                        }
                    }
                }
            }
            let first_tensor = parameters.budget.generate(shape, len, |i| {
                Ok(first_source.map_or(0.0, |values| values[i]))
            })?;
            let second_tensor = parameters.budget.generate(shape, len, |i| {
                Ok(second_source.map_or(0.0, |values| values[i]))
            })?;
            first.entries.push(first_tensor);
            second.entries.push(second_tensor);
        }
        Ok(Self {
            owner: parameters.owner,
            inventory: parameters.inventory_fingerprint(),
            expected_revision: parameters.revision,
            config,
            first,
            second,
            beta1_power,
            beta2_power,
            step,
        })
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }
    pub fn step_count(&self) -> u64 {
        self.step
    }
    pub fn beta1_power(&self) -> f64 {
        self.beta1_power
    }
    pub fn beta2_power(&self) -> f64 {
        self.beta2_power
    }

    pub fn first_moment(&self, parameter: usize) -> Option<&[f32]> {
        self.first
            .entries
            .get(parameter)
            .map(|tensor| tensor.tensor.data())
    }

    pub fn second_moment(&self, parameter: usize) -> Option<&[f32]> {
        self.second
            .entries
            .get(parameter)
            .map(|tensor| tensor.tensor.data())
    }

    pub fn expected_revision(&self) -> u64 {
        self.expected_revision
    }

    pub fn step(
        &mut self,
        parameters: &mut ParameterSet,
        learning_rate: f32,
    ) -> Result<AdamWStepMetrics, Error> {
        self.config.validate()?;
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(Error::InvalidLearningRate);
        }
        let rate = f64::from(learning_rate);
        let decay = f64::from(self.config.weight_decay);
        let decay_product = rate * decay;
        if !decay_product.is_finite() || decay_product >= 1.0 {
            return Err(Error::InvalidDecayProduct);
        }
        if self.owner != parameters.owner {
            return Err(Error::WrongParameterSet);
        }
        if self.inventory != parameters.inventory_fingerprint()
            || self.first.entries.len() != parameters.len()
            || self.second.entries.len() != parameters.len()
        {
            return Err(Error::InvalidOptimizerState);
        }
        if self.expected_revision != parameters.revision {
            return Err(Error::InvalidOptimizerState);
        }
        let next_step = self.step.checked_add(1).ok_or(Error::CounterExhausted)?;
        let next_revision = parameters.next_revision()?;
        let beta1 = f64::from(self.config.beta1);
        let beta2 = f64::from(self.config.beta2);
        let next_beta1_power = self.beta1_power * beta1;
        let next_beta2_power = self.beta2_power * beta2;
        let correction1 = 1.0 - next_beta1_power;
        let correction2 = 1.0 - next_beta2_power;
        if !next_beta1_power.is_finite()
            || !next_beta2_power.is_finite()
            || !correction1.is_finite()
            || !correction2.is_finite()
            || correction1 <= 0.0
            || correction2 <= 0.0
        {
            return Err(Error::InvalidBiasPower);
        }
        if parameters.len() != self.first.entries.len()
            || parameters.len() != self.second.entries.len()
        {
            return Err(Error::InvalidOptimizerState);
        }
        let mut work = mul(parameters.len(), 8)?;
        for (index, parameter) in parameters.parameters.entries.iter().enumerate() {
            let gradient = parameter
                .gradient
                .as_ref()
                .ok_or(Error::MissingGradient { parameter: index })?;
            same_shape(parameter.value.tensor.shape(), gradient.tensor.shape())?;
            same_shape(
                parameter.value.tensor.shape(),
                self.first.entries[index].tensor.shape(),
            )?;
            same_shape(
                parameter.value.tensor.shape(),
                self.second.entries[index].tensor.shape(),
            )?;
            work = add(work, mul(parameter.value.tensor.numel(), 24)?)?;
            check_finite(parameter.value.tensor.data(), Stage::Parameter)?;
            check_finite(gradient.tensor.data(), Stage::Gradient)?;
            check_finite(self.first.entries[index].tensor.data(), Stage::Candidate)?;
            check_finite(self.second.entries[index].tensor.data(), Stage::Candidate)?;
            for (coordinate, &value) in self.second.entries[index].tensor.data().iter().enumerate()
            {
                if value < 0.0 {
                    return Err(Error::InvalidSecondMoment {
                        parameter: index,
                        index: coordinate,
                    });
                }
            }
        }
        parameters.limits().check_work(Pass::AdamW, work)?;

        let mut staged_values = Table::allocate(&parameters.budget, parameters.len())?;
        let mut staged_first = Table::allocate(&parameters.budget, parameters.len())?;
        let mut staged_second = Table::allocate(&parameters.budget, parameters.len())?;
        let mut update_norm_squared = 0.0_f64;
        for (index, parameter) in parameters.parameters.entries.iter().enumerate() {
            let value = &parameter.value.tensor;
            let gradient = &parameter
                .gradient
                .as_ref()
                .expect("prevalidated gradient")
                .tensor;
            let old_first = &self.first.entries[index].tensor;
            let old_second = &self.second.entries[index].tensor;
            let next_first = parameters
                .budget
                .generate(value.shape(), value.numel(), |i| {
                    let left = beta1 * f64::from(old_first.data()[i]);
                    let one_minus_beta = 1.0 - beta1;
                    let right = one_minus_beta * f64::from(gradient.data()[i]);
                    finite((left + right) as f32, Stage::Candidate, i)
                })?;
            let next_second = parameters
                .budget
                .generate(value.shape(), value.numel(), |i| {
                    let left = beta2 * f64::from(old_second.data()[i]);
                    let gradient_wide = f64::from(gradient.data()[i]);
                    let square = gradient_wide * gradient_wide;
                    let one_minus_beta = 1.0 - beta2;
                    let right = one_minus_beta * square;
                    let rounded = finite((left + right) as f32, Stage::Candidate, i)?;
                    if rounded < 0.0 {
                        return Err(Error::InvalidSecondMoment {
                            parameter: index,
                            index: i,
                        });
                    }
                    Ok(rounded)
                })?;
            let next_value = parameters
                .budget
                .generate(value.shape(), value.numel(), |i| {
                    let m_hat = f64::from(next_first.tensor.data()[i]) / correction1;
                    let v_hat = f64::from(next_second.tensor.data()[i]) / correction2;
                    if !m_hat.is_finite() || !v_hat.is_finite() || v_hat < 0.0 {
                        return Err(Error::InvalidOptimizerState);
                    }
                    let denominator = v_hat.sqrt() + f64::from(self.config.epsilon);
                    if !denominator.is_finite() || denominator <= 0.0 {
                        return Err(Error::InvalidOptimizerState);
                    }
                    let decay_scale = if value.rank() == 2 {
                        1.0 - decay_product
                    } else {
                        1.0
                    };
                    let decayed = decay_scale * f64::from(value.data()[i]);
                    let adaptive = rate * m_hat / denominator;
                    let candidate = finite((decayed - adaptive) as f32, Stage::Candidate, i)?;
                    let delta = f64::from(candidate) - f64::from(value.data()[i]);
                    update_norm_squared += delta * delta;
                    Ok(candidate)
                })?;
            staged_values.entries.push(next_value);
            staged_first.entries.push(next_first);
            staged_second.entries.push(next_second);
        }
        let parameter_update_norm = update_norm_squared.sqrt();
        if !parameter_update_norm.is_finite() {
            return Err(Error::InvalidOptimizerState);
        }

        // No fallible operation follows: publish the three tables and counters together.
        for (index, parameter) in parameters.parameters.entries.iter_mut().enumerate() {
            std::mem::swap(&mut parameter.value, &mut staged_values.entries[index]);
            parameter.gradient = None;
            std::mem::swap(
                &mut self.first.entries[index],
                &mut staged_first.entries[index],
            );
            std::mem::swap(
                &mut self.second.entries[index],
                &mut staged_second.entries[index],
            );
        }
        parameters.revision = next_revision;
        self.expected_revision = next_revision;
        self.beta1_power = next_beta1_power;
        self.beta2_power = next_beta2_power;
        self.step = next_step;
        Ok(AdamWStepMetrics {
            step: next_step,
            parameter_update_norm,
        })
    }
}

/// Private FP64 token-weighted gradient window. It never writes partial sums to
/// ParameterSet gradients; `finish` publishes one normalized batch atomically.
#[derive(Debug)]
pub struct WeightedGradientAccumulator {
    owner: u64,
    inventory: u64,
    revision: u64,
    values: OwnedF64Table,
    total_supervised: u64,
    microbatches: usize,
    max_microbatches: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientClipMetrics {
    pub supervised_tokens: u64,
    pub pre_clip_norm: f64,
    pub post_clip_norm: f64,
    pub clipped: bool,
}

/// Explicit finite limits for repeated use of the exact TASK 005 candidate.
/// This is separate from the single-pass CandidateReferenceExperimentPolicy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateTrainingLimits {
    microbatches_per_update: usize,
    max_updates: u64,
    consumed_tokens: u64,
    supervised_tokens: u64,
    cumulative_work: u64,
    owned_memory_bytes: usize,
    max_runtime: Duration,
}

impl CandidateTrainingLimits {
    pub fn new(
        microbatches_per_update: usize,
        max_updates: u64,
        consumed_tokens: u64,
        supervised_tokens: u64,
        cumulative_work: u64,
        owned_memory_bytes: usize,
        max_runtime: Duration,
    ) -> Result<Self, Error> {
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
        {
            return Err(Error::InvalidTrainingPolicy);
        }
        Ok(Self {
            microbatches_per_update,
            max_updates,
            consumed_tokens,
            supervised_tokens,
            cumulative_work,
            owned_memory_bytes,
            max_runtime,
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
}

/// Run-scoped authorization and accounting for repeated candidate microbatches.
/// It binds once to the exact candidate inventory and carries cumulative
/// attempted work/token/runtime counters that can be checkpointed by the run.
#[derive(Debug)]
pub struct CandidateTrainingPolicy {
    limits: CandidateTrainingLimits,
    owner: Option<u64>,
    inventory: Option<u64>,
    consumed_tokens: u64,
    supervised_tokens: u64,
    cumulative_work: u64,
    completed_updates: u64,
    attempted_updates: u64,
    window_microbatches: usize,
    update_authorized: bool,
    prior_active: Duration,
    started: Instant,
}

impl CandidateTrainingPolicy {
    pub fn new(limits: CandidateTrainingLimits) -> Self {
        Self::restore(limits, 0, 0, 0, 0, 0, Duration::ZERO)
            .expect("zero progress is valid for validated limits")
    }

    pub fn restore(
        limits: CandidateTrainingLimits,
        consumed_tokens: u64,
        supervised_tokens: u64,
        cumulative_work: u64,
        completed_updates: u64,
        attempted_updates: u64,
        prior_active: Duration,
    ) -> Result<Self, Error> {
        if consumed_tokens > limits.consumed_tokens
            || supervised_tokens > limits.supervised_tokens
            || supervised_tokens > consumed_tokens
            || cumulative_work > limits.cumulative_work
            || completed_updates > limits.max_updates
            || attempted_updates > limits.max_updates
            || completed_updates > attempted_updates
            || prior_active > limits.max_runtime
        {
            return Err(Error::InvalidTrainingPolicy);
        }
        Ok(Self {
            limits,
            owner: None,
            inventory: None,
            consumed_tokens,
            supervised_tokens,
            cumulative_work,
            completed_updates,
            attempted_updates,
            window_microbatches: 0,
            update_authorized: false,
            prior_active,
            started: Instant::now(),
        })
    }

    pub fn limits(&self) -> CandidateTrainingLimits {
        self.limits
    }
    pub fn consumed_tokens(&self) -> u64 {
        self.consumed_tokens
    }
    pub fn supervised_tokens(&self) -> u64 {
        self.supervised_tokens
    }
    pub fn cumulative_work(&self) -> u64 {
        self.cumulative_work
    }
    pub fn completed_updates(&self) -> u64 {
        self.completed_updates
    }
    pub fn attempted_updates(&self) -> u64 {
        self.attempted_updates
    }
    pub fn window_microbatches(&self) -> usize {
        self.window_microbatches
    }
    pub fn active_runtime(&self) -> Duration {
        self.prior_active.saturating_add(self.started.elapsed())
    }

    /// Charge a complete fixed-shape microbatch before forward/backward starts.
    /// Bounds include the accepted ordinary forward ceiling, scoped reverse
    /// ceiling, and gradient transfer estimate, regardless of observed work.
    pub fn admit_microbatch(
        &mut self,
        parameters: &ParameterSet,
        shape: [usize; 2],
        consumed: u64,
        supervised: u64,
    ) -> Result<(), Error> {
        self.check_runtime()?;
        if shape != crate::CandidateReferenceExperimentPolicy::BATCH_SHAPE
            || consumed > (shape[0] * shape[1]) as u64
            || supervised > consumed
        {
            return Err(Error::InvalidTrainingPolicy);
        }
        crate::CandidateReferenceExperimentPolicy::validate_inventory(parameters)?;
        if parameters.accounted_bytes() > self.limits.owned_memory_bytes {
            return Err(Error::TrainingPolicyExhausted);
        }
        let inventory = parameters.inventory_fingerprint();
        match (self.owner, self.inventory) {
            (Some(owner), Some(fingerprint))
                if owner != parameters.owner || fingerprint != inventory =>
            {
                return Err(Error::WrongParameterSet);
            }
            (Some(_), Some(_)) => {}
            (None, None) => {
                self.owner = Some(parameters.owner);
                self.inventory = Some(inventory);
            }
            _ => return Err(Error::InvalidTrainingPolicy),
        }
        if self.window_microbatches >= self.limits.microbatches_per_update {
            return Err(Error::TrainingPolicyExhausted);
        }
        let next_consumed = self
            .consumed_tokens
            .checked_add(consumed)
            .ok_or(Error::CounterExhausted)?;
        let next_supervised = self
            .supervised_tokens
            .checked_add(supervised)
            .ok_or(Error::CounterExhausted)?;
        let microbatch_work = if supervised == 0 {
            0
        } else {
            (ResourceLimits::MAX_WORK as u64)
                .checked_add(crate::CandidateReferenceExperimentPolicy::BACKWARD_WORK as u64)
                .and_then(|work| work.checked_add((parameters.len() as u64).checked_mul(4)?))
                .ok_or(Error::CounterExhausted)?
        };
        let next_work = self
            .cumulative_work
            .checked_add(microbatch_work)
            .ok_or(Error::CounterExhausted)?;
        if next_consumed > self.limits.consumed_tokens
            || next_supervised > self.limits.supervised_tokens
            || next_work > self.limits.cumulative_work
        {
            return Err(Error::TrainingPolicyExhausted);
        }
        self.consumed_tokens = next_consumed;
        self.supervised_tokens = next_supervised;
        self.cumulative_work = next_work;
        self.window_microbatches += 1;
        Ok(())
    }

    /// Preflight and charge the bounded optimizer/normalization update work.
    pub fn authorize_update(&mut self, parameters: &ParameterSet) -> Result<(), Error> {
        self.check_runtime()?;
        if self.owner != Some(parameters.owner)
            || self.inventory != Some(parameters.inventory_fingerprint())
            || self.window_microbatches == 0
            || self.update_authorized
        {
            return Err(Error::InvalidTrainingPolicy);
        }
        let next_updates = self
            .attempted_updates
            .checked_add(1)
            .ok_or(Error::CounterExhausted)?;
        let update_work = (parameters.len() as u64)
            .checked_mul(30)
            .and_then(|v| {
                parameters
                    .parameters
                    .entries
                    .iter()
                    .try_fold(v, |total, parameter| {
                        total.checked_add((parameter.value.tensor.numel() as u64).checked_mul(30)?)
                    })
            })
            .ok_or(Error::CounterExhausted)?;
        let next_work = self
            .cumulative_work
            .checked_add(update_work)
            .ok_or(Error::CounterExhausted)?;
        if next_updates > self.limits.max_updates || next_work > self.limits.cumulative_work {
            return Err(Error::TrainingPolicyExhausted);
        }
        self.cumulative_work = next_work;
        self.attempted_updates = next_updates;
        self.update_authorized = true;
        Ok(())
    }

    /// Charge bounded non-training work (for example forward-only validation)
    /// against the same candidate lineage ceiling.
    pub fn charge_auxiliary_work(
        &mut self,
        parameters: &ParameterSet,
        work: u64,
    ) -> Result<(), Error> {
        self.check_runtime()?;
        crate::CandidateReferenceExperimentPolicy::validate_inventory(parameters)?;
        if self.owner.is_some_and(|owner| owner != parameters.owner)
            || self
                .inventory
                .is_some_and(|inventory| inventory != parameters.inventory_fingerprint())
        {
            return Err(Error::WrongParameterSet);
        }
        let next = self
            .cumulative_work
            .checked_add(work)
            .ok_or(Error::CounterExhausted)?;
        if next > self.limits.cumulative_work {
            return Err(Error::TrainingPolicyExhausted);
        }
        self.owner = Some(parameters.owner);
        self.inventory = Some(parameters.inventory_fingerprint());
        self.cumulative_work = next;
        Ok(())
    }

    /// Publish a completed optimizer transaction and open the next window.
    pub fn commit_update(&mut self) -> Result<(), Error> {
        if !self.update_authorized {
            return Err(Error::InvalidTrainingPolicy);
        }
        self.completed_updates = self
            .completed_updates
            .checked_add(1)
            .ok_or(Error::CounterExhausted)?;
        self.window_microbatches = 0;
        self.update_authorized = false;
        Ok(())
    }

    /// Drop an in-flight window after stop/failure; attempted counters remain charged.
    pub fn discard_window(&mut self) {
        self.window_microbatches = 0;
        self.update_authorized = false;
    }

    fn check_runtime(&self) -> Result<(), Error> {
        if self.active_runtime() > self.limits.max_runtime {
            return Err(Error::TrainingPolicyExhausted);
        }
        Ok(())
    }
}

impl WeightedGradientAccumulator {
    pub fn new(parameters: &ParameterSet, max_microbatches: usize) -> Result<Self, Error> {
        if max_microbatches == 0 || max_microbatches > ResourceLimits::MAX_EXPERIMENT_STEPS {
            return Err(Error::AccumulatorLimit {
                limit: ResourceLimits::MAX_EXPERIMENT_STEPS,
            });
        }
        let lengths = parameters
            .parameters
            .entries
            .iter()
            .map(|p| p.value.tensor.numel())
            .collect::<Vec<_>>();
        let values = parameters.budget.f64_table(&lengths)?;
        Ok(Self {
            owner: parameters.owner,
            inventory: parameters.inventory_fingerprint(),
            revision: parameters.revision,
            values,
            total_supervised: 0,
            microbatches: 0,
            max_microbatches,
        })
    }

    pub fn total_supervised_tokens(&self) -> u64 {
        self.total_supervised
    }
    pub fn microbatches(&self) -> usize {
        self.microbatches
    }

    /// Record a correctly admitted all-unsupervised microbatch without a
    /// backward pass or persistent parameter gradients.
    pub fn add_empty(&mut self, parameters: &ParameterSet) -> Result<(), Error> {
        if self.owner != parameters.owner {
            return Err(Error::WrongParameterSet);
        }
        if self.revision != parameters.revision {
            return Err(Error::StaleGradientBatch {
                batch: self.revision,
                current: parameters.revision,
            });
        }
        if self.inventory != parameters.inventory_fingerprint()
            || self.values.rows.len() != parameters.len()
        {
            return Err(Error::InvalidGradientInventory);
        }
        let next = self
            .microbatches
            .checked_add(1)
            .ok_or(Error::CounterExhausted)?;
        if next > self.max_microbatches {
            return Err(Error::AccumulatorLimit {
                limit: self.max_microbatches,
            });
        }
        self.microbatches = next;
        Ok(())
    }

    /// Consume `batch` on every outcome. All checks precede mutation of sums.
    pub fn add(
        &mut self,
        parameters: &ParameterSet,
        batch: GradientBatch,
        supervised: u64,
    ) -> Result<(), Error> {
        if self.owner != parameters.owner || batch.owner != parameters.owner {
            return Err(Error::WrongParameterSet);
        }
        if self.revision != parameters.revision || batch.revision != parameters.revision {
            return Err(Error::StaleGradientBatch {
                batch: batch.revision,
                current: parameters.revision,
            });
        }
        if self.inventory != parameters.inventory_fingerprint()
            || batch.gradients.entries.len() != parameters.len()
            || self.values.rows.len() != parameters.len()
        {
            return Err(Error::InvalidGradientInventory);
        }
        let next_microbatches = self
            .microbatches
            .checked_add(1)
            .ok_or(Error::CounterExhausted)?;
        if next_microbatches > self.max_microbatches {
            return Err(Error::AccumulatorLimit {
                limit: self.max_microbatches,
            });
        }
        let next_total = self
            .total_supervised
            .checked_add(supervised)
            .ok_or(Error::CounterExhausted)?;
        let mut work = mul(parameters.len(), 2)?;
        for (index, (parameter, incoming)) in parameters
            .parameters
            .entries
            .iter()
            .zip(&batch.gradients.entries)
            .enumerate()
        {
            if let Some(incoming) = incoming {
                same_shape(parameter.value.tensor.shape(), incoming.tensor.shape())?;
                if incoming.tensor.numel() != self.values.rows[index].len() {
                    return Err(Error::InvalidGradientInventory);
                }
                check_finite(incoming.tensor.data(), Stage::Gradient)?;
                work = add(work, mul(incoming.tensor.numel(), 2)?)?;
            }
        }
        parameters
            .limits()
            .check_work(Pass::WeightedAccumulation, work)?;

        if supervised != 0 {
            let count = supervised as f64;
            for (sum, incoming) in self.values.rows.iter_mut().zip(&batch.gradients.entries) {
                if let Some(incoming) = incoming {
                    for (accumulator, &gradient) in sum.iter_mut().zip(incoming.tensor.data()) {
                        let weighted = count * f64::from(gradient);
                        *accumulator += weighted;
                    }
                }
            }
        }
        self.total_supervised = next_total;
        self.microbatches = next_microbatches;
        Ok(())
    }

    pub fn finish(
        self,
        parameters: &ParameterSet,
        clip: Option<f32>,
    ) -> Result<(GradientBatch, GradientClipMetrics), Error> {
        if self.owner != parameters.owner {
            return Err(Error::WrongParameterSet);
        }
        if self.revision != parameters.revision {
            return Err(Error::StaleGradientBatch {
                batch: self.revision,
                current: parameters.revision,
            });
        }
        if self.inventory != parameters.inventory_fingerprint()
            || self.values.rows.len() != parameters.len()
        {
            return Err(Error::InvalidGradientInventory);
        }
        if self.total_supervised == 0 {
            return Err(Error::EmptySupervision);
        }
        if let Some(threshold) = clip {
            if !threshold.is_finite() || threshold <= 0.0 {
                return Err(Error::InvalidClipThreshold);
            }
        }
        let mut work = mul(parameters.len(), 3)?;
        for parameter in &parameters.parameters.entries {
            work = add(work, mul(parameter.value.tensor.numel(), 3)?)?;
        }
        parameters
            .limits()
            .check_work(Pass::WeightedAccumulation, work)?;
        let mut batch = parameters.batch()?;
        let total = self.total_supervised as f64;
        let mut norm_squared = 0.0_f64;
        for (index, parameter) in parameters.parameters.entries.iter().enumerate() {
            let values = &self.values.rows[index];
            if values.len() != parameter.value.tensor.numel() {
                return Err(Error::InvalidGradientInventory);
            }
            let normalized =
                parameters
                    .budget
                    .generate(parameter.value.tensor.shape(), values.len(), |i| {
                        finite((values[i] / total) as f32, Stage::Gradient, i)
                    })?;
            for &gradient in normalized.tensor.data() {
                let wide = f64::from(gradient);
                let square = wide * wide;
                norm_squared += square;
            }
            batch.gradients.entries[index] = Some(normalized);
        }
        let pre_clip_norm = norm_squared.sqrt();
        if !pre_clip_norm.is_finite() {
            return Err(Error::NonFinite {
                stage: Stage::Gradient,
                index: 0,
            });
        }
        let mut clipped = false;
        if let Some(threshold) = clip {
            let limit = f64::from(threshold);
            if pre_clip_norm > limit {
                let scale = limit / pre_clip_norm;
                let mut staged = Table::allocate(&parameters.budget, parameters.len())?;
                for gradient in &batch.gradients.entries {
                    let gradient = gradient.as_ref().expect("normalized inventory is complete");
                    staged.entries.push(Some(parameters.budget.generate(
                        gradient.tensor.shape(),
                        gradient.tensor.numel(),
                        |i| {
                            finite(
                                (f64::from(gradient.tensor.data()[i]) * scale) as f32,
                                Stage::Gradient,
                                i,
                            )
                        },
                    )?));
                }
                for (slot, replacement) in
                    batch.gradients.entries.iter_mut().zip(&mut staged.entries)
                {
                    *slot = replacement.take();
                }
                clipped = true;
            }
        }
        let mut post_squared = 0.0_f64;
        for gradient in batch.gradients.entries.iter().flatten() {
            for &value in gradient.tensor.data() {
                let wide = f64::from(value);
                let square = wide * wide;
                post_squared += square;
            }
        }
        let post_clip_norm = post_squared.sqrt();
        if !post_clip_norm.is_finite() {
            return Err(Error::NonFinite {
                stage: Stage::Gradient,
                index: 0,
            });
        }
        Ok((
            batch,
            GradientClipMetrics {
                supervised_tokens: self.total_supervised,
                pre_clip_norm,
                post_clip_norm,
                clipped,
            },
        ))
    }
}

#[cfg(test)]
mod task006_optimizer_tests {
    use super::*;

    const ORACLE: &str = include_str!("../../../tools/fixtures/task006-adamw.tsv");

    fn expected(name: &str) -> Vec<&'static str> {
        ORACLE
            .lines()
            .find(|line| line.starts_with(&format!("{name}\t")))
            .expect("frozen oracle row")
            .split('\t')
            .collect()
    }

    fn bits(value: f32) -> String {
        format!("{:08x}", value.to_bits())
    }
    fn bits64(value: f64) -> String {
        format!("{:016x}", value.to_bits())
    }

    fn parameters(values: &[f32], limits: ResourceLimits) -> ParameterSet {
        let mut set = ParameterSet::new(limits).unwrap();
        set.insert_generated(&[1, values.len()], |i| Ok(values[i]))
            .unwrap();
        set
    }

    fn gradient_batch(set: &ParameterSet, values: &[f32]) -> GradientBatch {
        let mut batch = set.batch().unwrap();
        let shape = set.parameters.entries[0].value.tensor.shape();
        batch.gradients.entries[0] = Some(
            set.budget
                .generate(shape, values.len(), |i| Ok(values[i]))
                .unwrap(),
        );
        batch
    }

    fn set_gradient(set: &mut ParameterSet, values: &[f32]) {
        let batch = gradient_batch(set, values);
        set.accumulate_gradients(batch).unwrap();
    }

    fn assert_oracle_state(name: &str, set: &ParameterSet, optimizer: &AdamW) {
        let fields = expected(name);
        assert_eq!(optimizer.step_count().to_string(), fields[1], "{name}");
        assert_eq!(bits64(optimizer.beta1_power()), fields[2], "{name}");
        assert_eq!(bits64(optimizer.beta2_power()), fields[3], "{name}");
        let value = set.value(set.id_at(0).unwrap()).unwrap();
        let first = optimizer.first_moment(0).unwrap();
        let second = optimizer.second_moment(0).unwrap();
        assert_eq!(value.numel(), first.len(), "{name}");
        assert_eq!(value.numel(), second.len(), "{name}");
        let mut index = 4;
        for coordinate in 0..value.numel() {
            assert_eq!(
                bits(value.data()[coordinate]),
                fields[index],
                "{name} parameter {coordinate}"
            );
            assert_eq!(
                bits(first[coordinate]),
                fields[index + 1],
                "{name} first moment {coordinate}"
            );
            assert_eq!(
                bits(second[coordinate]),
                fields[index + 2],
                "{name} second moment {coordinate}"
            );
            index += 3;
        }
        assert_eq!(index, fields.len(), "{name} fixture field count");
    }

    fn run_update(set: &mut ParameterSet, optimizer: &mut AdamW, gradient: &[f32], rate: f32) {
        set_gradient(set, gradient);
        optimizer.step(set, rate).unwrap();
    }

    #[test]
    fn matches_frozen_adamw_state_records() {
        let config = AdamWConfig::new(0.9, 0.999, 1e-8, 0.1).unwrap();
        let mut set = parameters(&[1.0], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, config).unwrap();
        run_update(&mut set, &mut optimizer, &[2.0], 0.01);
        assert_oracle_state("first_step", &set, &optimizer);

        let mut set = parameters(&[1.25, -0.5], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, config).unwrap();
        for (index, gradient) in [[2.0, -1.0], [-1.0, 0.5], [0.0, 0.0], [0.5, -2.0]]
            .iter()
            .enumerate()
        {
            run_update(&mut set, &mut optimizer, gradient, 0.01);
            assert_oracle_state(&format!("multi_{}", index + 1), &set, &optimizer);
        }

        let mut set = parameters(&[2.0], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, config).unwrap();
        run_update(&mut set, &mut optimizer, &[0.0], 0.1);
        assert_oracle_state("zero_gradient_with_decay", &set, &optimizer);

        let epsilon = AdamWConfig::new(0.0, 0.0, 0.1, 0.0).unwrap();
        let mut set = parameters(&[0.0], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, epsilon).unwrap();
        run_update(&mut set, &mut optimizer, &[0.02], 0.01);
        assert_eq!(
            bits(set.value(set.id_at(0).unwrap()).unwrap().data()[0]),
            expected("epsilon_outside_sqrt")[1]
        );
        assert_ne!(
            expected("epsilon_outside_sqrt")[1],
            expected("negative_control_epsilon_inside_sqrt")[1]
        );

        let decoupled = AdamWConfig::new(0.0, 0.0, 0.1, 0.5).unwrap();
        let mut set = parameters(&[1.0], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, decoupled).unwrap();
        run_update(&mut set, &mut optimizer, &[1.0], 0.1);
        assert_eq!(
            bits(set.value(set.id_at(0).unwrap()).unwrap().data()[0]),
            expected("decoupled_weight_decay")[1]
        );
        assert_ne!(
            expected("decoupled_weight_decay")[1],
            expected("negative_control_coupled_l2")[1]
        );
        let correct = f64::from(0.95_f32) - 0.1 / (1.0 + 0.1);
        let after_adaptive = (1.0 - 0.1 / (1.0 + 0.1)) * 0.95;
        let double_decay = 0.95 * 0.95 - 0.1 / (1.0 + 0.1);
        assert_eq!(bits(correct as f32), expected("decoupled_weight_decay")[1]);
        assert_eq!(
            bits(after_adaptive as f32),
            expected("negative_control_decay_after_adaptive")[1]
        );
        assert_eq!(
            bits(double_decay as f32),
            expected("negative_control_tied_embedding_double_decay")[1]
        );
        assert_ne!(
            bits(correct as f32),
            bits(double_decay as f32),
            "tied matrix must decay once"
        );

        let bias = AdamWConfig::new(0.5, 0.5, 1e-6, 0.0).unwrap();
        let mut set = parameters(&[0.5], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, bias).unwrap();
        run_update(&mut set, &mut optimizer, &[2.0], 0.1);
        let fields = expected("bias_corrected");
        assert_eq!(optimizer.step_count(), 1);
        assert_eq!(
            bits(set.value(set.id_at(0).unwrap()).unwrap().data()[0]),
            fields[1]
        );
        assert_eq!(bits(optimizer.first_moment(0).unwrap()[0]), fields[2]);
        assert_eq!(bits(optimizer.second_moment(0).unwrap()[0]), fields[3]);
        assert_ne!(
            expected("bias_corrected")[1],
            expected("negative_control_no_bias_correction")[1]
        );
    }

    #[test]
    fn weighted_accumulation_matches_oracle_and_discards_empty_windows() {
        let set = parameters(&[0.0, 0.0], ResourceLimits::default());
        let mut accumulator = WeightedGradientAccumulator::new(&set, 3).unwrap();
        accumulator
            .add(&set, gradient_batch(&set, &[2.0, 0.0]), 2)
            .unwrap();
        accumulator
            .add(&set, gradient_batch(&set, &[0.0, 3.0]), 1)
            .unwrap();
        accumulator.add_empty(&set).unwrap();
        let (batch, stats) = accumulator.finish(&set, Some(1.0)).unwrap();
        let id = set.id_at(0).unwrap();
        let gradient = batch.gradient(id).unwrap().unwrap().data();
        assert_eq!(
            bits(gradient[0]),
            expected("global_clip_below_threshold")[2]
        );
        assert_eq!(
            bits(gradient[1]),
            expected("global_clip_below_threshold")[3]
        );
        assert_eq!(
            bits(stats.pre_clip_norm as f32),
            expected("global_clip_below_threshold")[1]
        );
        assert!(stats.clipped);
        assert_eq!(stats.supervised_tokens, 3);
        assert_ne!(
            bits(gradient[0]),
            expected("negative_control_clip_wrong_mean")[2]
        );

        let mut normalized = WeightedGradientAccumulator::new(&set, 3).unwrap();
        normalized
            .add(&set, gradient_batch(&set, &[2.0, 0.0]), 2)
            .unwrap();
        normalized
            .add(&set, gradient_batch(&set, &[0.0, 3.0]), 1)
            .unwrap();
        normalized.add_empty(&set).unwrap();
        assert_eq!(normalized.microbatches(), 3);
        let (batch, stats) = normalized.finish(&set, None).unwrap();
        assert_eq!(stats.supervised_tokens, 3);
        assert_eq!(
            bits(batch.gradient(id).unwrap().unwrap().data()[0]),
            expected("token_weighted_mean")[1]
        );
        assert_eq!(
            bits(batch.gradient(id).unwrap().unwrap().data()[1]),
            expected("token_weighted_mean")[2]
        );

        let equal_threshold = f32::from_bits(
            u32::from_str_radix(expected("global_clip_below_threshold")[1], 16).unwrap(),
        );
        let accumulator = {
            let mut accumulator = WeightedGradientAccumulator::new(&set, 2).unwrap();
            accumulator
                .add(&set, gradient_batch(&set, &[2.0, 0.0]), 2)
                .unwrap();
            accumulator
                .add(&set, gradient_batch(&set, &[0.0, 3.0]), 1)
                .unwrap();
            accumulator
        };
        let (batch, stats) = accumulator.finish(&set, Some(equal_threshold)).unwrap();
        assert!(!stats.clipped);
        assert_eq!(
            bits(batch.gradient(id).unwrap().unwrap().data()[0]),
            expected("global_clip_at_threshold")[1]
        );
        let mut below = WeightedGradientAccumulator::new(&set, 1).unwrap();
        below
            .add(&set, gradient_batch(&set, &[2.0, 0.0]), 1)
            .unwrap();
        let (below_batch, below_stats) = below.finish(&set, Some(3.0)).unwrap();
        assert!(!below_stats.clipped);
        assert_eq!(
            bits(below_batch.gradient(id).unwrap().unwrap().data()[0]),
            bits(2.0)
        );
        assert_eq!(
            bits(below_batch.gradient(id).unwrap().unwrap().data()[1]),
            bits(0.0)
        );
        assert_eq!(
            expected("global_clip_above_threshold")[3],
            "False",
            "the oracle pins the norm-below-threshold control"
        );

        let mut extreme = WeightedGradientAccumulator::new(&set, 1).unwrap();
        extreme
            .add(&set, gradient_batch(&set, &[f32::MAX, -f32::MAX]), 1)
            .unwrap();
        let (extreme_batch, extreme_stats) = extreme.finish(&set, Some(1.0)).unwrap();
        let extreme_expected = expected("extreme_finite_gradient_clip");
        assert_eq!(bits64(extreme_stats.pre_clip_norm), extreme_expected[1]);
        assert_eq!(
            bits(extreme_batch.gradient(id).unwrap().unwrap().data()[0]),
            extreme_expected[2]
        );
        assert_eq!(
            bits(extreme_batch.gradient(id).unwrap().unwrap().data()[1]),
            extreme_expected[3]
        );
        assert!(extreme_stats.clipped);

        let empty = WeightedGradientAccumulator::new(&set, 1).unwrap();
        assert_eq!(
            empty.finish(&set, None).unwrap_err(),
            Error::EmptySupervision
        );
    }

    #[test]
    fn optimizer_rejections_preserve_committed_and_gradient_bits() {
        let config = AdamWConfig::new(0.9, 0.999, 1e-8, 0.1).unwrap();
        let mut set = parameters(&[1.0], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, config).unwrap();
        assert_eq!(
            optimizer.step(&mut set, 0.01),
            Err(Error::MissingGradient { parameter: 0 })
        );

        let malformed = set.budget.generate(&[1, 1], 1, |_| Ok(f32::NAN)).unwrap();
        set.parameters.entries[0].gradient = Some(malformed);
        let parameter_before = set.value(set.id_at(0).unwrap()).unwrap().data()[0].to_bits();
        let gradient_before =
            set.gradient(set.id_at(0).unwrap()).unwrap().unwrap().data()[0].to_bits();
        let first_before = optimizer.first_moment(0).unwrap()[0].to_bits();
        let second_before = optimizer.second_moment(0).unwrap()[0].to_bits();
        assert!(matches!(
            optimizer.step(&mut set, 0.01),
            Err(Error::NonFinite {
                stage: Stage::Gradient,
                ..
            })
        ));
        assert_eq!(
            set.value(set.id_at(0).unwrap()).unwrap().data()[0].to_bits(),
            parameter_before
        );
        assert_eq!(
            set.gradient(set.id_at(0).unwrap()).unwrap().unwrap().data()[0].to_bits(),
            gradient_before
        );
        assert_eq!(
            optimizer.first_moment(0).unwrap()[0].to_bits(),
            first_before
        );
        assert_eq!(
            optimizer.second_moment(0).unwrap()[0].to_bits(),
            second_before
        );

        let mut foreign = parameters(&[1.0], ResourceLimits::default());
        let mut foreign_optimizer = AdamW::new(&set, config).unwrap();
        assert_eq!(foreign_optimizer.owner, set.owner);
        assert_eq!(
            foreign_optimizer.step(&mut foreign, 0.01).unwrap_err(),
            Error::WrongParameterSet
        );

        let mut stale = parameters(&[1.0], ResourceLimits::default());
        let mut stale_optimizer = AdamW::new(&stale, config).unwrap();
        set_gradient(&mut stale, &[0.25]);
        stale.revision = stale.revision.checked_add(1).unwrap();
        let parameter_bits = stale.value(stale.id_at(0).unwrap()).unwrap().data()[0].to_bits();
        let gradient_bits = stale
            .gradient(stale.id_at(0).unwrap())
            .unwrap()
            .unwrap()
            .data()[0]
            .to_bits();
        assert_eq!(
            stale_optimizer.step(&mut stale, 0.01),
            Err(Error::InvalidOptimizerState)
        );
        assert_eq!(
            stale.value(stale.id_at(0).unwrap()).unwrap().data()[0].to_bits(),
            parameter_bits
        );
        assert_eq!(
            stale
                .gradient(stale.id_at(0).unwrap())
                .unwrap()
                .unwrap()
                .data()[0]
                .to_bits(),
            gradient_bits
        );

        let mut overflow = parameters(&[1.0], ResourceLimits::default());
        let mut overflow_optimizer = AdamW::new(&overflow, config).unwrap();
        set_gradient(&mut overflow, &[f32::MAX]);
        let before = overflow.value(overflow.id_at(0).unwrap()).unwrap().data()[0].to_bits();
        assert!(matches!(
            overflow_optimizer.step(&mut overflow, 0.01),
            Err(Error::NonFinite {
                stage: Stage::Candidate,
                ..
            })
        ));
        assert_eq!(
            overflow.value(overflow.id_at(0).unwrap()).unwrap().data()[0].to_bits(),
            before
        );
        assert_eq!(overflow_optimizer.step_count(), 0);
        assert!(overflow
            .gradient(overflow.id_at(0).unwrap())
            .unwrap()
            .is_some());
    }

    #[test]
    fn rank_two_matrices_decay_once_and_rank_one_gains_are_excluded() {
        let mut set = ParameterSet::new(ResourceLimits::default()).unwrap();
        let matrix = set.insert_generated(&[1, 1], |_| Ok(2.0)).unwrap();
        let gain = set.insert_generated(&[1], |_| Ok(2.0)).unwrap();
        let mut batch = set.batch().unwrap();
        batch.gradients.entries[0] = Some(set.budget.generate(&[1, 1], 1, |_| Ok(0.0)).unwrap());
        batch.gradients.entries[1] = Some(set.budget.generate(&[1], 1, |_| Ok(0.0)).unwrap());
        set.accumulate_gradients(batch).unwrap();
        let mut optimizer =
            AdamW::new(&set, AdamWConfig::new(0.9, 0.999, 1e-8, 0.1).unwrap()).unwrap();
        optimizer.step(&mut set, 0.1).unwrap();
        assert_eq!(
            set.value(matrix).unwrap().data()[0].to_bits(),
            1.98_f32.to_bits()
        );
        assert_eq!(
            set.value(gain).unwrap().data()[0].to_bits(),
            2.0_f32.to_bits()
        );
    }

    #[test]
    fn candidate_training_policy_is_finite_and_rejects_generic_inventories() {
        let limits = CandidateTrainingLimits::new(
            2,
            10_000,
            320_000,
            320_000,
            1_000_000_000,
            ResourceLimits::MAX_BYTES,
            std::time::Duration::from_secs(60),
        )
        .unwrap();
        assert!(CandidateTrainingLimits::new(
            1,
            10_001,
            320_000,
            320_000,
            1_000_000_000,
            ResourceLimits::MAX_BYTES,
            std::time::Duration::from_secs(60),
        )
        .is_err());
        let mut policy = CandidateTrainingPolicy::new(limits);
        let set = parameters(&[1.0], ResourceLimits::default());
        assert_eq!(
            policy.admit_microbatch(&set, [1, 32], 32, 32),
            Err(Error::InvalidCandidateInventory)
        );
        assert_eq!(policy.consumed_tokens(), 0);
        assert_eq!(policy.supervised_tokens(), 0);
        assert_eq!(policy.cumulative_work(), 0);
        assert!(CandidateTrainingLimits::new(
            1,
            1,
            0,
            0,
            1,
            ResourceLimits::MAX_BYTES,
            std::time::Duration::from_secs(1),
        )
        .is_err());
    }

    #[test]
    fn staging_faults_release_memory_and_preserve_state() {
        let config = AdamWConfig::new(0.9, 0.999, 1e-8, 0.1).unwrap();
        let mut set = parameters(&[1.0, 2.0], ResourceLimits::default());
        let mut optimizer = AdamW::new(&set, config).unwrap();
        set_gradient(&mut set, &[0.5, -0.25]);
        let value_before = set.value(set.id_at(0).unwrap()).unwrap().data().to_vec();
        let grad_before = set
            .gradient(set.id_at(0).unwrap())
            .unwrap()
            .unwrap()
            .data()
            .to_vec();
        let first_before = optimizer.first_moment(0).unwrap().to_vec();
        let bytes = set.accounted_bytes();
        set.budget.fail_allocation_after(3);
        assert_eq!(optimizer.step(&mut set, 0.01), Err(Error::AllocationFailed));
        assert_eq!(set.accounted_bytes(), bytes);
        assert_eq!(
            set.value(set.id_at(0).unwrap()).unwrap().data(),
            value_before
        );
        assert_eq!(
            set.gradient(set.id_at(0).unwrap()).unwrap().unwrap().data(),
            grad_before
        );
        assert_eq!(optimizer.first_moment(0).unwrap(), first_before);
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn accumulator_rejections_are_atomic_and_budgeted() {
        let mut set = parameters(&[0.0, 0.0], ResourceLimits::default());
        let mut accumulator = WeightedGradientAccumulator::new(&set, 1).unwrap();
        accumulator
            .add(&set, gradient_batch(&set, &[1.0, 2.0]), 1)
            .unwrap();
        let sums = accumulator.values.rows.clone();
        assert_eq!(
            accumulator.add(&set, gradient_batch(&set, &[3.0, 4.0]), 1),
            Err(Error::AccumulatorLimit { limit: 1 })
        );
        assert_eq!(accumulator.values.rows, sums);
        assert_eq!(accumulator.total_supervised, 1);
        assert_eq!(accumulator.microbatches, 1);

        let foreign = parameters(&[0.0, 0.0], ResourceLimits::default());
        let foreign_batch = gradient_batch(&foreign, &[1.0, 2.0]);
        assert_eq!(
            accumulator.add(&set, foreign_batch, 1),
            Err(Error::WrongParameterSet)
        );
        assert_eq!(accumulator.values.rows, sums);

        let mut stale = WeightedGradientAccumulator::new(&set, 2).unwrap();
        let stale_batch = gradient_batch(&set, &[1.0, 2.0]);
        set.revision = set.revision.checked_add(1).unwrap();
        assert!(matches!(
            stale.add(&set, stale_batch, 1),
            Err(Error::StaleGradientBatch { .. })
        ));
        assert_eq!(stale.total_supervised, 0);
        assert_eq!(stale.microbatches, 0);

        let mut overflow = WeightedGradientAccumulator::new(&foreign, 2).unwrap();
        overflow.total_supervised = u64::MAX;
        let before = overflow.values.rows.clone();
        assert_eq!(
            overflow.add(&foreign, gradient_batch(&foreign, &[1.0, 2.0]), 1),
            Err(Error::CounterExhausted)
        );
        assert_eq!(overflow.values.rows, before);
        assert_eq!(overflow.total_supervised, u64::MAX);
        assert_eq!(overflow.microbatches, 0);

        let mut count_overflow = WeightedGradientAccumulator::new(&foreign, 2).unwrap();
        count_overflow.microbatches = usize::MAX;
        assert_eq!(
            count_overflow.add_empty(&foreign),
            Err(Error::CounterExhausted)
        );
        assert_eq!(count_overflow.microbatches, usize::MAX);

        let mut non_finite = WeightedGradientAccumulator::new(&foreign, 2).unwrap();
        let before = non_finite.values.rows.clone();
        assert!(matches!(
            non_finite.add(&foreign, gradient_batch(&foreign, &[f32::NAN, 1.0]), 1),
            Err(Error::NonFinite {
                stage: Stage::Gradient,
                ..
            })
        ));
        assert_eq!(non_finite.values.rows, before);
        assert_eq!(non_finite.total_supervised, 0);
        assert_eq!(non_finite.microbatches, 0);

        let generous = parameters(&[1.0], ResourceLimits::default());
        let exact_bytes = generous.accounted_bytes();
        let tight = parameters(&[1.0], ResourceLimits::new(exact_bytes, 10, 10, 100));
        assert_eq!(tight.accounted_bytes(), exact_bytes);
        assert!(WeightedGradientAccumulator::new(&tight, 1).is_err());
        assert_eq!(tight.accounted_bytes(), exact_bytes);
    }

    #[test]
    fn restore_rejects_negative_second_moments_and_counter_exhaustion_is_atomic() {
        let mut set = parameters(&[1.0], ResourceLimits::default());
        let config = AdamWConfig::new(0.9, 0.999, 1e-8, 0.0).unwrap();
        assert!(matches!(
            AdamW::restore(&set, config, 0, 1.0, 1.0, &[vec![0.0]], &[vec![-1.0]]),
            Err(Error::InvalidSecondMoment { .. })
        ));

        let mut optimizer = AdamW::new(&set, config).unwrap();
        optimizer.step = u64::MAX;
        set_gradient(&mut set, &[0.0]);
        let before = set.value(set.id_at(0).unwrap()).unwrap().data()[0].to_bits();
        assert_eq!(optimizer.step(&mut set, 0.01), Err(Error::CounterExhausted));
        assert_eq!(
            set.value(set.id_at(0).unwrap()).unwrap().data()[0].to_bits(),
            before
        );
        assert!(set.gradient(set.id_at(0).unwrap()).unwrap().is_some());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f32) -> Tensor {
        Tensor::from_data(&[], &[value], Default::default()).unwrap()
    }

    fn pair_with_gradients(staging_bytes: usize) -> (ParameterSet, [ParameterId; 2]) {
        let base = 2 * size_of::<Parameter>() + 4 * size_of::<f32>();
        let limit = base + staging_bytes + size_of::<f32>();
        let mut set = ParameterSet::new(ResourceLimits::new(limit, 10, 10, 100)).unwrap();
        let ids = [
            set.insert(scalar(1.0)).unwrap(),
            set.insert(scalar(2.0)).unwrap(),
        ];
        for (entry, value) in set.parameters.entries.iter_mut().zip([3.0, 4.0]) {
            entry.gradient = Some(set.budget.adopt(scalar(value)).unwrap());
        }
        assert_eq!(set.accounted_bytes(), base);
        (set, ids)
    }

    #[test]
    fn later_zero_grad_staging_failure_preserves_everything() {
        let (mut set, ids) = pair_with_gradients(2 * size_of::<Option<OwnedTensor>>());
        let bytes = set.accounted_bytes();
        let limit = set.limits().max_bytes();
        assert_eq!(
            set.zero_grad(),
            Err(Error::MemoryLimit {
                limit,
                requested: limit + size_of::<f32>()
            })
        );
        assert_eq!(set.accounted_bytes(), bytes);
        assert_eq!(set.revision(), 2);
        for (id, (value, gradient)) in ids.into_iter().zip([(1.0, 3.0), (2.0, 4.0)]) {
            assert_eq!(set.value(id).unwrap().data(), [value]);
            assert_eq!(set.gradient(id).unwrap().unwrap().data(), [gradient]);
        }
    }

    #[test]
    fn later_sgd_staging_failure_releases_every_candidate() {
        let (mut set, ids) = pair_with_gradients(2 * size_of::<OwnedTensor>());
        let bytes = set.accounted_bytes();
        let limit = set.limits().max_bytes();
        assert_eq!(
            Sgd::new(1.0).unwrap().step(&mut set),
            Err(Error::MemoryLimit {
                limit,
                requested: limit + size_of::<f32>()
            })
        );
        assert_eq!(set.accounted_bytes(), bytes);
        assert_eq!(set.revision(), 2);
        for (id, (value, gradient)) in ids.into_iter().zip([(1.0, 3.0), (2.0, 4.0)]) {
            assert_eq!(set.value(id).unwrap().data(), [value]);
            assert_eq!(set.gradient(id).unwrap().unwrap().data(), [gradient]);
        }
    }

    #[test]
    fn invalid_inventory_shape_finiteness_and_revision_are_rejected() {
        let mut set = ParameterSet::new(ResourceLimits::default()).unwrap();
        let id = set.insert(scalar(1.0)).unwrap();
        let invalid = ParameterId { index: 1, ..id };
        assert!(matches!(set.value(invalid), Err(Error::InvalidParameterId)));
        let mut malformed = set.batch().unwrap();
        malformed.gradients.entries.clear();
        assert_eq!(
            set.accumulate_gradients(malformed),
            Err(Error::InvalidGradientInventory)
        );
        let mut malformed = set.batch().unwrap();
        let vector = Tensor::from_data(&[1], &[1.0], Default::default()).unwrap();
        malformed.gradients.entries[0] = Some(set.budget.adopt(vector).unwrap());
        assert!(matches!(
            set.accumulate_gradients(malformed),
            Err(Error::Tensor(_))
        ));
        let mut malformed = set.batch().unwrap();
        malformed.gradients.entries[0] = Some(set.budget.adopt(scalar(f32::NAN)).unwrap());
        assert!(matches!(
            set.accumulate_gradients(malformed),
            Err(Error::NonFinite {
                stage: Stage::Gradient,
                ..
            })
        ));
        let opt = Sgd::new(1.0).unwrap();
        let vector = Tensor::from_data(&[1], &[1.0], Default::default()).unwrap();
        set.parameters.entries[0].gradient = Some(set.budget.adopt(vector).unwrap());
        assert!(matches!(opt.step(&mut set), Err(Error::Tensor(_))));
        set.parameters.entries[0].gradient = Some(set.budget.adopt(scalar(f32::NAN)).unwrap());
        assert!(matches!(
            opt.step(&mut set),
            Err(Error::NonFinite {
                stage: Stage::Gradient,
                ..
            })
        ));
        set.zero_grad().unwrap();
        set.revision = u64::MAX;
        assert_eq!(opt.step(&mut set), Err(Error::RevisionExhausted));
        assert_eq!(set.insert(scalar(0.0)), Err(Error::RevisionExhausted));
        assert_eq!(set.value(id).unwrap().data(), [1.0]);
        assert_eq!(set.gradient(id).unwrap().unwrap().data(), [0.0]);
        assert_eq!(set.revision(), u64::MAX);
    }
}
