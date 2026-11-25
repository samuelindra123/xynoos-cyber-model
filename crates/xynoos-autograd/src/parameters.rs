use xynoos_tensor::Tensor;

use crate::budget::{add, mul, Budget, OwnedTensor, Table};
use crate::error::{check_finite, finite, same_shape};
use crate::{identity, AutogradError as Error, Pass, ResourceLimits, Stage};

/// Stable handle to a parameter in its originating set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterId {
    owner: u64,
    pub(crate) index: usize,
}

/// Exclusive ownership of parameter values and their persistent gradients.
#[derive(Debug)]
pub struct ParameterSet {
    pub(crate) budget: Budget,
    owner: u64,
    revision: u64,
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

impl ParameterSet {
    pub fn new(limits: ResourceLimits) -> Result<Self, Error> {
        Ok(Self {
            owner: identity()?,
            revision: 0,
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

    pub fn len(&self) -> usize {
        self.parameters.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn revision(&self) -> u64 {
        self.revision
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
        Ok(&self.lookup(id)?.value.tensor)
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
    pub fn accumulate_gradients(&mut self, mut batch: GradientBatch) -> Result<(), Error> {
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
        Ok(())
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
