use xynoos_tensor::{Tensor, TensorLimits};

use crate::budget::{add, mul, OwnedTensor, Table};
use crate::error::{check_finite, finite, same_shape};
use crate::kernels::Op;
use crate::reference::{Kernel, Reference, ReferenceBatch};
use crate::{
    identity, AutogradError as Error, GradientBatch, ParameterId, ParameterSet, Pass, Stage,
};

/// Stable value handle scoped to one tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueId {
    owner: u64,
    index: usize,
}

#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) value: OwnedTensor,
    op: Op,
    differentiable: bool,
    reachable: bool,
}

#[derive(Debug)]
enum State {
    Recording,
    Failed,
    Complete(Table<Option<OwnedTensor>>),
}

/// Explicit single-pass tape with an immutable parameter-set borrow.
///
/// ```compile_fail
/// fn mutate_while_recording(set: &mut xynoos_autograd::ParameterSet) {
///     let tape = xynoos_autograd::Tape::new(set).unwrap();
///     set.zero_grad().unwrap();
///     drop(tape);
/// }
/// ```
#[derive(Debug)]
pub struct Tape<'p> {
    parameters: &'p ParameterSet,
    owner: u64,
    nodes: Table<Node>,
    operations: usize,
    forward_work: usize,
    state: State,
    reference: Option<Reference>,
    backward_work: usize,
    snapshot_work: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeMetrics {
    pub values: usize,
    pub operations: usize,
    pub forward_work: usize,
    pub backward_work: usize,
    /// Parameter snapshot work already included in forward_work.
    pub snapshot_work: usize,
    pub gradient_transfer_work: usize,
}

impl<'p> Tape<'p> {
    pub fn new(parameters: &'p ParameterSet) -> Result<Self, Error> {
        Ok(Self {
            parameters,
            owner: identity()?,
            nodes: Table::empty(),
            operations: 0,
            forward_work: 0,
            state: State::Recording,
            reference: None,
            backward_work: 0,
            snapshot_work: 0,
        })
    }

    /// Initialize a reference graph with owned, charged next-token batch constants.
    pub fn for_reference(
        parameters: &'p ParameterSet,
        batch: ReferenceBatch<'_>,
        vocabulary: usize,
        heads: usize,
    ) -> Result<Self, Error> {
        let mut tape = Self::new(parameters)?;
        let (reference, work) = Reference::new(&parameters.budget, batch, vocabulary, heads)?;
        tape.reference = Some(reference);
        tape.forward_work = work;
        Ok(tape)
    }

    pub fn metrics(&self) -> Result<TapeMetrics, Error> {
        Ok(TapeMetrics {
            values: self.nodes.entries.len(),
            operations: self.operations,
            forward_work: self.forward_work,
            backward_work: self.backward_work,
            snapshot_work: self.snapshot_work,
            gradient_transfer_work: match &self.state {
                State::Complete(adjoints) => self.transfer_work(adjoints)?,
                _ => 0,
            },
        })
    }

    /// Masked embedding gather from a single `[V,D]` parameter snapshot.
    pub fn embedding(&mut self, embedding: ValueId) -> Result<ValueId, Error> {
        self.record_reference(Kernel::Gather(self.index(embedding)?))
    }

    /// Dense projection of `[B,T,D]` with `[D,N]` weights, or `[N,D]` when transposed.
    pub fn project(
        &mut self,
        input: ValueId,
        weight: ValueId,
        transposed: bool,
    ) -> Result<ValueId, Error> {
        self.record_reference(Kernel::Dense(
            self.index(input)?,
            self.index(weight)?,
            transposed,
        ))
    }

    /// Featurewise RMS normalization and masking of invalid rows.
    pub fn rms_norm(&mut self, input: ValueId, gain: ValueId) -> Result<ValueId, Error> {
        self.record_reference(Kernel::Norm(self.index(input)?, self.index(gain)?))
    }

    /// Interleaved RoPE with constant base 10,000 and per-sample positions.
    pub fn rope(&mut self, input: ValueId) -> Result<ValueId, Error> {
        self.record_reference(Kernel::Rope(self.index(input)?))
    }

    /// Explicit causal multi-head attention with a binary64 probability table.
    pub fn causal_attention(
        &mut self,
        q: ValueId,
        k: ValueId,
        v: ValueId,
    ) -> Result<ValueId, Error> {
        self.record_reference(Kernel::Attention(
            self.index(q)?,
            self.index(k)?,
            self.index(v)?,
        ))
    }

    pub fn swiglu(&mut self, gate: ValueId, value: ValueId) -> Result<ValueId, Error> {
        self.record_reference(Kernel::SwiGlu(self.index(gate)?, self.index(value)?))
    }

    pub fn masked_residual(&mut self, input: ValueId, branch: ValueId) -> Result<ValueId, Error> {
        self.record_reference(Kernel::Residual(self.index(input)?, self.index(branch)?))
    }

    pub fn next_token_loss(&mut self, logits: ValueId) -> Result<ValueId, Error> {
        self.record_reference(Kernel::Loss(self.index(logits)?))
    }

    fn reference(&self) -> Result<&Reference, Error> {
        self.reference.as_ref().ok_or(Error::InvalidReferenceShape)
    }

    fn record_reference(&mut self, kernel: Kernel) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        let limits = self.parameters.limits();
        if self.operations == limits.operations {
            return Err(Error::OperationLimit {
                limit: limits.operations,
            });
        }
        let reference = self.reference()?;
        let (shape, rank, len, work) = reference.output(kernel, &self.nodes.entries)?;
        let work = add(add(work, reference.nodes.move_work())?, 1)?;
        let replacement = self.reserve_node(work)?;
        let reference = self.reference()?;
        let saved_nodes = reference
            .nodes
            .growth(&self.parameters.budget, limits.operations)?;
        let (value, recorded) = reference.forward(
            &self.parameters.budget,
            kernel,
            &self.nodes.entries,
            &shape[..rank],
            len,
        )?;
        check_finite(value.tensor.data(), Stage::Forward)?;
        let differentiable = kernel
            .edges()
            .into_iter()
            .flatten()
            .any(|i| self.nodes.entries[i].differentiable);
        let reference = self
            .reference
            .as_mut()
            .ok_or(Error::InvalidReferenceShape)?;
        let op = Op::Reference(reference.nodes.entries.len());
        reference.nodes.push(saved_nodes, recorded);
        self.operations = add(self.operations, 1)?;
        Ok(self.publish(replacement, op, value, differentiable))
    }

    fn edges(&self, op: Op) -> [Option<usize>; 3] {
        if let Op::Reference(i) = op {
            return self.reference.as_ref().unwrap().nodes.entries[i]
                .kernel
                .edges();
        }
        let [a, b] = op.edges();
        [a, b, None]
    }

    pub fn value(&self, id: ValueId) -> Result<&Tensor, Error> {
        Ok(&self.nodes.entries[self.index(id)?].value.tensor)
    }

    pub fn input(&mut self, value: &Tensor) -> Result<ValueId, Error> {
        self.record_leaf(Op::Input, value)
    }

    pub fn parameter(&mut self, id: ParameterId) -> Result<ValueId, Error> {
        let (index, source) = self.parameters.snapshot_source(id)?;
        let cost = mul(source.numel(), 2)?;
        let next = add(self.snapshot_work, cost)?;
        let id = self.record_leaf(Op::Parameter(index), source)?;
        self.snapshot_work = next;
        Ok(id)
    }

    pub fn add(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, Error> {
        self.binary(a, b, Op::Add)
    }

    pub fn subtract(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, Error> {
        self.binary(a, b, Op::Subtract)
    }

    pub fn multiply(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, Error> {
        self.binary(a, b, Op::Multiply)
    }

    pub fn scale(&mut self, input: ValueId, factor: f32) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        finite(factor, Stage::Scale, 0)?;
        self.operation(Op::Scale(self.index(input)?, factor))
    }

    pub fn sum(&mut self, input: ValueId) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        self.operation(Op::Sum(self.index(input)?))
    }

    pub fn mean(&mut self, input: ValueId) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        self.operation(Op::Mean(self.index(input)?))
    }

    pub fn dot(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, Error> {
        self.binary(a, b, Op::Dot)
    }

    pub fn matmul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, Error> {
        self.binary(a, b, Op::Matmul)
    }

    pub fn transpose(&mut self, input: ValueId) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        self.operation(Op::Transpose(self.index(input)?))
    }

    /// Nonempty equal-shape mean squared error with a constant input leaf target.
    pub fn mse(&mut self, prediction: ValueId, target: ValueId) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        let prediction = self.index(prediction)?;
        let target = self.index(target)?;
        if !matches!(self.nodes.entries[target].op, Op::Input) {
            return Err(Error::ConstantTargetRequired);
        }
        self.operation(Op::Mse(prediction, target))
    }

    /// Run backward with seed 1.
    pub fn backward(&mut self, root: ValueId) -> Result<(), Error> {
        self.backward_with_seed(root, 1.0)
    }

    /// Execute one deterministic reverse pass. Any error seals the tape.
    pub fn backward_with_seed(&mut self, root: ValueId, seed: f32) -> Result<(), Error> {
        if !matches!(self.state, State::Recording) {
            return Err(Error::BackwardAlreadyRun);
        }
        self.state = State::Failed;
        let root = self.index(root)?;
        let rank = self.nodes.entries[root].value.tensor.rank();
        if rank != 0 {
            return Err(Error::NonScalarRoot { rank });
        }
        finite(seed, Stage::Seed, 0)?;
        self.plan_backward(root)?;
        let adjoints = if self.reference.is_some() {
            self.run_reference_backward(root, seed)?
        } else {
            self.run_backward(root, seed)?
        };
        self.state = State::Complete(adjoints);
        Ok(())
    }

    /// Read a successful adjoint. Constants and disconnected values return None.
    pub fn gradient(&self, id: ValueId) -> Result<Option<&Tensor>, Error> {
        let index = self.index(id)?;
        let State::Complete(adjoints) = &self.state else {
            return Err(Error::BackwardRequired);
        };
        Ok(adjoints.entries[index]
            .as_ref()
            .map(|gradient| &gradient.tensor))
    }

    /// End the tape's borrow and move parameter adjoints into a single-use batch.
    /// Multiple snapshots of one parameter are merged in reverse insertion order.
    pub fn into_parameter_gradients(mut self) -> Result<GradientBatch, Error> {
        let State::Complete(adjoints) = &self.state else {
            return Err(Error::BackwardRequired);
        };
        let work = self.transfer_work(adjoints)?;
        let State::Complete(adjoints) = &mut self.state else {
            return Err(Error::BackwardRequired);
        };
        self.parameters
            .limits()
            .check_work(Pass::GradientTransfer, work)?;
        let mut batch = self.parameters.batch()?;
        for (node, gradient) in self.nodes.entries.iter().zip(&mut adjoints.entries).rev() {
            if let Op::Parameter(index) = node.op {
                let Some(gradient) = gradient.take() else {
                    continue;
                };
                let slot = &mut batch.gradients.entries[index];
                if let Some(existing) = slot {
                    same_shape(existing.tensor.shape(), gradient.tensor.shape())?;
                    for (i, &contribution) in gradient.tensor.data().iter().enumerate() {
                        accumulate(&mut existing.tensor, i, contribution)?;
                    }
                } else {
                    *slot = Some(gradient);
                }
            }
        }
        Ok(batch)
    }

    fn transfer_work(&self, adjoints: &Table<Option<OwnedTensor>>) -> Result<usize, Error> {
        let mut work = add(self.parameters.len(), mul(self.nodes.entries.len(), 2)?)?;
        for (node, gradient) in self.nodes.entries.iter().zip(&adjoints.entries) {
            if let (Op::Parameter(_), Some(gradient)) = (node.op, gradient) {
                work = add(work, mul(gradient.tensor.numel(), 4)?)?;
            }
        }
        Ok(work)
    }

    fn binary(
        &mut self,
        a: ValueId,
        b: ValueId,
        operation: impl FnOnce(usize, usize) -> Op,
    ) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        self.operation(operation(self.index(a)?, self.index(b)?))
    }

    fn index(&self, id: ValueId) -> Result<usize, Error> {
        if id.owner != self.owner || id.index >= self.nodes.entries.len() {
            return Err(Error::InvalidValueId);
        }
        Ok(id.index)
    }

    fn ensure_recording(&self) -> Result<(), Error> {
        if matches!(self.state, State::Recording) {
            Ok(())
        } else {
            Err(Error::TapeSealed)
        }
    }

    fn reserve_node(&mut self, work: usize) -> Result<Option<Table<Node>>, Error> {
        let limits = self.parameters.limits();
        if self.nodes.entries.len() == limits.values {
            return Err(Error::ValueLimit {
                limit: limits.values,
            });
        }
        let cost = add(add(work, 1)?, self.nodes.move_work())?;
        let next = add(self.forward_work, cost)?;
        limits.check_work(Pass::Forward, next)?;
        let growth = self.nodes.growth(&self.parameters.budget, limits.values)?;
        self.forward_work = next;
        Ok(growth)
    }

    fn record_leaf(&mut self, op: Op, source: &Tensor) -> Result<ValueId, Error> {
        self.ensure_recording()?;
        let replacement = self.reserve_node(mul(source.numel(), 2)?)?;
        check_finite(source.data(), Stage::Input)?;
        let value = self
            .parameters
            .budget
            .tensor(source.rank(), source.numel(), || {
                source.try_clone(TensorLimits::default())
            })?;
        Ok(self.publish(replacement, op, value, matches!(op, Op::Parameter(_))))
    }

    fn operation(&mut self, op: Op) -> Result<ValueId, Error> {
        let limit = self.parameters.limits().operations;
        if self.operations == limit {
            return Err(Error::OperationLimit { limit });
        }
        let layout = op.layout(&self.nodes.entries)?;
        let replacement = self.reserve_node(layout.work)?;
        let value = self
            .parameters
            .budget
            .tensor(layout.rank, layout.len, || op.forward(&self.nodes.entries))?;
        check_finite(value.tensor.data(), Stage::Forward)?;
        let differentiable = op
            .edges()
            .into_iter()
            .flatten()
            .any(|i| self.nodes.entries[i].differentiable);
        self.operations = add(self.operations, 1)?;
        Ok(self.publish(replacement, op, value, differentiable))
    }

    fn publish(
        &mut self,
        replacement: Option<Table<Node>>,
        op: Op,
        value: OwnedTensor,
        differentiable: bool,
    ) -> ValueId {
        let index = self.nodes.entries.len();
        self.nodes.push(
            replacement,
            Node {
                value,
                op,
                differentiable,
                reachable: false,
            },
        );
        ValueId {
            owner: self.owner,
            index,
        }
    }

    fn plan_backward(&mut self, root: usize) -> Result<(), Error> {
        self.nodes.entries[root].reachable = self.nodes.entries[root].differentiable;
        // Cover the reachability walk, adjoint table initialization, reverse walk and seed.
        let reference = self.reference.as_ref();
        let scans = if reference.is_some() { 5 } else { 3 };
        let mut work = add(mul(self.nodes.entries.len(), scans)?, 2)?;
        self.parameters.limits().check_work(Pass::Backward, work)?;
        for index in (0..self.nodes.entries.len()).rev() {
            let node = &self.nodes.entries[index];
            if node.reachable {
                let op = node.op;
                if reference.is_some() {
                    work = add(work, mul(node.value.tensor.numel(), 5)?)?;
                }
                for (side, edge) in self.edges(op).into_iter().enumerate() {
                    if let Some(edge) = edge.filter(|&edge| self.nodes.entries[edge].differentiable)
                    {
                        let cost = if let Op::Reference(i) = op {
                            let reference = reference.unwrap();
                            reference.reverse_work(
                                reference.nodes.entries[i].kernel,
                                side,
                                &self.nodes.entries,
                                edge,
                            )?
                        } else {
                            op.backward_work(side, &self.nodes.entries, edge)?
                        };
                        work = add(work, cost)?;
                        self.nodes.entries[edge].reachable = true;
                    }
                }
            }
        }
        self.backward_work = work;
        self.parameters.limits().check_work(Pass::Backward, work)
    }

    fn run_reference_backward(
        &self,
        root: usize,
        seed: f32,
    ) -> Result<Table<Option<OwnedTensor>>, Error> {
        let budget = &self.parameters.budget;
        let nodes = &self.nodes.entries;
        let reference = self.reference()?;
        let mut adjoints = Table::allocate(budget, nodes.len())?;
        adjoints.entries.resize_with(nodes.len(), || None);
        let mut pending = Table::allocate(budget, nodes.len())?;
        pending.entries.resize_with(nodes.len(), || None);
        if nodes[root].reachable {
            pending.entries[root] = Some(Table::filled(budget, 1, f64::from(seed))?);
        }
        for (index, node) in nodes.iter().enumerate().rev() {
            let Some(sum) = pending.entries[index].take() else {
                continue;
            };
            let value = &node.value.tensor;
            let upstream = budget.generate(value.shape(), value.numel(), |i| {
                finite(sum.entries[i] as f32, Stage::Gradient, i)
            })?;
            drop(sum);
            for (side, edge) in self.edges(node.op).into_iter().enumerate() {
                let Some(edge) = edge.filter(|&i| nodes[i].differentiable) else {
                    continue;
                };
                let shape = &nodes[edge].value.tensor;
                let slot = &mut pending.entries[edge];
                if slot.is_none() {
                    *slot = Some(Table::filled(budget, shape.numel(), 0_f64)?);
                }
                let target = &mut slot.as_mut().unwrap().entries;
                if let Op::Reference(i) = node.op {
                    reference.reverse(
                        budget,
                        &reference.nodes.entries[i],
                        nodes,
                        side,
                        &upstream.tensor,
                        target,
                    )?;
                } else {
                    for (i, result) in target.iter_mut().enumerate() {
                        let delta = node
                            .op
                            .backward_coordinate(side, nodes, &upstream.tensor, i);
                        *result += f64::from(finite(delta, Stage::Contribution, i)?);
                    }
                }
            }
            adjoints.entries[index] = Some(upstream);
        }
        Ok(adjoints)
    }

    fn run_backward(&self, root: usize, seed: f32) -> Result<Table<Option<OwnedTensor>>, Error> {
        let budget = &self.parameters.budget;
        let nodes = &self.nodes.entries;
        let mut adjoints = Table::allocate(budget, nodes.len())?;
        adjoints.entries.resize_with(nodes.len(), || None);
        if nodes[root].reachable {
            adjoints.entries[root] = Some(budget.generate(&[], 1, |_| Ok(seed))?);
        }
        for index in (0..nodes.len()).rev() {
            let (earlier, current) = adjoints.entries.split_at_mut(index);
            if let Some(upstream) = &current[0] {
                let op = nodes[index].op;
                for (side, input) in op.edges().into_iter().enumerate() {
                    if let Some(input) = input.filter(|&input| nodes[input].differentiable) {
                        let value = &nodes[input].value.tensor;
                        let coordinate = |i| {
                            finite(
                                op.backward_coordinate(side, nodes, &upstream.tensor, i),
                                Stage::Contribution,
                                i,
                            )
                        };
                        if let Some(existing) = &mut earlier[input] {
                            for i in 0..value.numel() {
                                accumulate(&mut existing.tensor, i, coordinate(i)?)?;
                            }
                        } else {
                            earlier[input] =
                                Some(budget.generate(value.shape(), value.numel(), coordinate)?);
                        }
                    }
                }
            }
        }
        Ok(adjoints)
    }
}

fn accumulate(target: &mut Tensor, i: usize, contribution: f32) -> Result<(), Error> {
    let old = target.get_flat_mut(i)?;
    *old = finite(
        (f64::from(*old) + f64::from(contribution)) as f32,
        Stage::Gradient,
        i,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_owner_out_of_range_is_rejected_before_lookup() {
        let set = ParameterSet::new(Default::default()).unwrap();
        let mut tape = Tape::new(&set).unwrap();
        let id = ValueId {
            owner: tape.owner,
            index: usize::MAX,
        };
        assert!(matches!(tape.value(id), Err(Error::InvalidValueId)));
        assert_eq!(tape.sum(id), Err(Error::InvalidValueId));
        assert_eq!(tape.backward(id), Err(Error::InvalidValueId));
    }

    #[test]
    fn reference_allocation_failure_preserves_published_state_and_cleans_up() {
        let mut set = ParameterSet::new(Default::default()).unwrap();
        let logits = Tensor::zeros(&[1, 1, 260], Default::default()).unwrap();
        let id = set.insert(logits).unwrap();
        let baseline = set.accounted_bytes();
        let revision = set.revision();
        let mut tape = Tape::for_reference(
            &set,
            ReferenceBatch {
                shape: [1, 1],
                input: &[1],
                target: &[2],
                valid: None,
                loss: None,
            },
            260,
            1,
        )
        .unwrap();
        let prediction = tape.parameter(id).unwrap();
        let after_snapshot = set.accounted_bytes();
        set.budget.fail_allocation_after(2);
        assert_eq!(
            tape.next_token_loss(prediction),
            Err(Error::AllocationFailed)
        );
        assert_eq!(tape.metrics().unwrap().values, 1);
        assert_eq!(set.accounted_bytes(), after_snapshot);
        let loss = tape.next_token_loss(prediction).unwrap();
        let before_backward = set.accounted_bytes();
        set.budget.fail_allocation_after(5);
        assert_eq!(tape.backward(loss), Err(Error::AllocationFailed));
        assert_eq!(tape.backward(loss), Err(Error::BackwardAlreadyRun));
        assert!(matches!(
            tape.gradient(prediction),
            Err(Error::BackwardRequired)
        ));
        assert_eq!(set.accounted_bytes(), before_backward);
        assert!(set.gradient(id).unwrap().is_none());
        assert_eq!(set.revision(), revision);
        drop(tape);
        assert_eq!(set.accounted_bytes(), baseline);
        println!("TASK005 injected forward and reverse allocation failures: AllocationFailed");
    }
}
