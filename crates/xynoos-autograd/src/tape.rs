use xynoos_tensor::{Tensor, TensorLimits};

use crate::budget::{add, mul, OwnedTensor, Table};
use crate::error::{check_finite, finite, same_shape};
use crate::kernels::Op;
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
        })
    }

    pub fn value(&self, id: ValueId) -> Result<&Tensor, Error> {
        Ok(&self.nodes.entries[self.index(id)?].value.tensor)
    }

    pub fn input(&mut self, value: &Tensor) -> Result<ValueId, Error> {
        self.record_leaf(Op::Input, value)
    }

    pub fn parameter(&mut self, id: ParameterId) -> Result<ValueId, Error> {
        self.record_leaf(Op::Parameter(id), self.parameters.value(id)?)
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
        let adjoints = self.run_backward(root, seed)?;
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
        let State::Complete(adjoints) = &mut self.state else {
            return Err(Error::BackwardRequired);
        };
        let mut work = add(self.parameters.len(), mul(self.nodes.entries.len(), 2)?)?;
        for (node, gradient) in self.nodes.entries.iter().zip(&adjoints.entries) {
            if let (Op::Parameter(_), Some(gradient)) = (node.op, gradient) {
                work = add(work, mul(gradient.tensor.numel(), 4)?)?;
            }
        }
        self.parameters
            .limits()
            .check_work(Pass::GradientTransfer, work)?;
        let mut batch = self.parameters.batch()?;
        for (node, gradient) in self.nodes.entries.iter().zip(&mut adjoints.entries).rev() {
            if let (Op::Parameter(id), Some(gradient)) = (node.op, gradient.take()) {
                let slot = &mut batch.gradients.entries[id.index];
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
        let mut work = add(mul(self.nodes.entries.len(), 3)?, 2)?;
        self.parameters.limits().check_work(Pass::Backward, work)?;
        for index in (0..self.nodes.entries.len()).rev() {
            let node = &self.nodes.entries[index];
            if node.reachable {
                let op = node.op;
                for (side, edge) in op.edges().into_iter().enumerate() {
                    if let Some(edge) = edge.filter(|&edge| self.nodes.entries[edge].differentiable)
                    {
                        work = add(work, op.backward_work(side, &self.nodes.entries, edge)?)?;
                        self.nodes.entries[edge].reachable = true;
                    }
                }
            }
        }
        self.parameters.limits().check_work(Pass::Backward, work)
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
}
