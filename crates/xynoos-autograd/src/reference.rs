//! Fixed-rank kernels used by the decoder reference tape.

mod attention;
mod kernels;

use crate::budget::{add, mul, tensor_bytes, Budget, OwnedTensor, Table};
use crate::error::{finite, same_shape};
use crate::tape::Node;
use crate::{AutogradError as Error, Pass, Stage};
use xynoos_tensor::Tensor;

const PAD: u32 = 258;
const DOC_SEP: u32 = 259;
const EPSILON: f64 = 1e-5;

/// Borrowed, row-major next-token inputs. Missing masks mean all true.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceBatch<'a> {
    pub shape: [usize; 2],
    pub input: &'a [u32],
    pub target: &'a [u32],
    pub valid: Option<&'a [u8]>,
    pub loss: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Kernel {
    Gather(usize),
    Dense(usize, usize, bool),
    Norm(usize, usize),
    Rope(usize),
    Attention(usize, usize, usize),
    SwiGlu(usize, usize),
    Loss(usize),
    Residual(usize, usize),
}

impl Kernel {
    pub(crate) fn edges(self) -> [Option<usize>; 3] {
        use Kernel::*;
        match self {
            Gather(a) | Rope(a) | Loss(a) => [Some(a), None, None],
            Dense(a, b, _) | Norm(a, b) | SwiGlu(a, b) | Residual(a, b) => [Some(a), Some(b), None],
            Attention(a, b, c) => [Some(a), Some(b), Some(c)],
        }
    }
}

#[derive(Debug)]
pub(crate) struct Recorded {
    pub(crate) kernel: Kernel,
    probabilities: Table<f64>,
}

#[derive(Debug)]
pub(crate) struct Reference {
    pub(crate) nodes: Table<Recorded>,
    shape: [usize; 2],
    heads: usize,
    vocab: usize,
    input: Table<u32>,
    target: Table<u32>,
    valid: Table<u8>,
    loss: Table<u8>,
    active: usize,
}

impl Reference {
    pub(crate) fn new(
        budget: &Budget,
        batch: ReferenceBatch<'_>,
        vocab: usize,
        heads: usize,
    ) -> Result<(Self, usize), Error> {
        let [b, t] = batch.shape;
        let rows = mul(b, t)?;
        if !(1..=64).contains(&b)
            || !(1..=2048).contains(&t)
            || rows > 65_536
            || heads == 0
            || !(260..=772).contains(&vocab)
            || batch.input.len() != rows
            || batch.target.len() != rows
            || batch.valid.is_some_and(|m| m.len() != rows)
            || batch.loss.is_some_and(|m| m.len() != rows)
        {
            return Err(Error::InvalidReferenceShape);
        }
        let work = mul(rows, 24)?;
        budget.limits.check_work(Pass::Forward, work)?;
        let mut active = 0;
        for i in 0..rows {
            let input = batch.input[i];
            let target = batch.target[i];
            let valid = batch.valid.map_or(1, |m| m[i]);
            let loss = batch.loss.map_or(1, |m| m[i]);
            let interior = i % t + 1 < t && batch.valid.is_none_or(|m| m[i + 1] != 0);
            if valid > 1
                || loss > 1
                || (valid == 0 && (loss != 0 || interior))
                || (target == PAD && loss != 0)
            {
                return Err(Error::InvalidMask { index: i });
            }
            if input as usize >= vocab
                || (valid != 0 && matches!(input, PAD | DOC_SEP))
                || (valid == 0 && input != PAD)
            {
                return Err(Error::InvalidToken {
                    index: i,
                    target: false,
                });
            }
            if target as usize >= vocab
                || (valid != 0 && target == DOC_SEP)
                || (valid == 0 && target != PAD)
            {
                return Err(Error::InvalidToken {
                    index: i,
                    target: true,
                });
            }
            if valid != 0 && interior && target != batch.input[i + 1] {
                return Err(Error::InvalidTargetShift { index: i });
            }
            active += usize::from(loss);
        }
        if active == 0 {
            return Err(Error::NoSupervisedTargets);
        }
        let mask = |m: Option<&[u8]>| match m {
            Some(m) => Table::copy(budget, m),
            None => Table::filled(budget, rows, 1),
        };
        Ok((
            Self {
                nodes: Table::empty(),
                shape: batch.shape,
                vocab,
                heads,
                active,
                input: Table::copy(budget, batch.input)?,
                target: Table::copy(budget, batch.target)?,
                valid: mask(batch.valid)?,
                loss: mask(batch.loss)?,
            },
            work,
        ))
    }

    fn rows(&self) -> usize {
        self.input.entries.len()
    }

    fn activation(&self, t: &Tensor) -> Result<usize, Error> {
        match t.shape() {
            [b, n, d] if [*b, *n] == self.shape && *d > 0 => Ok(*d),
            _ => Err(Error::InvalidReferenceShape),
        }
    }

    fn head_width(&self, width: usize) -> Result<usize, Error> {
        if width == 0
            || !width.is_multiple_of(self.heads)
            || !(width / self.heads).is_multiple_of(2)
        {
            return Err(Error::InvalidReferenceShape);
        }
        Ok(width / self.heads)
    }

    fn square(&self) -> Result<usize, Error> {
        let cells = mul(mul(self.rows(), self.heads)?, self.shape[1])?;
        // Probabilities are retained in binary64. Apply the same single-buffer
        // ceiling as differentiable tensors.
        tensor_bytes(0, mul(cells, 2)?)?;
        Ok(cells)
    }

    pub(crate) fn output(
        &self,
        kernel: Kernel,
        nodes: &[Node],
    ) -> Result<([usize; 3], usize, usize, usize), Error> {
        let value = |i: usize| &nodes[i].value.tensor;
        let rows = self.rows();
        let (width, rank, work) = match kernel {
            Kernel::Gather(a) => match value(a).shape() {
                [v, d] if *v == self.vocab && *d > 0 => {
                    (*d, 3, add(mul(rows, 3)?, mul(mul(rows, *d)?, 3)?)?)
                }
                _ => return Err(Error::InvalidReferenceShape),
            },
            Kernel::Dense(a, b, transposed) => {
                let d = self.activation(value(a))?;
                let n = match value(b).shape() {
                    [n, k] if transposed && *k == d && *n > 0 => *n,
                    [k, n] if !transposed && *k == d && *n > 0 => *n,
                    _ => return Err(Error::InvalidReferenceShape),
                };
                (n, 3, mul(mul(rows, n)?, add(mul(d, 2)?, 3)?)?)
            }
            Kernel::Norm(a, b) => {
                let d = self.activation(value(a))?;
                same_shape(value(b).shape(), &[d])?;
                (d, 3, mul(rows, add(mul(d, 7)?, 6)?)?)
            }
            Kernel::Rope(a) => {
                let d = self.activation(value(a))?;
                self.head_width(d)?;
                (d, 3, mul(value(a).numel(), 18)?)
            }
            Kernel::Attention(a, b, c) => {
                let d = self.activation(value(a))?;
                same_shape(value(a).shape(), value(b).shape())?;
                same_shape(value(a).shape(), value(c).shape())?;
                let h = self.head_width(d)?;
                let cells = self.square()?;
                (
                    d,
                    3,
                    add(
                        add(mul(cells, add(mul(h, 4)?, 12)?)?, mul(value(a).numel(), 4)?)?,
                        4,
                    )?,
                )
            }
            Kernel::Residual(a, b) => {
                let d = self.activation(value(a))?;
                same_shape(value(a).shape(), value(b).shape())?;
                (d, 3, mul(value(a).numel(), 5)?)
            }
            Kernel::SwiGlu(a, b) => {
                let d = self.activation(value(a))?;
                same_shape(value(a).shape(), value(b).shape())?;
                (d, 3, mul(value(a).numel(), 10)?)
            }
            Kernel::Loss(a) => {
                same_shape(
                    value(a).shape(),
                    &[self.shape[0], self.shape[1], self.vocab],
                )?;
                (
                    0,
                    0,
                    add(
                        mul(rows, 3)?,
                        add(mul(mul(self.active, self.vocab)?, 6)?, 4)?,
                    )?,
                )
            }
        };
        let len = if rank == 0 { 1 } else { mul(rows, width)? };
        tensor_bytes(rank, len)?;
        Ok(([self.shape[0], self.shape[1], width], rank, len, work))
    }

    pub(crate) fn reverse_work(
        &self,
        kernel: Kernel,
        side: usize,
        nodes: &[Node],
        input: usize,
    ) -> Result<usize, Error> {
        let t = |i: usize| &nodes[i].value.tensor;
        let cells = t(input).numel();
        let base = add(mul(cells, 6)?, 1)?;
        let work = match kernel {
            Kernel::Gather(a) => add(cells, mul(self.rows(), add(mul(t(a).shape()[1], 2)?, 3)?)?)?,
            Kernel::Dense(_, b, transposed) => {
                let inner = if side == 0 {
                    t(b).shape()[usize::from(!transposed)]
                } else {
                    self.rows()
                };
                mul(mul(cells, inner)?, 2)?
            }
            Kernel::Norm(a, _) => mul(
                self.rows(),
                add(mul(t(a).shape()[2], if side == 0 { 13 } else { 7 })?, 10)?,
            )?,
            Kernel::Rope(_) => mul(cells, 18)?,
            Kernel::Attention(a, _, _) => {
                let d = t(a).shape()[2];
                let pair = if side == 2 { 2 } else { 8 };
                add(
                    add(
                        cells,
                        mul(self.square()?, add(mul(d / self.heads, pair)?, 10)?)?,
                    )?,
                    4,
                )?
            }
            Kernel::SwiGlu(..) => mul(cells, 14)?,
            Kernel::Residual(..) => cells,
            Kernel::Loss(_) => add(
                mul(self.rows(), 3)?,
                add(mul(mul(self.active, self.vocab)?, 9)?, 4)?,
            )?,
        };
        add(base, work)
    }

    pub(crate) fn forward(
        &self,
        budget: &Budget,
        kernel: Kernel,
        nodes: &[Node],
        shape: &[usize],
        len: usize,
    ) -> Result<(OwnedTensor, Recorded), Error> {
        let t = |i: usize| &nodes[i].value.tensor;
        let mut probabilities = Table::empty();
        let output = match kernel {
            Kernel::Gather(a) => {
                let d = t(a).shape()[1];
                budget.generate(shape, len, |i| {
                    let row = i / d;
                    Ok(if self.valid.entries[row] == 0 {
                        0.
                    } else {
                        t(a).data()[self.input.entries[row] as usize * d + i % d]
                    })
                })?
            }
            Kernel::Dense(a, b, transposed) => {
                let dense = kernels::Dense::new(t(a), t(b), transposed);
                budget.generate(shape, len, |i| Ok(dense.forward(i)))?
            }
            Kernel::Norm(a, b) => {
                let d = shape[2];
                let mut inverse = 0.;
                budget.generate(shape, len, |i| {
                    let row = i / d;
                    if self.valid.entries[row] == 0 {
                        return Ok(0.);
                    }
                    let x = t(a).data();
                    if i % d == 0 {
                        inverse = kernels::rms(&x[i..i + d]);
                    }
                    Ok((f64::from(t(b).data()[i % d]) * f64::from(x[i]) / inverse) as f32)
                })?
            }
            Kernel::Rope(a) => {
                let d = shape[2];
                let mut pair = [0.; 2];
                budget.generate(shape, len, |i| {
                    if i % 2 == 0 {
                        pair = self.rotate(t(a).data(), i, d, false);
                    }
                    Ok(pair[i % 2])
                })?
            }
            Kernel::Attention(a, b, c) => {
                let context = attention::Attention::new(self, t(a), t(b), t(c));
                probabilities = context.probabilities(budget)?;
                budget.generate(shape, len, |i| Ok(context.apply(&probabilities.entries, i)))?
            }
            Kernel::Residual(a, b) => budget.generate(shape, len, |i| {
                Ok(if self.valid.entries[i / shape[2]] == 0 {
                    0.
                } else {
                    t(a).data()[i] + t(b).data()[i]
                })
            })?,
            Kernel::SwiGlu(a, b) => budget.generate(shape, len, |i| {
                let z = t(a).data()[i];
                Ok(kernels::silu(z, kernels::sigmoid(z)) * t(b).data()[i])
            })?,
            Kernel::Loss(a) => budget.generate(shape, len, |_| Ok(self.loss(t(a))))?,
        };
        Ok((
            output,
            Recorded {
                kernel,
                probabilities,
            },
        ))
    }

    pub(crate) fn reverse(
        &self,
        budget: &Budget,
        node: &Recorded,
        nodes: &[Node],
        side: usize,
        grad: &Tensor,
        target: &mut [f64],
    ) -> Result<(), Error> {
        let t = |i: usize| &nodes[i].value.tensor;
        let g = grad.data();
        let mut emit = |i: usize, delta: f32| -> Result<(), Error> {
            target[i] += f64::from(finite(delta, Stage::Contribution, i)?);
            Ok(())
        };
        match node.kernel {
            Kernel::Gather(a) => {
                let d = t(a).shape()[1];
                let mut scratch = Table::filled(budget, t(a).numel(), 0_f64)?;
                for (row, &id) in self.input.entries.iter().enumerate() {
                    if self.valid.entries[row] != 0 {
                        for k in 0..d {
                            scratch.entries[id as usize * d + k] += f64::from(g[row * d + k]);
                        }
                    }
                }
                for (i, &x) in scratch.entries.iter().enumerate() {
                    emit(i, x as f32)?;
                }
            }
            Kernel::Dense(a, b, transposed) => {
                let dense = kernels::Dense::new(t(a), t(b), transposed);
                let len = if side == 0 {
                    t(a).numel()
                } else {
                    t(b).numel()
                };
                for i in 0..len {
                    emit(i, dense.reverse(g, side, i))?;
                }
            }
            Kernel::Norm(a, b) => self.norm_reverse(budget, t(a), t(b), g, side, &mut emit)?,
            Kernel::Rope(a) => {
                let d = t(a).shape()[2];
                for i in (0..g.len()).step_by(2) {
                    let [x, y] = self.rotate(g, i, d, true);
                    emit(i, x)?;
                    emit(i + 1, y)?;
                }
            }
            Kernel::Attention(a, b, c) => {
                let context = attention::Attention::new(self, t(a), t(b), t(c));
                context.reverse(budget, &node.probabilities.entries, g, side, &mut emit)?;
            }
            Kernel::Residual(a, _) => {
                let d = t(a).shape()[2];
                for (i, &x) in g.iter().enumerate() {
                    if self.valid.entries[i / d] != 0 {
                        emit(i, x)?;
                    }
                }
            }
            Kernel::SwiGlu(a, b) => {
                for (i, (&z, &upstream)) in t(a).data().iter().zip(g).enumerate() {
                    let p = kernels::sigmoid(z);
                    let local = if side == 0 {
                        f64::from(t(b).data()[i]) * (p + f64::from(z) * p * (1. - p))
                    } else {
                        f64::from(kernels::silu(z, p))
                    };
                    emit(i, (f64::from(upstream) * local) as f32)?;
                }
            }
            Kernel::Loss(a) => {
                for (i, z) in t(a).data().chunks_exact(self.vocab).enumerate() {
                    if self.loss.entries[i] == 0 {
                        continue;
                    }
                    let (peak, sum) = kernels::log_partition(z);
                    let target_id = self.target.entries[i] as usize;
                    for (j, &value) in z.iter().enumerate() {
                        let p = (f64::from(value) - peak).exp() / sum;
                        emit(
                            i * self.vocab + j,
                            (f64::from(g[0]) * (p - f64::from(j == target_id)) / self.active as f64)
                                as f32,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}
