use xynoos_tensor::{Tensor, TensorError, TensorLimits};

use crate::budget::{add, mul};
use crate::error::same_shape;
use crate::tape::Node;
use crate::AutogradError as Error;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Op {
    Input,
    Parameter(usize),
    Add(usize, usize),
    Subtract(usize, usize),
    Multiply(usize, usize),
    Scale(usize, f32),
    Sum(usize),
    Mean(usize),
    Dot(usize, usize),
    Matmul(usize, usize),
    Transpose(usize),
    Mse(usize, usize),
}

pub(crate) struct Layout {
    pub(crate) rank: usize,
    pub(crate) len: usize,
    pub(crate) work: usize,
}

impl Op {
    /// Edges are listed in deterministic accumulation order. MSE target is constant.
    pub(crate) fn edges(self) -> [Option<usize>; 2] {
        use Op::*;
        match self {
            Input | Parameter(_) => [None, None],
            Scale(a, _) | Sum(a) | Mean(a) | Transpose(a) | Mse(a, _) => [Some(a), None],
            Add(a, b) | Subtract(a, b) | Multiply(a, b) | Dot(a, b) | Matmul(a, b) => {
                [Some(a), Some(b)]
            }
        }
    }

    pub(crate) fn layout(self, nodes: &[Node]) -> Result<Layout, Error> {
        use Op::*;
        let tensor = |i: usize| &nodes[i].value.tensor;
        let unary = |a, factor| -> Result<Layout, Error> {
            let a = tensor(a);
            Ok(Layout {
                rank: a.rank(),
                len: a.numel(),
                work: mul(a.numel(), factor)?,
            })
        };
        let reduction = |n, factor, extra| -> Result<Layout, Error> {
            Ok(Layout {
                rank: 0,
                len: 1,
                work: add(mul(n, factor)?, extra)?,
            })
        };
        match self {
            Add(a, b) | Subtract(a, b) | Multiply(a, b) => {
                same_shape(tensor(a).shape(), tensor(b).shape())?;
                unary(a, 3)
            }
            Scale(a, _) => unary(a, 3),
            Transpose(a) => {
                rank(tensor(a), 2)?;
                unary(a, 3)
            }
            Sum(a) => reduction(tensor(a).numel(), 1, 2),
            Mean(a) => reduction(nonempty(tensor(a))?, 1, 3),
            Dot(a, b) => {
                rank(tensor(a), 1)?;
                rank(tensor(b), 1)?;
                same_shape(tensor(a).shape(), tensor(b).shape())?;
                reduction(tensor(a).numel(), 2, 2)
            }
            Matmul(a, b) => {
                rank(tensor(a), 2)?;
                rank(tensor(b), 2)?;
                let a = tensor(a).shape();
                let b = tensor(b).shape();
                if a[1] != b[0] {
                    return Err(TensorError::MatmulDimensionMismatch {
                        left: a[1],
                        right: b[0],
                    }
                    .into());
                }
                let len = mul(a[0], b[1])?;
                Ok(Layout {
                    rank: 2,
                    len,
                    work: add(mul(len, 3)?, product_work(len, a[1])?)?,
                })
            }
            Mse(a, b) => {
                same_shape(tensor(a).shape(), tensor(b).shape())?;
                reduction(nonempty(tensor(a))?, 3, 3)
            }
            Input | Parameter(_) => unreachable!("leaves are recorded separately"),
        }
    }

    pub(crate) fn forward(self, nodes: &[Node]) -> Result<Tensor, TensorError> {
        use Op::*;
        let t = |i: usize| &nodes[i].value.tensor;
        let limits = TensorLimits::default();
        let scalar = |x| Tensor::from_data(&[], &[x], limits);
        match self {
            Add(a, b) => t(a).add(t(b), limits),
            Subtract(a, b) => t(a).subtract(t(b), limits),
            Multiply(a, b) => t(a).multiply(t(b), limits),
            Scale(a, c) => t(a).scale(c, limits),
            Sum(a) => scalar(t(a).sum()),
            Mean(a) => {
                let mut total = 0.0;
                for &value in t(a).data() {
                    total += f64::from(value);
                }
                scalar((total / t(a).numel() as f64) as f32)
            }
            Dot(a, b) => scalar(t(a).dot(t(b))?),
            Matmul(a, b) => t(a).matmul(t(b), limits),
            Transpose(a) => t(a).transpose(limits),
            Mse(a, b) => {
                let mut total = 0.0;
                for (&prediction, &target) in t(a).data().iter().zip(t(b).data()) {
                    let difference = f64::from(prediction) - f64::from(target);
                    total += difference * difference;
                }
                scalar((total / t(a).numel() as f64) as f32)
            }
            Input | Parameter(_) => unreachable!("leaves are recorded separately"),
        }
    }

    pub(crate) fn backward_work(
        self,
        side: usize,
        nodes: &[Node],
        input: usize,
    ) -> Result<usize, Error> {
        let cells = nodes[input].value.tensor.numel();
        // Charge each cell for conversion, finite checks, additive accumulation and a write.
        let base = add(mul(cells, 6)?, 1)?;
        match self {
            Self::Matmul(a, b) => {
                let inner = if side == 0 {
                    nodes[b].value.tensor.shape()[1]
                } else {
                    nodes[a].value.tensor.shape()[0]
                };
                add(base, product_work(cells, inner)?)
            }
            Self::Mse(..) => add(base, mul(cells, 4)?),
            _ => add(base, cells),
        }
    }

    pub(crate) fn backward_coordinate(
        self,
        side: usize,
        nodes: &[Node],
        grad: &Tensor,
        i: usize,
    ) -> f32 {
        use Op::*;
        let t = |index: usize| &nodes[index].value.tensor;
        let g = grad.data();
        match self {
            Add(..) => g[i],
            Subtract(..) => {
                if side == 0 {
                    g[i]
                } else {
                    -g[i]
                }
            }
            Multiply(a, b) => g[i] * t(if side == 0 { b } else { a }).data()[i],
            Scale(_, c) => c * g[i],
            Sum(_) => g[0],
            Mean(a) => (f64::from(g[0]) / t(a).numel() as f64) as f32,
            Dot(a, b) => g[0] * t(if side == 0 { b } else { a }).data()[i],
            Matmul(a, b) => matmul_derivative(t(a), t(b), grad, side, i),
            Transpose(a) => {
                let rows = t(a).shape()[0];
                let columns = t(a).shape()[1];
                g[(i % columns) * rows + i / columns]
            }
            Mse(a, b) => {
                let difference = f64::from(t(a).data()[i]) - f64::from(t(b).data()[i]);
                ((f64::from(g[0]) * 2.0 * difference) / t(a).numel() as f64) as f32
            }
            Input | Parameter(_) => unreachable!("leaves have no reverse edges"),
        }
    }
}

fn product_work(output_cells: usize, inner: usize) -> Result<usize, Error> {
    mul(mul(output_cells, inner)?, 2)
}

fn rank(tensor: &Tensor, expected: usize) -> Result<(), Error> {
    let actual = tensor.rank();
    if actual != expected {
        return Err(TensorError::RankMismatch { expected, actual }.into());
    }
    Ok(())
}

fn nonempty(tensor: &Tensor) -> Result<usize, Error> {
    if tensor.numel() == 0 {
        return Err(Error::EmptyReduction);
    }
    Ok(tensor.numel())
}

fn matmul_derivative(a: &Tensor, b: &Tensor, g: &Tensor, side: usize, i: usize) -> f32 {
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    let mut total = 0.0;
    // Called only for a nonempty derivative, so the corresponding row width is nonzero.
    if side == 0 {
        let (row, q) = (i / k, i % k);
        for column in 0..n {
            total += f64::from(g.data()[row * n + column]) * f64::from(b.data()[q * n + column]);
        }
    } else {
        let (q, column) = (i / n, i % n);
        for row in 0..m {
            total += f64::from(a.data()[row * k + q]) * f64::from(g.data()[row * n + column]);
        }
    }
    total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_work_checks_overflow_and_empty_outputs() {
        assert_eq!(product_work(usize::MAX, 0), Ok(0));
        assert_eq!(product_work(0, usize::MAX), Ok(0));
        assert_eq!(product_work(usize::MAX, 1), Err(Error::ArithmeticOverflow));
    }
}
