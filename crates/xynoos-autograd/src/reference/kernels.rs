use super::{Budget, Error, Reference, Table, EPSILON};
use xynoos_tensor::Tensor;

pub(super) struct Dense<'a> {
    x: &'a [f32],
    w: &'a [f32],
    rows: usize,
    inner: usize,
    width: usize,
    transposed: bool,
}

impl<'a> Dense<'a> {
    pub(super) fn new(x: &'a Tensor, w: &'a Tensor, transposed: bool) -> Self {
        let inner = x.shape()[2];
        Self {
            x: x.data(),
            w: w.data(),
            rows: x.numel() / inner,
            inner,
            width: w.shape()[usize::from(!transposed)],
            transposed,
        }
    }

    fn weight(&self, k: usize, j: usize) -> f64 {
        f64::from(
            self.w[if self.transposed {
                j * self.inner + k
            } else {
                k * self.width + j
            }],
        )
    }

    pub(super) fn forward(&self, i: usize) -> f32 {
        let row = i / self.width;
        let j = i % self.width;
        let mut sum = 0.;
        for k in 0..self.inner {
            sum += f64::from(self.x[row * self.inner + k]) * self.weight(k, j);
        }
        sum as f32
    }

    pub(super) fn reverse(&self, g: &[f32], side: usize, i: usize) -> f32 {
        let mut sum = 0.;
        if side == 0 {
            let (row, k) = (i / self.inner, i % self.inner);
            for j in 0..self.width {
                sum += f64::from(g[row * self.width + j]) * self.weight(k, j);
            }
        } else {
            let (k, j) = if self.transposed {
                (i % self.inner, i / self.inner)
            } else {
                (i / self.width, i % self.width)
            };
            for row in 0..self.rows {
                sum += f64::from(self.x[row * self.inner + k]) * f64::from(g[row * self.width + j]);
            }
        }
        sum as f32
    }
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.;
    for (&u, &v) in a.iter().zip(b) {
        sum += f64::from(u) * f64::from(v);
    }
    sum
}

pub(super) fn rms(row: &[f32]) -> f64 {
    (EPSILON + dot(row, row) / row.len() as f64).sqrt()
}

pub(super) fn sigmoid(z: f32) -> f64 {
    let x = f64::from(z);
    let e = (-x.abs()).exp();
    if x >= 0. {
        1. / (1. + e)
    } else {
        e / (1. + e)
    }
}

pub(super) fn silu(z: f32, sigmoid: f64) -> f32 {
    (f64::from(z) * sigmoid) as f32
}

pub(super) fn log_partition(z: &[f32]) -> (f64, f64) {
    let mut peak = f64::from(z[0]);
    for &value in &z[1..] {
        peak = peak.max(f64::from(value));
    }
    let mut denominator = 0.;
    for &value in z {
        denominator += (f64::from(value) - peak).exp();
    }
    (peak, denominator)
}

impl Reference {
    pub(super) fn loss(&self, logits: &Tensor) -> f32 {
        let mut total = 0.;
        for (row, z) in logits.data().chunks_exact(self.vocab).enumerate() {
            if self.loss.entries[row] != 0 {
                let (peak, sum) = log_partition(z);
                total += sum.ln() + (peak - f64::from(z[self.target.entries[row] as usize]));
            }
        }
        (total / self.active as f64) as f32
    }

    pub(super) fn rotate(&self, values: &[f32], i: usize, d: usize, inverse: bool) -> [f32; 2] {
        let position = (i / d) % self.shape[1];
        let width = d / self.heads;
        let j = (i % width) / 2;
        let phase = position as f64 * 10000_f64.powf(-2. * j as f64 / width as f64);
        let c = f64::from(phase.cos() as f32);
        let s = f64::from(phase.sin() as f32) * if inverse { -1. } else { 1. };
        let a = f64::from(values[i]);
        let b = f64::from(values[i + 1]);
        [(a * c - b * s) as f32, (a * s + b * c) as f32]
    }

    pub(super) fn norm_reverse(
        &self,
        budget: &Budget,
        x: &Tensor,
        gain: &Tensor,
        grad: &[f32],
        side: usize,
        emit: &mut impl FnMut(usize, f32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let d = gain.numel();
        let mut gains = if side == 1 {
            Table::filled(budget, d, 0_f64)?
        } else {
            Table::empty()
        };
        for (row, (xs, gs)) in x
            .data()
            .chunks_exact(d)
            .zip(grad.chunks_exact(d))
            .enumerate()
        {
            if self.valid.entries[row] == 0 {
                continue;
            }
            let r = rms(xs);
            if side == 1 {
                for k in 0..d {
                    gains.entries[k] += f64::from(gs[k]) * f64::from(xs[k]) / r;
                }
            } else {
                let mut c = 0.;
                for k in 0..d {
                    c += f64::from(gs[k]) * f64::from(gain.data()[k]) * f64::from(xs[k]);
                }
                let div = d as f64 * r * r * r;
                for k in 0..d {
                    let delta = f64::from(gs[k]) * f64::from(gain.data()[k]) / r
                        - f64::from(xs[k]) * c / div;
                    emit(row * d + k, delta as f32)?;
                }
            }
        }
        for (i, x) in gains.entries.into_iter().enumerate() {
            emit(i, x as f32)?;
        }
        Ok(())
    }
}
