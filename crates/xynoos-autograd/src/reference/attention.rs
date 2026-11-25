use super::{kernels::dot, tensor_bytes, Budget, Error, Reference, Table};
use xynoos_tensor::Tensor;

/// Views [B,T,D] as [B,T,H,d] with no storage permutation.
pub(super) struct Attention<'a> {
    state: &'a Reference,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    d: usize,
    width: usize,
    time: usize,
    scale: f64,
}

impl<'a> Attention<'a> {
    pub(super) fn new(state: &'a Reference, q: &'a Tensor, k: &'a Tensor, v: &'a Tensor) -> Self {
        let d = q.shape()[2];
        let width = d / state.heads;
        Self {
            state,
            q: q.data(),
            k: k.data(),
            v: v.data(),
            d,
            width,
            time: state.shape[1],
            scale: 1. / (width as f64).sqrt(),
        }
    }

    fn head(&self, row: usize, h: usize) -> usize {
        row * self.d + h * self.width
    }

    fn sample_start(&self, row: usize) -> usize {
        row - row % self.time
    }

    fn probs(&self, row: usize, h: usize) -> usize {
        (row * self.state.heads + h) * self.time
    }

    pub(super) fn probabilities(&self, budget: &Budget) -> Result<Table<f64>, Error> {
        let cells = self.state.square()?;
        let mut p = Table::allocate(budget, cells)?;
        tensor_bytes(0, super::mul(p.entries.capacity(), 2)?)?;
        p.entries.resize(cells, 0_f64);
        for row in 0..self.state.rows() {
            if self.state.valid.entries[row] == 0 {
                continue;
            }
            let start = self.sample_start(row);
            for h in 0..self.state.heads {
                let q = self.head(row, h);
                let offset = self.probs(row, h);
                // Prefix-valid inputs imply that all keys through a valid query are valid.
                // No score storage is initialized with a nonfinite sentinel.
                for key in start..=row {
                    let k = self.head(key, h);
                    p.entries[offset + key - start] =
                        dot(&self.q[q..q + self.width], &self.k[k..k + self.width]) * self.scale;
                }
                let allowed = &mut p.entries[offset..offset + row - start + 1];
                let mut max = allowed[0];
                for &x in &allowed[1..] {
                    max = max.max(x);
                }
                let mut sum = 0.;
                for x in &mut *allowed {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for x in allowed {
                    *x /= sum;
                }
            }
        }
        Ok(p)
    }

    pub(super) fn apply(&self, p: &[f64], i: usize) -> f32 {
        let row = i / self.d;
        if self.state.valid.entries[row] == 0 {
            return 0.;
        }
        let feature = i % self.d;
        let offset = self.probs(row, feature / self.width);
        let start = self.sample_start(row);
        let mut sum = 0.;
        for key in start..=row {
            sum += p[offset + key - start] * f64::from(self.v[key * self.d + feature]);
        }
        sum as f32
    }

    pub(super) fn reverse(
        &self,
        budget: &Budget,
        p: &[f64],
        g: &[f32],
        side: usize,
        emit: &mut impl FnMut(usize, f32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut result = Table::filled(budget, g.len(), 0_f64)?;
        let mut dp = if side < 2 {
            Table::filled(budget, self.time, 0_f64)?
        } else {
            Table::empty()
        };
        for row in 0..self.state.rows() {
            if self.state.valid.entries[row] == 0 {
                continue;
            }
            let start = self.sample_start(row);
            for h in 0..self.state.heads {
                let q = self.head(row, h);
                let offset = self.probs(row, h);
                let allowed = &p[offset..offset + row - start + 1];
                let mut mean = 0.;
                if side < 2 {
                    for (s, &p) in allowed.iter().enumerate() {
                        let k = self.head(start + s, h);
                        dp.entries[s] = dot(&g[q..q + self.width], &self.v[k..k + self.width]);
                        mean += p * dp.entries[s];
                    }
                }
                for (s, &p) in allowed.iter().enumerate() {
                    let k = self.head(start + s, h);
                    match side {
                        0 | 1 => {
                            let dz = p * (dp.entries[s] - mean) * self.scale;
                            let (dst, src, values) = if side == 0 {
                                (q, k, self.k)
                            } else {
                                (k, q, self.q)
                            };
                            for j in 0..self.width {
                                result.entries[dst + j] += dz * f64::from(values[src + j]);
                            }
                        }
                        _ => {
                            for j in 0..self.width {
                                result.entries[k + j] += p * f64::from(g[q + j]);
                            }
                        }
                    }
                }
            }
        }
        for (i, &x) in result.entries.iter().enumerate() {
            emit(i, x as f32)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReferenceBatch, ResourceLimits};

    #[test]
    fn masked_and_future_probabilities_are_exact_zero() {
        let budget = Budget::new(ResourceLimits::default());
        let (reference, _) = Reference::new(
            &budget,
            ReferenceBatch {
                shape: [2, 3],
                input: &[1, 2, 258, 258, 258, 258],
                target: &[2, 3, 258, 258, 258, 258],
                valid: Some(&[1, 1, 0, 0, 0, 0]),
                loss: Some(&[1, 1, 0, 0, 0, 0]),
            },
            260,
            1,
        )
        .unwrap();
        let values = Tensor::from_data(
            &[2, 3, 2],
            &[1., 0., 0., 1., 99., 99., 1., 1., 2., 2., 3., 3.],
            Default::default(),
        )
        .unwrap();
        let attn = Attention::new(&reference, &values, &values, &values);
        let saved = attn.probabilities(&budget).unwrap();
        assert_eq!(saved.entries[0], 1.);
        assert!((saved.entries[3] + saved.entries[4] - 1.).abs() < 1e-15);
        for i in [1, 2].into_iter().chain(5..18) {
            assert_eq!(saved.entries[i].to_bits(), 0);
        }
    }
}
