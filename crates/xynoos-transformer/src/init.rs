use crate::ParameterSpec;

pub(super) struct SplitMix {
    state: u64,
    spare: Option<f32>,
}

impl SplitMix {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5).min(9_007_199_254_740_991.) / 9_007_199_254_740_992.
    }

    fn normal(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let radius = (-2. * self.uniform().ln()).sqrt();
        let angle = 2. * std::f64::consts::PI * self.uniform();
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }

    pub(super) fn parameter(
        &mut self,
        spec: ParameterSpec,
        layers: usize,
    ) -> impl FnMut(usize) -> f32 + '_ {
        let shape = spec.shape();
        let bound = if shape.len() == 2 && spec.index() != 0 {
            let xavier = (6. / (shape[0] as f64 + shape[1] as f64)).sqrt();
            let scale = if spec.residual_output() {
                1. / (2. * layers as f64).sqrt()
            } else {
                1.
            };
            xavier * scale
        } else {
            0.
        };
        move |_| {
            if spec.index() == 0 {
                (f64::from(self.normal()) * 0.02) as f32
            } else if spec.shape().len() == 1 {
                1.
            } else {
                ((2. * self.uniform() - 1.) * bound) as f32
            }
        }
    }
}
