use std::fmt;
use xynoos_autograd::{AutogradError, Pass, ResourceLimits};
use xynoos_tensor::{TensorError, TensorLimits};

/// Dimensions of the fixed TASK 005 reference architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub vocabulary: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub context: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidConfiguration,
    VocabularyMismatch { model: usize, tokenizer: usize },
    InvalidBatchShape,
    InvalidInventory,
    Autograd(AutogradError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Autograd(error) => error.fmt(f),
            other => write!(f, "{other:?}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Autograd(error) => Some(error),
            _ => None,
        }
    }
}

impl From<AutogradError> for Error {
    fn from(error: AutogradError) -> Self {
        Self::Autograd(error)
    }
}

pub(crate) fn add(a: usize, b: usize) -> Result<usize, Error> {
    Ok(a.checked_add(b).ok_or(AutogradError::ArithmeticOverflow)?)
}

pub(crate) fn mul(a: usize, b: usize) -> Result<usize, Error> {
    Ok(a.checked_mul(b).ok_or(AutogradError::ArithmeticOverflow)?)
}

fn tensor(elements: usize, cell: usize) -> Result<(), Error> {
    let bytes = mul(elements, cell)?;
    let limit = TensorLimits::MAX_BYTES;
    if bytes > limit {
        return Err(
            AutogradError::Tensor(TensorError::PayloadLimitExceeded { bytes, limit }).into(),
        );
    }
    Ok(())
}

impl Config {
    pub const MATHEMATICAL: Self = Self {
        vocabulary: 260,
        width: 16,
        layers: 1,
        heads: 2,
        ffn: 48,
        context: 8,
    };
    pub const INTEGRATION: Self = Self {
        vocabulary: 292,
        width: 32,
        layers: 4,
        heads: 4,
        ffn: 96,
        context: 64,
    };
    pub const RESEARCH_CANDIDATE: Self = Self {
        vocabulary: 292,
        width: 96,
        layers: 8,
        heads: 4,
        ffn: 256,
        context: 128,
    };

    pub fn parameter_count(self) -> Result<usize, Error> {
        let Self {
            vocabulary: v,
            width: d,
            layers: n,
            heads: h,
            ffn: f,
            context,
        } = self;
        if !(260..=772).contains(&v)
            || d == 0
            || n == 0
            || h == 0
            || f == 0
            || context == 0
            || !d.is_multiple_of(h)
            || !(d / h).is_multiple_of(2)
        {
            return Err(Error::InvalidConfiguration);
        }
        let embedding = mul(v, d)?;
        let attention = mul(d, d)?;
        let ff = mul(d, f)?;
        let block = add(add(mul(attention, 4)?, mul(ff, 3)?)?, mul(d, 2)?)?;
        let count = add(add(embedding, d)?, mul(n, block)?)?;
        for elements in [embedding, attention, ff, d] {
            tensor(elements, size_of::<f32>())?;
        }
        Ok(count)
    }

    pub(crate) fn preflight_batch(
        self,
        [b, t]: [usize; 2],
        limits: ResourceLimits,
        parameter_count: usize,
        live_bytes: usize,
    ) -> Result<(), Error> {
        let rows = mul(b, t)?;
        if !(1..=64).contains(&b) || !(1..=self.context.min(2048)).contains(&t) || rows > 65_536 {
            return Err(Error::InvalidBatchShape);
        }
        for width in [self.width, self.ffn, self.vocabulary] {
            tensor(mul(rows, width)?, size_of::<f32>())?;
        }
        let probability_cells = mul(mul(rows, self.heads)?, t)?;
        tensor(probability_cells, size_of::<f64>())?;
        let operations = add(mul(self.layers, 15)?, 4)?;
        let values = add(self.inventory_len()?, operations)?;
        if operations > limits.max_operations() {
            return Err(AutogradError::OperationLimit {
                limit: limits.max_operations(),
            }
            .into());
        }
        if values > limits.max_values() {
            return Err(AutogradError::ValueLimit {
                limit: limits.max_values(),
            }
            .into());
        }
        // Unavoidable dense multiply/add contribution, before kernel-specific
        // charges, scans, masks, node growth and parameter snapshots.
        let (d, f, v, n) = (self.width, self.ffn, self.vocabulary, self.layers);
        let block_weights = add(mul(mul(4, d)?, d)?, mul(mul(3, d)?, f)?)?;
        let dense_products = add(mul(n, block_weights)?, mul(v, d)?)?;
        let dense_floor = mul(mul(rows, dense_products)?, 2)?;
        if dense_floor > limits.max_work() {
            return Err(AutogradError::WorkLimit {
                pass: Pass::Forward,
                requested: dense_floor,
                limit: limits.max_work(),
            }
            .into());
        }
        // Floor for simultaneously live buffers after forward. The tape then
        // charges metadata, capacity slack and transient overlap as they occur.
        let block_activations = add(mul(12, d)?, mul(3, f)?)?;
        let row_widths = add(add(mul(n, block_activations)?, mul(2, d)?)?, v)?;
        let cells = add(add(mul(rows, row_widths)?, parameter_count)?, 1)?;
        let bytes = add(
            mul(cells, size_of::<f32>())?,
            mul(mul(n, probability_cells)?, size_of::<f64>())?,
        )?;
        let batch_bytes = mul(rows, 2 * (size_of::<u32>() + size_of::<u8>()))?;
        let requested = add(live_bytes, add(bytes, batch_bytes)?)?;
        if requested > limits.max_bytes() {
            return Err(AutogradError::MemoryLimit {
                requested,
                limit: limits.max_bytes(),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn inventory_len(self) -> Result<usize, Error> {
        add(mul(self.layers, 9)?, 2)
    }

    pub(crate) fn inventory_ranks(self) -> Result<usize, Error> {
        add(mul(self.layers, 16)?, 3)
    }

    pub(crate) fn initialization_work(self, count: usize) -> Result<usize, Error> {
        // Bound the generated inventory, RNG draws, transcendentals and registrations.
        add(mul(count, 22)?, mul(self.inventory_len()?, 20)?)
    }
}

pub(crate) fn check_init_work(limits: ResourceLimits, requested: usize) -> Result<(), Error> {
    if requested > limits.max_work() {
        return Err(AutogradError::WorkLimit {
            pass: Pass::ParameterInsertion,
            requested,
            limit: limits.max_work(),
        }
        .into());
    }
    Ok(())
}
