use std::{
    cell::Cell,
    mem::{size_of, swap},
    rc::Rc,
};
use xynoos_tensor::{Tensor, TensorError, TensorLimits};

use crate::{AutogradError as Error, Pass};

/// Immutable limits shared by a parameter set, its tapes, and outstanding batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLimits {
    pub(crate) bytes: usize,
    pub(crate) values: usize,
    pub(crate) operations: usize,
    pub(crate) work: usize,
}

impl ResourceLimits {
    pub const MAX_BYTES: usize = 128 * 1024 * 1024;
    pub const MAX_VALUES: usize = 100_000;
    pub const MAX_OPERATIONS: usize = 100_000;
    pub const MAX_WORK: usize = 100_000_000;
    pub const MAX_EXPERIMENT_STEPS: usize = 10_000;

    /// Set effective limits. Requests above the project ceilings are clamped.
    pub fn new(bytes: usize, values: usize, operations: usize, work: usize) -> Self {
        Self {
            bytes: bytes.min(Self::MAX_BYTES),
            values: values.min(Self::MAX_VALUES),
            operations: operations.min(Self::MAX_OPERATIONS),
            work: work.min(Self::MAX_WORK),
        }
    }

    pub fn max_bytes(self) -> usize {
        self.bytes
    }
    pub fn max_values(self) -> usize {
        self.values
    }
    pub fn max_operations(self) -> usize {
        self.operations
    }
    pub fn max_work(self) -> usize {
        self.work
    }

    pub(crate) fn check_work(self, pass: Pass, requested: usize) -> Result<(), Error> {
        if requested > self.work {
            return Err(Error::WorkLimit {
                pass,
                requested,
                limit: self.work,
            });
        }
        Ok(())
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self::new(
            Self::MAX_BYTES,
            Self::MAX_VALUES,
            Self::MAX_OPERATIONS,
            Self::MAX_WORK,
        )
    }
}

pub(crate) fn add(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_add(right).ok_or(Error::ArithmeticOverflow)
}

pub(crate) fn mul(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_mul(right).ok_or(Error::ArithmeticOverflow)
}

#[derive(Debug)]
pub(crate) struct Budget {
    pub(crate) limits: ResourceLimits,
    used: Rc<Cell<usize>>,
}

impl Budget {
    pub(crate) fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            used: Rc::new(Cell::new(0)),
        }
    }

    pub(crate) fn used(&self) -> usize {
        self.used.get()
    }

    // Cell is exclusively an allocation ledger. It never grants access to values or gradients.
    fn reserve(&self, bytes: usize) -> Result<Charge, Error> {
        let mut charge = Charge {
            used: Rc::clone(&self.used),
            bytes: 0,
            limit: self.limits.bytes,
        };
        charge.resize(bytes)?;
        Ok(charge)
    }

    pub(crate) fn adopt(&self, tensor: Tensor) -> Result<OwnedTensor, Error> {
        let charge = self.reserve(tensor.retained_bytes()?)?;
        Ok(OwnedTensor {
            tensor,
            _charge: charge,
        })
    }

    /// Preflight requested tensor bytes, perform a fallible tensor operation,
    /// then reconcile the returned allocation while it is still detached.
    pub(crate) fn tensor(
        &self,
        rank: usize,
        len: usize,
        create: impl FnOnce() -> Result<Tensor, TensorError>,
    ) -> Result<OwnedTensor, Error> {
        let mut charge = self.reserve(tensor_bytes(rank, len)?)?;
        let tensor = create()?;
        charge.resize(tensor.retained_bytes()?)?;
        Ok(OwnedTensor {
            tensor,
            _charge: charge,
        })
    }

    /// Reserve and reconcile numerical capacity before running coordinate kernels.
    pub(crate) fn generate(
        &self,
        shape: &[usize],
        len: usize,
        mut coordinate: impl FnMut(usize) -> Result<f32, Error>,
    ) -> Result<OwnedTensor, Error> {
        let mut charge = self.reserve(tensor_bytes(shape.len(), len)?)?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|_| Error::AllocationFailed)?;
        charge.resize(tensor_bytes(shape.len(), data.capacity())?)?;
        for index in 0..len {
            data.push(coordinate(index)?);
        }
        let tensor = Tensor::from_vec(shape, data, TensorLimits::default())?;
        charge.resize(tensor.retained_bytes()?)?;
        Ok(OwnedTensor {
            tensor,
            _charge: charge,
        })
    }
}

fn tensor_bytes(rank: usize, elements: usize) -> Result<usize, Error> {
    let bytes = mul(elements, size_of::<f32>())?;
    if bytes > TensorLimits::MAX_BYTES {
        return Err(TensorError::PayloadLimitExceeded {
            bytes,
            limit: TensorLimits::MAX_BYTES,
        }
        .into());
    }
    add(bytes, mul(rank, size_of::<usize>())?)
}

#[derive(Debug)]
struct Charge {
    used: Rc<Cell<usize>>,
    bytes: usize,
    limit: usize,
}

impl Charge {
    fn resize(&mut self, bytes: usize) -> Result<(), Error> {
        let requested = add(self.used.get() - self.bytes, bytes)?;
        if requested > self.limit {
            return Err(Error::MemoryLimit {
                requested,
                limit: self.limit,
            });
        }
        self.used.set(requested);
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.used.set(self.used.get() - self.bytes);
    }
}

#[derive(Debug)]
pub(crate) struct OwnedTensor {
    pub(crate) tensor: Tensor,
    _charge: Charge,
}

#[derive(Debug)]
pub(crate) struct Table<T> {
    pub(crate) entries: Vec<T>,
    _charge: Option<Charge>,
}

impl<T> Table<T> {
    pub(crate) const fn empty() -> Self {
        Self {
            entries: Vec::new(),
            _charge: None,
        }
    }

    pub(crate) fn allocate(budget: &Budget, capacity: usize) -> Result<Self, Error> {
        if capacity == 0 {
            return Ok(Self::empty());
        }
        let mut charge = budget.reserve(mul(capacity, size_of::<T>())?)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|_| Error::AllocationFailed)?;
        charge.resize(mul(entries.capacity(), size_of::<T>())?)?;
        Ok(Self {
            entries,
            _charge: Some(charge),
        })
    }

    pub(crate) fn move_work(&self) -> usize {
        if self.entries.len() == self.entries.capacity() {
            self.entries.len()
        } else {
            0
        }
    }

    /// Detached replacement charges the temporary overlap with the original backing store.
    pub(crate) fn growth(&self, budget: &Budget, max: usize) -> Result<Option<Self>, Error> {
        let len = self.entries.len();
        if len < self.entries.capacity() {
            return Ok(None);
        }
        let minimum = add(len, 1)?;
        let capacity = mul(len, 2)?.max(minimum).min(max.max(minimum));
        match Self::allocate(budget, capacity) {
            Err(Error::MemoryLimit { .. }) if capacity > minimum => {
                Ok(Some(Self::allocate(budget, minimum)?))
            }
            result => result.map(Some),
        }
    }

    pub(crate) fn push(&mut self, replacement: Option<Self>, value: T) {
        if let Some(mut replacement) = replacement {
            replacement.entries.append(&mut self.entries);
            swap(self, &mut replacement);
        }
        self.entries.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_reconciliation_does_not_inflate_retained_charge() {
        let budget = Budget::new(ResourceLimits::new(8, 0, 0, 0));
        let mut reservation = budget.reserve(4).unwrap();
        assert!(matches!(
            reservation.resize(9),
            Err(Error::MemoryLimit { .. })
        ));
        assert_eq!(budget.used(), 4);
        drop(reservation);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn overflow_and_exact_metadata_capacity_are_checked() {
        assert_eq!(mul(usize::MAX, 2), Err(Error::ArithmeticOverflow));
        assert_eq!(add(usize::MAX, 1), Err(Error::ArithmeticOverflow));
        let budget = Budget::new(ResourceLimits::new(size_of::<u64>(), 0, 0, 0));
        let table = Table::<u64>::allocate(&budget, 1).unwrap();
        assert_eq!(budget.used(), size_of::<u64>());
        assert!(matches!(
            Table::<u8>::allocate(&budget, 1),
            Err(Error::MemoryLimit { .. })
        ));
        drop(table);
        assert_eq!(budget.used(), 0);
    }
}
