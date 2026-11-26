use std::{cell::Cell, mem::size_of, rc::Rc};

use xynoos_autograd::ResourceLimits;

use crate::{Result, TrainingError};

const MIB: usize = 1024 * 1024;
const PARAMETERS: usize = 914_400 * size_of::<f32>();
const MOMENTS: usize = PARAMETERS * 2;
const ACCUMULATOR: usize = 914_400 * size_of::<f64>();
const TRANSFER: usize = PARAMETERS;
const TAPE: usize = 16_037_368;
const OPTIMIZER_STAGING: usize = PARAMETERS * 3;
const CHECKPOINT_IO: usize = 64 * 1024;
const RESUME_STAGING: usize = PARAMETERS * 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPhase {
    Initialization,
    Backward,
    Update,
    CheckpointWrite,
    Resume,
    Validation,
}

/// Conservative child-budget plan for the fixed 914,400-parameter reference.
/// External data/tokenizer allocations are reserved separately from autograd's
/// tensor ledger; this plan coordinates both under the 128 MiB run ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingMemoryPlan {
    data_allowance: usize,
    dataset_metadata: usize,
    dataset_buffers: usize,
    reader_buffers: usize,
    tokenizer_buffers: usize,
    runtime_metadata: usize,
    owned_cap: usize,
}

impl TrainingMemoryPlan {
    pub fn candidate(data_allowance: usize, owned_cap: usize) -> Result<Self> {
        if data_allowance == 0
            || data_allowance > 64 * MIB
            || owned_cap == 0
            || owned_cap > ResourceLimits::MAX_BYTES
        {
            return Err(TrainingError::InvalidConfiguration);
        }
        // A modest fixture allowance: each child is capped independently and
        // their sum cannot exceed the caller-selected allowance.
        let dataset_metadata = (data_allowance / 4).clamp(1, MIB);
        let dataset_buffers = (data_allowance / 4).clamp(1, MIB);
        let reader_buffers = (data_allowance / 32).clamp(1, 64 * 1024);
        let tokenizer_buffers = data_allowance
            .checked_sub(dataset_metadata + dataset_buffers + reader_buffers)
            .ok_or(TrainingError::InvalidConfiguration)?;
        let plan = Self {
            data_allowance,
            dataset_metadata,
            dataset_buffers,
            reader_buffers,
            tokenizer_buffers,
            runtime_metadata: 256 * 1024,
            owned_cap,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn candidate_test() -> Self {
        Self::candidate(4 * MIB, 128 * MIB).expect("fixed test plan is valid")
    }

    pub fn owned_cap(self) -> usize {
        self.owned_cap
    }
    pub fn data_allowance(self) -> usize {
        self.data_allowance
    }
    pub fn dataset_metadata_bytes(self) -> usize {
        self.dataset_metadata
    }
    pub fn dataset_buffer_bytes(self) -> usize {
        self.dataset_buffers
    }
    pub fn reader_buffer_bytes(self) -> usize {
        self.reader_buffers
    }
    pub fn tokenizer_buffer_bytes(self) -> usize {
        self.tokenizer_buffers
    }
    pub fn runtime_metadata_bytes(self) -> usize {
        self.runtime_metadata
    }
    pub fn persistent_bytes(self) -> usize {
        PARAMETERS + MOMENTS + ACCUMULATOR + self.data_allowance + self.runtime_metadata
    }

    pub fn phase_bytes(self, phase: MemoryPhase) -> usize {
        match phase {
            MemoryPhase::Initialization => 256 * 1024,
            MemoryPhase::Backward => TAPE + TRANSFER,
            MemoryPhase::Update => OPTIMIZER_STAGING + 2 * TRANSFER,
            MemoryPhase::CheckpointWrite => CHECKPOINT_IO,
            MemoryPhase::Resume => RESUME_STAGING,
            MemoryPhase::Validation => TAPE,
        }
    }

    pub fn validate(self) -> Result<()> {
        let persistent = self.persistent_bytes();
        let mut peak = persistent;
        for phase in [
            MemoryPhase::Initialization,
            MemoryPhase::Backward,
            MemoryPhase::Update,
            MemoryPhase::CheckpointWrite,
            MemoryPhase::Resume,
            MemoryPhase::Validation,
        ] {
            let requested = persistent
                .checked_add(self.phase_bytes(phase))
                .ok_or(TrainingError::ArithmeticOverflow)?;
            peak = peak.max(requested);
        }
        if peak > self.owned_cap {
            return Err(TrainingError::ResourceLimit {
                requested: peak,
                limit: self.owned_cap,
            });
        }
        let external = self
            .dataset_metadata
            .checked_add(self.dataset_buffers)
            .and_then(|v| v.checked_add(self.reader_buffers))
            .and_then(|v| v.checked_add(self.tokenizer_buffers))
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if external > self.data_allowance || self.data_allowance > 64 * MIB {
            return Err(TrainingError::ResourceLimit {
                requested: external,
                limit: self.data_allowance,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Ledger {
    current: Cell<usize>,
    peak: Cell<usize>,
    cap: usize,
}

#[derive(Debug, Clone)]
pub struct TrainingRunBudget {
    ledger: Rc<Ledger>,
    plan: TrainingMemoryPlan,
}

impl TrainingRunBudget {
    pub fn new(plan: TrainingMemoryPlan) -> Result<Self> {
        plan.validate()?;
        let persistent = plan.persistent_bytes();
        Ok(Self {
            ledger: Rc::new(Ledger {
                current: Cell::new(persistent),
                peak: Cell::new(persistent),
                cap: plan.owned_cap,
            }),
            plan,
        })
    }

    pub fn reserve_phase(&self, phase: MemoryPhase) -> Result<MemoryReservation> {
        let bytes = self.plan.phase_bytes(phase);
        let current = self.ledger.current.get();
        let requested = current
            .checked_add(bytes)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if requested > self.ledger.cap {
            return Err(TrainingError::ResourceLimit {
                requested,
                limit: self.ledger.cap,
            });
        }
        self.ledger.current.set(requested);
        self.ledger.peak.set(self.ledger.peak.get().max(requested));
        Ok(MemoryReservation {
            ledger: Rc::clone(&self.ledger),
            bytes,
        })
    }

    pub fn charged_bytes(&self) -> usize {
        self.ledger.current.get()
    }
    pub fn peak_charged_bytes(&self) -> usize {
        self.ledger.peak.get()
    }
    pub fn persistent_baseline_bytes(&self) -> usize {
        self.plan.persistent_bytes()
    }
    pub fn plan(&self) -> TrainingMemoryPlan {
        self.plan
    }
}

#[derive(Debug)]
pub struct MemoryReservation {
    ledger: Rc<Ledger>,
    bytes: usize,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.ledger
            .current
            .set(self.ledger.current.get() - self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_run_plan_preflights_and_temporary_phases_release() {
        let plan = TrainingMemoryPlan::candidate_test();
        assert!(plan.validate().is_ok());
        assert!(plan.data_allowance() <= 64 * MIB);
        let budget = TrainingRunBudget::new(plan).unwrap();
        let baseline = budget.charged_bytes();
        {
            let _backward = budget.reserve_phase(MemoryPhase::Backward).unwrap();
            assert!(budget.charged_bytes() > baseline);
        }
        assert_eq!(budget.charged_bytes(), baseline);
        assert!(budget.peak_charged_bytes() > baseline);
    }

    #[test]
    fn incompatible_child_budgets_reject_before_allocation() {
        assert!(TrainingMemoryPlan::candidate(65 * MIB, 128 * MIB).is_err());
        let plan = TrainingMemoryPlan::candidate(4 * MIB, 8 * MIB);
        assert!(plan.is_err());
    }
}
