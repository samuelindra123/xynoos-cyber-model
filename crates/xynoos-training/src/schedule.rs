use xynoos_autograd::ResourceLimits;

use crate::{Result, TrainingError};

/// One-based linear warmup followed by linear decay, computed in binary64
/// from widened binary32 configuration values and narrowed once per update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearWarmupDecay {
    total_updates: u64,
    warmup_updates: u64,
    peak_rate: f32,
    final_rate: f32,
}

impl LinearWarmupDecay {
    pub fn new(
        total_updates: u64,
        warmup_updates: u64,
        peak_rate: f32,
        final_rate: f32,
    ) -> Result<Self> {
        if total_updates < 2
            || total_updates > ResourceLimits::MAX_EXPERIMENT_STEPS as u64
            || warmup_updates >= total_updates
            || !peak_rate.is_finite()
            || !final_rate.is_finite()
            || peak_rate <= 0.0
            || final_rate <= 0.0
            || final_rate > peak_rate
        {
            return Err(TrainingError::InvalidSchedule);
        }
        Ok(Self {
            total_updates,
            warmup_updates,
            peak_rate,
            final_rate,
        })
    }

    pub fn total_updates(self) -> u64 {
        self.total_updates
    }
    pub fn warmup_updates(self) -> u64 {
        self.warmup_updates
    }
    pub fn peak_rate(self) -> f32 {
        self.peak_rate
    }
    pub fn final_rate(self) -> f32 {
        self.final_rate
    }

    pub fn validate_decay(self, weight_decay: f32) -> Result<()> {
        if !weight_decay.is_finite()
            || weight_decay < 0.0
            || f64::from(self.peak_rate) * f64::from(weight_decay) >= 1.0
        {
            return Err(TrainingError::InvalidSchedule);
        }
        Ok(())
    }

    /// Return the rate for the next update after `completed_updates`.
    pub fn next_rate(self, completed_updates: u64) -> Result<f32> {
        let update = completed_updates
            .checked_add(1)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        if update > self.total_updates {
            return Err(TrainingError::ScheduleComplete);
        }
        let peak = f64::from(self.peak_rate);
        let final_rate = f64::from(self.final_rate);
        let value = if self.warmup_updates > 0 && update <= self.warmup_updates {
            peak * update as f64 / self.warmup_updates as f64
        } else if self.warmup_updates > 0 {
            final_rate
                + (peak - final_rate) * (self.total_updates - update) as f64
                    / (self.total_updates - self.warmup_updates) as f64
        } else {
            final_rate
                + (peak - final_rate) * (self.total_updates - update) as f64
                    / (self.total_updates - 1) as f64
        };
        let narrowed = value as f32;
        if !narrowed.is_finite() || narrowed <= 0.0 {
            return Err(TrainingError::InvalidSchedule);
        }
        Ok(narrowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(warmup: u64) -> LinearWarmupDecay {
        LinearWarmupDecay::new(5, warmup, 0.1, 0.01).unwrap()
    }

    #[test]
    fn one_based_boundaries_and_resume_are_deterministic() {
        let warm = schedule(2);
        assert_eq!(warm.next_rate(0).unwrap(), 0.05);
        assert_eq!(warm.next_rate(1).unwrap(), 0.1);
        assert_eq!(warm.next_rate(2).unwrap(), 0.07);
        assert_eq!(warm.next_rate(3).unwrap(), 0.04);
        assert_eq!(warm.next_rate(4).unwrap(), 0.01);
        assert_eq!(warm.next_rate(2).unwrap(), warm.next_rate(2).unwrap());
        assert_eq!(warm.next_rate(5), Err(TrainingError::ScheduleComplete));
    }

    #[test]
    fn zero_warmup_and_guards() {
        let schedule = schedule(0);
        assert_eq!(schedule.next_rate(0).unwrap(), 0.1);
        assert_eq!(schedule.next_rate(4).unwrap(), 0.01);
        assert!(schedule.validate_decay(0.01).is_ok());
        assert!(schedule.validate_decay(10.0).is_err());
        assert!(LinearWarmupDecay::new(1, 0, 0.1, 0.01).is_err());
        assert!(LinearWarmupDecay::new(5, 5, 0.1, 0.01).is_err());
        assert!(LinearWarmupDecay::new(5, 0, f32::NAN, 0.01).is_err());
        assert!(LinearWarmupDecay::new(5, 0, 0.1, 0.0).is_err());
        assert!(LinearWarmupDecay::new(5, 0, 0.1, 0.2).is_err());
    }
}
