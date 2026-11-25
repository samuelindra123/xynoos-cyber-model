//! Fixed Transformer graph with no public access to its recording tape.

use super::{
    add, mul, Error, GradientBatch, ParameterSet, ReferenceBatch, Tape, TapeMetrics, Tensor,
    ValueId,
};
use crate::ResourceLimits;

/// Opt in to the fixed TASK 005 candidate reverse-work ceiling.
///
/// The policy can only select a complete, inventory-checked reference graph
/// with 292 tokens, width 96, eight blocks, four heads and FFN width 256.
/// At the model entry point the context cap must be 128. The fixed graph
/// accepts only a batch of one sequence of length 32.
#[derive(Debug, Clone, Copy)]
pub struct CandidateReferenceExperimentPolicy;

impl CandidateReferenceExperimentPolicy {
    pub const VOCABULARY: usize = 292;
    pub const WIDTH: usize = 96;
    pub const LAYERS: usize = 8;
    pub const HEADS: usize = 4;
    pub const FFN: usize = 256;
    pub const CONTEXT: usize = 128;
    pub const BATCH_SHAPE: [usize; 2] = [1, 32];
    pub const BACKWARD_WORK: usize = 160_000_000;

    fn inventory_shapes() -> impl Iterator<Item = &'static [usize]> {
        const D: usize = CandidateReferenceExperimentPolicy::WIDTH;
        const F: usize = CandidateReferenceExperimentPolicy::FFN;
        const BLOCK: [&[usize]; 9] = [
            &[D],
            &[D, D],
            &[D, D],
            &[D, D],
            &[D, D],
            &[D],
            &[D, F],
            &[D, F],
            &[F, D],
        ];
        [&[Self::VOCABULARY, D][..]]
            .into_iter()
            .chain(BLOCK.into_iter().cycle().take(Self::LAYERS * BLOCK.len()))
            .chain([&[D][..]])
    }

    fn validate_inventory(parameters: &ParameterSet) -> Result<(), Error> {
        if !parameters.has_candidate_inventory() || parameters.len() != 74 {
            return Err(Error::InvalidCandidateInventory);
        }
        for (index, shape) in Self::inventory_shapes().enumerate() {
            if parameters.value(parameters.id_at(index)?)?.shape() != shape {
                return Err(Error::InvalidCandidateInventory);
            }
        }
        Ok(())
    }
}

/// A closed reference Transformer graph whose only reverse root is its loss.
#[derive(Debug)]
pub struct ReferenceTransformerEvaluation<'p> {
    tape: Tape<'p>,
    logits: ValueId,
    loss: ValueId,
    candidate: bool,
}

impl<'p> ReferenceTransformerEvaluation<'p> {
    /// Record the reference graph under the parameter set's ordinary limits.
    pub fn record(
        parameters: &'p ParameterSet,
        batch: ReferenceBatch<'_>,
        vocabulary: usize,
        heads: usize,
        layers: usize,
    ) -> Result<Self, Error> {
        Self::record_graph(parameters, batch, vocabulary, heads, layers, false)
    }

    /// Record the fixed candidate after checking the actual inventory and batch.
    pub fn record_candidate(
        parameters: &'p ParameterSet,
        batch: ReferenceBatch<'_>,
        _policy: CandidateReferenceExperimentPolicy,
    ) -> Result<Self, Error> {
        if batch.shape != CandidateReferenceExperimentPolicy::BATCH_SHAPE {
            return Err(Error::InvalidCandidateBatchShape {
                actual: batch.shape,
            });
        }
        CandidateReferenceExperimentPolicy::validate_inventory(parameters)?;
        Self::record_graph(
            parameters,
            batch,
            CandidateReferenceExperimentPolicy::VOCABULARY,
            CandidateReferenceExperimentPolicy::HEADS,
            CandidateReferenceExperimentPolicy::LAYERS,
            true,
        )
    }

    fn record_graph(
        parameters: &'p ParameterSet,
        batch: ReferenceBatch<'_>,
        vocabulary: usize,
        heads: usize,
        layers: usize,
        candidate: bool,
    ) -> Result<Self, Error> {
        let expected_entries = add(mul(layers, 9)?, 2)?;
        if layers == 0 || parameters.len() != expected_entries {
            return Err(Error::InvalidReferenceShape);
        }
        let mut tape = Tape::for_reference(parameters, batch, vocabulary, heads)?;
        let embedding = tape.parameter(parameters.id_at(0)?)?;
        let mut x = tape.embedding(embedding)?;
        for layer in 0..layers {
            let start = add(mul(layer, 9)?, 1)?;
            x = block(&mut tape, x, start)?;
        }
        let gain = tape.parameter(parameters.id_at(expected_entries - 1)?)?;
        let h = tape.rms_norm(x, gain)?;
        let logits = tape.project(h, embedding, true)?;
        let loss = tape.next_token_loss(logits)?;
        Ok(Self {
            tape,
            logits,
            loss,
            candidate,
        })
    }

    pub fn logits(&self) -> &Tensor {
        self.tape.value(self.logits).expect("recorded logits")
    }

    pub fn loss(&self) -> f32 {
        self.tape.value(self.loss).expect("recorded loss").data()[0]
    }

    pub fn metrics(&self) -> Result<TapeMetrics, Error> {
        self.tape.metrics()
    }

    /// Run the complete loss backward. On failure the tape is dropped.
    pub fn backward(mut self) -> Result<ReferenceTransformerBackward<'p>, Error> {
        let legacy = self.tape.parameters.limits().max_work();
        let limit = if self.candidate && legacy == ResourceLimits::MAX_WORK {
            CandidateReferenceExperimentPolicy::BACKWARD_WORK
        } else {
            legacy
        };
        self.tape.backward_with_limit(self.loss, 1.0, limit)?;
        Ok(ReferenceTransformerBackward(self.tape))
    }
}

fn block(tape: &mut Tape<'_>, x: ValueId, start: usize) -> Result<ValueId, Error> {
    let mut param = |offset| tape.parameter(tape.parameters.id_at(add(start, offset)?)?);
    let [ag, wq, wk, wv, wo, fg, wg, wu, wd] = [
        param(0)?,
        param(1)?,
        param(2)?,
        param(3)?,
        param(4)?,
        param(5)?,
        param(6)?,
        param(7)?,
        param(8)?,
    ];
    let u = tape.rms_norm(x, ag)?;
    let q = tape.project(u, wq, false)?;
    let q = tape.rope(q)?;
    let k = tape.project(u, wk, false)?;
    let k = tape.rope(k)?;
    let v = tape.project(u, wv, false)?;
    let attention = tape.causal_attention(q, k, v)?;
    let projection = tape.project(attention, wo, false)?;
    let xa = tape.masked_residual(x, projection)?;
    let r = tape.rms_norm(xa, fg)?;
    let gate = tape.project(r, wg, false)?;
    let value = tape.project(r, wu, false)?;
    let gated = tape.swiglu(gate, value)?;
    let down = tape.project(gated, wd, false)?;
    tape.masked_residual(xa, down)
}

/// Completed reverse pass with its owned tape awaiting gradient transfer.
#[derive(Debug)]
pub struct ReferenceTransformerBackward<'p>(Tape<'p>);

impl ReferenceTransformerBackward<'_> {
    /// Consume the tape and move its parameter adjoints into a gradient batch.
    pub fn into_gradients(self) -> Result<(GradientBatch, TapeMetrics), Error> {
        let metrics = self.0.metrics()?;
        Ok((self.0.into_parameter_gradients()?, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::Kernel;
    use crate::tape::Op;

    const INPUT: [u32; 32] = [3; 32];
    const BATCH: ReferenceBatch<'_> = ReferenceBatch {
        shape: [1, 32],
        input: &INPUT,
        target: &INPUT,
        valid: None,
        loss: None,
    };

    #[test]
    fn lower_level_candidate_rejects_wrong_inventory_and_batch_before_recording() {
        let mut set = ParameterSet::with_candidate_reference_inventory(Default::default()).unwrap();
        for _ in 0..74 {
            set.insert_generated(&[], |_| Ok(1.)).unwrap();
        }
        let baseline = set.accounted_bytes();
        let revision = set.revision();
        let policy = CandidateReferenceExperimentPolicy;
        assert_eq!(
            ReferenceTransformerEvaluation::record_candidate(&set, BATCH, policy).unwrap_err(),
            Error::InvalidCandidateInventory
        );
        let actual = [2, 16];
        assert_eq!(
            ReferenceTransformerEvaluation::record_candidate(
                &set,
                ReferenceBatch {
                    shape: actual,
                    ..BATCH
                },
                policy,
            )
            .unwrap_err(),
            Error::InvalidCandidateBatchShape { actual }
        );
        assert_eq!(set.accounted_bytes(), baseline);
        assert_eq!(set.peak_accounted_bytes(), baseline);
        assert_eq!(set.revision(), revision);
        assert!((0..74).all(|i| set.gradient(set.id_at(i).unwrap()).unwrap().is_none()));
        println!("TASK005-R scoped low-level inventory and batch failures before recording");
    }

    #[test]
    #[ignore = "bounded hosted candidate transaction and fault-injection controls"]
    fn candidate_failures_release_tape_and_preserve_transactions() {
        let policy = CandidateReferenceExperimentPolicy;
        let mut set = ParameterSet::with_candidate_reference_inventory(Default::default()).unwrap();
        for shape in CandidateReferenceExperimentPolicy::inventory_shapes() {
            let weight = if shape.len() == 1 { 1.0 } else { 0.01 };
            set.insert_generated(shape, |_| Ok(weight)).unwrap();
        }
        let observe = set.budget.observe_charged_bytes();
        let base = observe();
        let ids: Vec<_> = (0..74).map(|i| set.id_at(i).unwrap()).collect();
        let revision = set.revision();

        let evaluation =
            ReferenceTransformerEvaluation::record_candidate(&set, BATCH, policy).unwrap();
        let kernels = &evaluation.tape.reference().unwrap().nodes.entries;
        assert!(matches!(kernels[0].kernel, Kernel::Gather(0)));
        assert!(matches!(
            kernels[kernels.len() - 2].kernel,
            Kernel::Dense(_, 0, true)
        ));
        assert!(matches!(
            evaluation.tape.nodes.entries[0].op,
            Op::Parameter(0)
        ));
        assert_eq!(
            evaluation
                .tape
                .nodes
                .entries
                .iter()
                .filter(|node| matches!(node.op, Op::Parameter(0)))
                .count(),
            1
        );
        set.budget.fail_allocation_after(8);
        assert_eq!(evaluation.backward().unwrap_err(), Error::AllocationFailed);
        assert_eq!(observe(), base);
        assert!(ids.iter().all(|&id| set.gradient(id).unwrap().is_none()));

        let evaluation =
            ReferenceTransformerEvaluation::record_candidate(&set, BATCH, policy).unwrap();
        let reverse = evaluation.backward().unwrap();
        set.budget.fail_allocation_after(0);
        assert_eq!(
            reverse.into_gradients().unwrap_err(),
            Error::AllocationFailed
        );
        assert_eq!(observe(), base);
        assert!(ids.iter().all(|&id| set.gradient(id).unwrap().is_none()));
        assert_eq!(set.revision(), revision);

        let evaluate = |set: &ParameterSet| {
            ReferenceTransformerEvaluation::record_candidate(set, BATCH, policy)
                .unwrap()
                .backward()
                .unwrap()
                .into_gradients()
                .unwrap()
                .0
        };
        let first = evaluate(&set);
        let second = evaluate(&set);
        set.accumulate_gradients(first).unwrap();
        let initial_gradients: Vec<_> = ids
            .iter()
            .map(|&id| set.gradient(id).unwrap().unwrap().data().to_vec())
            .collect();
        let committed = observe()
            - (4 * 914_400
                + 131 * size_of::<usize>()
                + 74 * size_of::<Option<super::super::OwnedTensor>>());
        set.budget.fail_allocation_after(4);
        assert_eq!(
            set.accumulate_gradients(second),
            Err(Error::AllocationFailed)
        );
        assert_eq!(observe(), committed);
        assert_eq!(set.revision(), revision);
        for (&id, original) in ids.iter().zip(&initial_gradients) {
            assert_eq!(set.gradient(id).unwrap().unwrap().data(), original);
            let value = set.value(id).unwrap();
            let expected = if value.rank() == 1 { 1.0 } else { 0.01 };
            assert!(value.data().iter().all(|&v| v == expected));
        }

        let stale = evaluate(&set);
        // Inserting an extra parameter advances the revision without an optimizer step.
        let new_id = set.insert_generated(&[], |_| Ok(7.)).unwrap();
        let current = set.revision();
        assert_ne!(revision, current);
        let before_reject = observe();
        assert_eq!(
            set.accumulate_gradients(stale),
            Err(Error::StaleGradientBatch {
                batch: revision,
                current
            })
        );
        let released = before_reject - observe();
        assert_eq!(
            released,
            4 * 914_400
                + 131 * size_of::<usize>()
                + 74 * size_of::<Option<super::super::OwnedTensor>>()
        );
        assert_eq!(set.value(new_id).unwrap().data(), [7.]);
        assert!(set.gradient(new_id).unwrap().is_none());
        for (&id, original) in ids.iter().zip(&initial_gradients) {
            assert_eq!(set.gradient(id).unwrap().unwrap().data(), original);
        }
        drop(set);
        assert_eq!(observe(), 0);
        println!("TASK005-R candidate faults: tied snapshot identity, reverse and transfer AllocationFailed, staged accumulation AllocationFailed, StaleGradientBatch, ledger after drop={}", observe());
    }
}
