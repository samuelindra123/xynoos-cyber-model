use xynoos_autograd::{
    AutogradError as Error, GradientBatch, ParameterId, ParameterSet, Pass, ResourceLimits, Sgd,
    Stage, Tape, ValueId,
};
use xynoos_tensor::{Tensor, TensorError};

fn tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor::from_data(shape, values, Default::default()).unwrap()
}

fn scalar(value: f32) -> Tensor {
    tensor(&[], &[value])
}

fn set() -> ParameterSet {
    ParameterSet::new(ResourceLimits::default()).unwrap()
}

fn gradient<'a>(tape: &'a Tape<'_>, value: ValueId) -> &'a [f32] {
    tape.gradient(value).unwrap().unwrap().data()
}

fn batch(parameters: &ParameterSet, id: ParameterId, seed: f32) -> GradientBatch {
    let mut tape = Tape::new(parameters).unwrap();
    let root = tape.parameter(id).unwrap();
    tape.backward_with_seed(root, seed).unwrap();
    tape.into_parameter_gradients().unwrap()
}

#[test]
fn identifiers_reject_foreign_and_dropped_owners() {
    let mut first = set();
    let id = first.insert(scalar(1.0)).unwrap();
    let mut second = set();
    second.insert(scalar(2.0)).unwrap();
    assert!(matches!(second.value(id), Err(Error::InvalidParameterId)));
    let retired = {
        let mut tape = Tape::new(&first).unwrap();
        tape.parameter(id).unwrap()
    };
    let mut tape = Tape::new(&first).unwrap();
    let local = tape.parameter(id).unwrap();
    assert!(matches!(tape.value(retired), Err(Error::InvalidValueId)));
    assert_eq!(tape.add(local, retired), Err(Error::InvalidValueId));
    assert_eq!(
        Tape::new(&second).unwrap().parameter(id),
        Err(Error::InvalidParameterId)
    );
}

#[test]
fn forward_snapshots_and_shared_edges_obey_the_chain_rule() {
    let mut parameters = set();
    let id = parameters.insert(scalar(2.0)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let x = tape.parameter(id).unwrap();
    let xx = tape.multiply(x, x).unwrap();
    let x_plus_xx = tape.add(x, xx).unwrap();
    let scale = tape.scale(x_plus_xx, 3.0).unwrap();
    let other = tape.subtract(x_plus_xx, x).unwrap();
    let root = tape.add(scale, other).unwrap();
    tape.backward(root).unwrap();
    assert_eq!(tape.value(xx).unwrap().data(), [4.0]);
    assert_eq!(tape.value(root).unwrap().data(), [22.0]);
    assert_eq!(gradient(&tape, x_plus_xx), [4.0]);
    assert_eq!(gradient(&tape, x), [19.0]);
    assert_eq!(tape.backward(root), Err(Error::BackwardAlreadyRun));
    assert_eq!(tape.add(x, x), Err(Error::TapeSealed));
}

#[test]
fn polynomial_with_constant_leaf() {
    let mut parameters = set();
    let p = parameters.insert(scalar(-0.5)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let x = tape.parameter(p).unwrap();
    let mut external = scalar(2.0);
    let constant = tape.input(&external).unwrap();
    *external.get_flat_mut(0).unwrap() = 10.0;
    let triple = tape.scale(x, 3.0).unwrap();
    let affine = tape.add(triple, constant).unwrap();
    let root = tape.multiply(affine, affine).unwrap();
    tape.backward(root).unwrap();
    assert_eq!(tape.value(root).unwrap().data(), [0.25]);
    assert_eq!(gradient(&tape, x), [3.0]);
    assert!(tape.gradient(constant).unwrap().is_none());
}

#[test]
fn vector_product_shared_path_and_mean() {
    let mut parameters = set();
    let a_id = parameters.insert(tensor(&[3], &[1.0, 2.0, 4.0])).unwrap();
    let b = parameters.insert(tensor(&[3], &[2.0, -1.0, 0.5])).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let a = tape.parameter(a_id).unwrap();
    let b = tape.parameter(b).unwrap();
    let ab = tape.multiply(a, b).unwrap();
    let z = tape.add(ab, a).unwrap();
    let root = tape.sum(z).unwrap();
    tape.backward(root).unwrap();
    assert_eq!(gradient(&tape, a), [3.0, 0.0, 1.5]);
    assert_eq!(gradient(&tape, b), [1.0, 2.0, 4.0]);

    let mut mean_tape = Tape::new(&parameters).unwrap();
    let a = mean_tape.parameter(a_id).unwrap();
    let root = mean_tape.mean(a).unwrap();
    mean_tape.backward_with_seed(root, -6.0).unwrap();
    assert_eq!(gradient(&mean_tape, a), [-2.0; 3]);
}

#[test]
fn dot_and_transpose_hand_derivatives() {
    let mut parameters = set();
    let a = parameters.insert(tensor(&[2], &[2.0, -3.0])).unwrap();
    let b = parameters.insert(tensor(&[2], &[4.0, 1.5])).unwrap();
    let matrix = parameters.insert(tensor(&[2, 3], &[0.0; 6])).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let a = tape.parameter(a).unwrap();
    let b = tape.parameter(b).unwrap();
    let root = tape.dot(a, b).unwrap();
    tape.backward_with_seed(root, 2.0).unwrap();
    assert_eq!(tape.value(root).unwrap().data(), [3.5]);
    assert_eq!(gradient(&tape, a), [8.0, 3.0]);
    assert_eq!(gradient(&tape, b), [4.0, -6.0]);

    let mut tape = Tape::new(&parameters).unwrap();
    let matrix = tape.parameter(matrix).unwrap();
    let t = tape.transpose(matrix).unwrap();
    let weights = tape
        .input(&tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let product = tape.multiply(t, weights).unwrap();
    let root = tape.sum(product).unwrap();
    tape.backward(root).unwrap();
    assert_eq!(gradient(&tape, matrix), [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn mse_accepts_default_nonunit_negative_and_downstream_seeds() {
    let mut parameters = set();
    let p = parameters.insert(tensor(&[2], &[2.0, -1.0])).unwrap();
    for (seed, downstream, expected) in [
        (1.0, None, [1.0, -3.0]),
        (2.5, None, [2.5, -7.5]),
        (-2.0, None, [-2.0, 6.0]),
        (2.0, Some(3.0), [6.0, -18.0]),
    ] {
        let mut tape = Tape::new(&parameters).unwrap();
        let prediction = tape.parameter(p).unwrap();
        let target = tape.input(&tensor(&[2], &[1.0, 2.0])).unwrap();
        let loss = tape.mse(prediction, target).unwrap();
        let root = downstream.map_or(loss, |c| tape.scale(loss, c).unwrap());
        if seed == 1.0 {
            tape.backward(root).unwrap();
        } else {
            tape.backward_with_seed(root, seed).unwrap();
        }
        assert_eq!(tape.value(loss).unwrap().data(), [5.0]);
        assert_eq!(gradient(&tape, prediction), expected);
        assert!(tape.gradient(target).unwrap().is_none());
    }
}

#[test]
fn mse_requires_a_constant_leaf_and_equal_nonempty_shapes() {
    let mut parameters = set();
    let id = parameters.insert(scalar(1.0)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let p = tape.parameter(id).unwrap();
    assert_eq!(tape.mse(p, p), Err(Error::ConstantTargetRequired));
    let target = tape.input(&scalar(0.0)).unwrap();
    let expression = tape.scale(target, 1.0).unwrap();
    assert_eq!(tape.mse(p, expression), Err(Error::ConstantTargetRequired));
    let empty = tape.input(&tensor(&[0], &[])).unwrap();
    assert_eq!(tape.mse(empty, empty), Err(Error::EmptyReduction));
    assert!(matches!(
        tape.mse(p, empty),
        Err(Error::Tensor(TensorError::ShapeMismatch { .. }))
    ));
    assert!(tape.mse(p, target).is_ok());
}

#[test]
fn empty_sum_dot_and_mean() {
    let mut parameters = set();
    let x = parameters.insert(tensor(&[0], &[])).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let x = tape.parameter(x).unwrap();
    assert_eq!(tape.mean(x), Err(Error::EmptyReduction));
    let sum = tape.sum(x).unwrap();
    let dot = tape.dot(x, x).unwrap();
    assert_eq!(tape.value(sum).unwrap().data()[0].to_bits(), 0);
    assert_eq!(tape.value(dot).unwrap().data()[0].to_bits(), 0);
    let root = tape.add(sum, dot).unwrap();
    tape.backward(root).unwrap();
    let grad = tape.gradient(x).unwrap().unwrap();
    assert_eq!(grad.shape(), [0]);
    assert!(grad.data().is_empty());
}

#[test]
fn matrix_hand_derivatives_and_zero_dimensions() {
    for (m, k, n) in [(2, 1, 3), (2, 0, 3), (0, 2, 3), (2, 3, 0), (0, 0, 0)] {
        let mut parameters = set();
        let a_values = if (m, k, n) == (2, 1, 3) {
            vec![2.0, -1.0]
        } else {
            vec![1.0; m * k]
        };
        let a = parameters.insert(tensor(&[m, k], &a_values)).unwrap();
        let b = parameters
            .insert(tensor(&[k, n], &[1.0; 6][..k * n]))
            .unwrap();
        let mut tape = Tape::new(&parameters).unwrap();
        let a = tape.parameter(a).unwrap();
        let b = tape.parameter(b).unwrap();
        let product = tape.matmul(a, b).unwrap();
        assert_eq!(tape.value(product).unwrap().shape(), [m, n]);
        let weights = tape
            .input(&tensor(&[m, n], &[1.0, -1.0, 3.0, 2.0, 5.0, -6.0][..m * n]))
            .unwrap();
        let weighted = tape.multiply(product, weights).unwrap();
        let root = tape.sum(weighted).unwrap();
        tape.backward(root).unwrap();
        assert_eq!(tape.gradient(a).unwrap().unwrap().shape(), [m, k]);
        assert_eq!(tape.gradient(b).unwrap().unwrap().shape(), [k, n]);
        if (m, k, n) == (2, 1, 3) {
            assert_eq!(gradient(&tape, a), [3.0, 1.0]);
            assert_eq!(gradient(&tape, b), [0.0, -7.0, 12.0]);
        } else {
            assert_eq!(gradient(&tape, a), vec![0.0; m * k]);
            assert_eq!(gradient(&tape, b), vec![0.0; k * n]);
        }
    }
}

#[test]
fn nonfinite_values_and_failure_lifecycle() {
    let mut parameters = set();
    assert!(matches!(
        parameters.insert(scalar(f32::INFINITY)),
        Err(Error::NonFinite {
            stage: Stage::Parameter,
            ..
        })
    ));
    assert_eq!(parameters.accounted_bytes(), 0);
    let p = parameters.insert(scalar(0.0)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            tape.input(&scalar(bad)),
            Err(Error::NonFinite {
                stage: Stage::Input,
                ..
            })
        ));
    }
    let x = tape.parameter(p).unwrap();
    assert!(matches!(
        tape.scale(x, f32::NAN),
        Err(Error::NonFinite {
            stage: Stage::Scale,
            ..
        })
    ));
    let huge = tape.input(&scalar(f32::MAX)).unwrap();
    assert!(matches!(
        tape.add(huge, huge),
        Err(Error::NonFinite {
            stage: Stage::Forward,
            ..
        })
    ));
    assert!(matches!(
        tape.mse(x, huge),
        Err(Error::NonFinite {
            stage: Stage::Forward,
            ..
        })
    ));
    let root = tape.scale(x, f32::MAX).unwrap();
    let bytes = parameters.accounted_bytes();
    assert!(matches!(
        tape.backward_with_seed(root, 2.0),
        Err(Error::NonFinite {
            stage: Stage::Contribution,
            ..
        })
    ));
    assert_eq!(parameters.accounted_bytes(), bytes);
    assert!(matches!(tape.gradient(x), Err(Error::BackwardRequired)));
    assert_eq!(tape.backward(root), Err(Error::BackwardAlreadyRun));
    assert_eq!(tape.sum(x), Err(Error::TapeSealed));
    assert!(matches!(
        tape.into_parameter_gradients(),
        Err(Error::BackwardRequired)
    ));
    assert!(parameters.gradient(p).unwrap().is_none());
}

#[test]
fn backward_rejects_bad_root_and_seed() {
    let parameters = set();
    let mut tape = Tape::new(&parameters).unwrap();
    let vector = tape.input(&tensor(&[1], &[1.0])).unwrap();
    assert_eq!(tape.backward(vector), Err(Error::NonScalarRoot { rank: 1 }));
    let mut tape = Tape::new(&parameters).unwrap();
    let root = tape.input(&scalar(0.0)).unwrap();
    assert!(matches!(
        tape.backward_with_seed(root, f32::NEG_INFINITY),
        Err(Error::NonFinite {
            stage: Stage::Seed,
            ..
        })
    ));
    let mut tape = Tape::new(&parameters).unwrap();
    let root = tape.input(&scalar(0.0)).unwrap();
    tape.backward(root).unwrap();
    assert!(tape.gradient(root).unwrap().is_none());
}

#[test]
fn batches_merge_snapshots_accumulate_microbatches_and_check_revisions() {
    let mut parameters = set();
    let p = parameters.insert(scalar(2.0)).unwrap();
    let disconnected = parameters.insert(scalar(-7.0)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let p1 = tape.parameter(p).unwrap();
    let p2 = tape.parameter(p).unwrap();
    let other = tape.parameter(disconnected).unwrap();
    let root = tape.multiply(p1, p2).unwrap();
    tape.backward(root).unwrap();
    assert!(tape.gradient(other).unwrap().is_none());
    assert_eq!(gradient(&tape, p1), [2.0]);
    assert_eq!(gradient(&tape, p2), [2.0]);
    let first = tape.into_parameter_gradients().unwrap();
    let second = batch(&parameters, p, 3.0);
    let stale = batch(&parameters, p, 1.0);
    let revision = parameters.revision();
    assert_eq!(
        set().accumulate_gradients(batch(&parameters, p, 1.0)),
        Err(Error::WrongParameterSet)
    );
    parameters.zero_grad().unwrap();
    parameters.accumulate_gradients(first).unwrap();
    parameters.accumulate_gradients(second).unwrap();
    assert_eq!(parameters.revision(), revision);
    assert_eq!(parameters.gradient(p).unwrap().unwrap().data(), [7.0]);
    Sgd::new(0.25).unwrap().step(&mut parameters).unwrap();
    assert_eq!(parameters.value(p).unwrap().data(), [0.25]);
    assert_eq!(parameters.value(disconnected).unwrap().data(), [-7.0]);
    assert!(parameters.gradient(p).unwrap().is_none());
    assert!(parameters.gradient(disconnected).unwrap().is_none());
    assert_eq!(parameters.revision(), revision + 1);
    assert_eq!(
        parameters.accumulate_gradients(stale),
        Err(Error::StaleGradientBatch {
            batch: revision,
            current: revision + 1
        })
    );
    let stale = batch(&parameters, p, 1.0);
    parameters.insert(scalar(1.0)).unwrap();
    assert!(matches!(
        parameters.accumulate_gradients(stale),
        Err(Error::StaleGradientBatch { .. })
    ));
}

#[test]
fn sgd_validates_rates_missing_gradients_and_empty_set() {
    for lr in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(matches!(Sgd::new(lr), Err(Error::InvalidLearningRate)));
    }
    let opt = Sgd::new(0.5).unwrap();
    let mut parameters = set();
    opt.step(&mut parameters).unwrap();
    assert_eq!(parameters.revision(), 0);
    let p = parameters.insert(scalar(3.0)).unwrap();
    let rev = parameters.revision();
    assert_eq!(
        opt.step(&mut parameters),
        Err(Error::MissingGradient { parameter: 0 })
    );
    assert_eq!(parameters.value(p).unwrap().data(), [3.0]);
    assert_eq!(parameters.revision(), rev);
    parameters.zero_grad().unwrap();
    opt.step(&mut parameters).unwrap();
    assert_eq!(parameters.value(p).unwrap().data(), [3.0]);
}

#[test]
fn sgd_and_accumulation_roll_back_when_later_candidate_overflows() {
    let mut parameters = set();
    let a = parameters.insert(scalar(0.0)).unwrap();
    let b = parameters.insert(scalar(0.0)).unwrap();
    let make_batch = |parameters: &ParameterSet| {
        let mut tape = Tape::new(parameters).unwrap();
        let a = tape.parameter(a).unwrap();
        let b = tape.parameter(b).unwrap();
        let scaled = tape.scale(b, f32::MAX).unwrap();
        let root = tape.add(a, scaled).unwrap();
        tape.backward(root).unwrap();
        tape.into_parameter_gradients().unwrap()
    };
    let initial = make_batch(&parameters);
    parameters.accumulate_gradients(initial).unwrap();
    let bytes = parameters.accounted_bytes();
    let revision = parameters.revision();
    let incoming = make_batch(&parameters);
    assert!(matches!(
        parameters.accumulate_gradients(incoming),
        Err(Error::NonFinite {
            stage: Stage::Gradient,
            ..
        })
    ));
    assert!(matches!(
        Sgd::new(2.0).unwrap().step(&mut parameters),
        Err(Error::NonFinite {
            stage: Stage::Candidate,
            ..
        })
    ));
    assert_eq!(parameters.gradient(a).unwrap().unwrap().data(), [1.0]);
    assert_eq!(parameters.gradient(b).unwrap().unwrap().data(), [f32::MAX]);
    assert_eq!(parameters.value(a).unwrap().data(), [0.0]);
    assert_eq!(parameters.value(b).unwrap().data(), [0.0]);
    assert_eq!(parameters.accounted_bytes(), bytes);
    assert_eq!(parameters.revision(), revision);
}

#[test]
fn overflow_while_merging_snapshots_releases_the_tape() {
    let mut parameters = set();
    let p = parameters.insert(scalar(0.0)).unwrap();
    let bytes = parameters.accounted_bytes();
    let mut tape = Tape::new(&parameters).unwrap();
    let first = tape.parameter(p).unwrap();
    let second = tape.parameter(p).unwrap();
    let left = tape.scale(first, f32::MAX).unwrap();
    let right = tape.scale(second, f32::MAX).unwrap();
    let root = tape.add(left, right).unwrap();
    tape.backward(root).unwrap();
    assert!(matches!(
        tape.into_parameter_gradients(),
        Err(Error::NonFinite {
            stage: Stage::Gradient,
            ..
        })
    ));
    assert_eq!(parameters.accounted_bytes(), bytes);
}

#[test]
fn resource_policy_clamps_and_charges_simultaneous_tapes_and_batches() {
    assert_eq!(
        ResourceLimits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX),
        ResourceLimits::default()
    );
    let mut parameters = set();
    let p = parameters.insert(scalar(0.0)).unwrap();
    let base = parameters.accounted_bytes();
    let first = batch(&parameters, p, 1.0);
    let one_batch = parameters.accounted_bytes() - base;
    assert!(one_batch > size_of::<f32>());
    let second = batch(&parameters, p, 2.0);
    assert_eq!(parameters.accounted_bytes(), base + 2 * one_batch);
    drop(second);
    let mut tape = Tape::new(&parameters).unwrap();
    tape.parameter(p).unwrap();
    let tape_bytes = parameters.accounted_bytes();
    let mut other = Tape::new(&parameters).unwrap();
    other.parameter(p).unwrap();
    assert_eq!(
        parameters.accounted_bytes(),
        2 * tape_bytes - base - one_batch
    );
    drop(other);
    drop(tape);
    parameters.accumulate_gradients(first).unwrap();
    assert_eq!(parameters.accounted_bytes(), base + size_of::<f32>());
}

#[test]
fn exact_count_and_work_boundaries_and_empty_matmul_cost() {
    let parameters = ParameterSet::new(ResourceLimits::new(4096, 2, 1, 8)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let x = tape.input(&scalar(1.0)).unwrap();
    let root = tape.scale(x, 2.0).unwrap();
    assert_eq!(
        tape.scale(root, 2.0),
        Err(Error::OperationLimit { limit: 1 })
    );
    assert_eq!(
        tape.input(&scalar(0.0)),
        Err(Error::ValueLimit { limit: 2 })
    );
    tape.backward(root).unwrap();
    let mut parameters = ParameterSet::new(ResourceLimits::new(4096, 2, 1, 8)).unwrap();
    let p = parameters.insert(scalar(1.0)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let x = tape.parameter(p).unwrap();
    let root = tape.scale(x, 2.0).unwrap();
    assert!(matches!(
        tape.backward(root),
        Err(Error::WorkLimit {
            pass: Pass::Backward,
            ..
        })
    ));

    for work in [23, 24] {
        let parameters = ParameterSet::new(ResourceLimits::new(4096, 10, 10, work)).unwrap();
        let mut tape = Tape::new(&parameters).unwrap();
        let a = tape.input(&tensor(&[2, 0], &[])).unwrap();
        let b = tape.input(&tensor(&[0, 3], &[])).unwrap();
        let product = tape.matmul(a, b);
        if work == 23 {
            assert_eq!(
                product,
                Err(Error::WorkLimit {
                    pass: Pass::Forward,
                    requested: 24,
                    limit: 23
                })
            );
        } else {
            assert_eq!(tape.value(product.unwrap()).unwrap().data(), [0.0; 6]);
        }
    }
}

#[test]
fn matrix_estimate_rejects_checked_overflow_without_allocating_payload() {
    let parameters = set();
    let mut tape = Tape::new(&parameters).unwrap();
    let a = tape.input(&tensor(&[usize::MAX, 0], &[])).unwrap();
    let b = tape.input(&tensor(&[0, 2], &[])).unwrap();
    let bytes = parameters.accounted_bytes();
    assert_eq!(tape.matmul(a, b), Err(Error::ArithmeticOverflow));
    assert_eq!(parameters.accounted_bytes(), bytes);
}

#[test]
fn independent_work_budgets_for_accumulation_and_optimizer() {
    let mut parameters = ParameterSet::new(ResourceLimits::new(4096, 5, 5, 7)).unwrap();
    let p = parameters.insert(scalar(2.0)).unwrap();
    let one = batch(&parameters, p, 1.0);
    let two = batch(&parameters, p, 2.0);
    parameters.accumulate_gradients(one).unwrap();
    assert!(matches!(
        parameters.accumulate_gradients(two),
        Err(Error::WorkLimit {
            pass: Pass::Accumulation,
            ..
        })
    ));
    assert!(matches!(
        Sgd::new(0.5).unwrap().step(&mut parameters),
        Err(Error::WorkLimit {
            pass: Pass::Sgd,
            ..
        })
    ));
    assert_eq!(parameters.gradient(p).unwrap().unwrap().data(), [1.0]);
    assert_eq!(parameters.value(p).unwrap().data(), [2.0]);
}

#[test]
fn transfer_preflights_peak_without_double_charging_the_moved_adjoint() {
    let mut reference = set();
    let p = reference.insert(scalar(0.0)).unwrap();
    let baseline = reference.accounted_bytes();
    let mut tape = Tape::new(&reference).unwrap();
    let root = tape.parameter(p).unwrap();
    tape.backward(root).unwrap();
    let full_tape = reference.accounted_bytes();
    let batch = tape.into_parameter_gradients().unwrap();
    let batch_table = reference.accounted_bytes() - baseline - size_of::<f32>();
    drop(batch);
    let peak = full_tape + batch_table;
    for limit in [peak - 1, peak] {
        let mut parameters = ParameterSet::new(ResourceLimits::new(limit, 10, 10, 100)).unwrap();
        let p = parameters.insert(scalar(0.0)).unwrap();
        let mut tape = Tape::new(&parameters).unwrap();
        let root = tape.parameter(p).unwrap();
        tape.backward(root).unwrap();
        let batch = tape.into_parameter_gradients();
        if limit < peak {
            assert!(
                matches!(batch, Err(Error::MemoryLimit { requested, limit: observed })
                if requested == peak && observed == limit)
            );
            assert_eq!(parameters.accounted_bytes(), baseline);
        } else {
            let batch = batch.unwrap();
            assert_eq!(
                parameters.accounted_bytes(),
                baseline + batch_table + size_of::<f32>()
            );
            parameters.accumulate_gradients(batch).unwrap();
            assert_eq!(parameters.accounted_bytes(), baseline + size_of::<f32>());
        }
    }
}

#[test]
fn rejected_forward_growth_and_backward_memory_leave_accounting_unchanged() {
    let mut reference = set();
    let p = reference.insert(scalar(1.0)).unwrap();
    let persistent = reference.accounted_bytes();
    let mut tape = Tape::new(&reference).unwrap();
    let root = tape.parameter(p).unwrap();
    let one_leaf = reference.accounted_bytes();
    tape.backward(root).unwrap();
    let one_adjoint = reference.accounted_bytes();

    let mut parameters = ParameterSet::new(ResourceLimits::new(one_leaf, 10, 10, 100)).unwrap();
    let p = parameters.insert(scalar(1.0)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let root = tape.parameter(p).unwrap();
    assert!(matches!(
        tape.scale(root, 3.0),
        Err(Error::MemoryLimit { .. })
    ));
    assert_eq!(tape.value(root).unwrap().data(), [1.0]);
    assert_eq!(parameters.accounted_bytes(), one_leaf);
    drop(tape);
    assert_eq!(parameters.accounted_bytes(), persistent);

    let mut parameters =
        ParameterSet::new(ResourceLimits::new(one_adjoint - 1, 10, 10, 100)).unwrap();
    let p = parameters.insert(scalar(1.0)).unwrap();
    let mut tape = Tape::new(&parameters).unwrap();
    let root = tape.parameter(p).unwrap();
    assert!(matches!(
        tape.backward(root),
        Err(Error::MemoryLimit { .. })
    ));
    assert_eq!(parameters.accounted_bytes(), one_leaf);
    assert!(parameters.gradient(p).unwrap().is_none());
}

#[test]
fn aggregate_memory_charges_the_retained_spare_capacity_of_adopted_tensors() {
    let mut reference = set();
    reference.insert(scalar(1.0)).unwrap();
    let exact = reference.accounted_bytes();
    let mut parameters = ParameterSet::new(ResourceLimits::new(exact, 10, 10, 100)).unwrap();
    let mut data = Vec::with_capacity(2);
    data.push(1.0);
    let with_spare_capacity = Tensor::from_vec(&[], data, Default::default()).unwrap();
    assert!(matches!(
        parameters.insert(with_spare_capacity),
        Err(Error::MemoryLimit { .. })
    ));
    assert!(parameters.is_empty());
    assert_eq!(parameters.accounted_bytes(), 0);
    parameters.insert(scalar(1.0)).unwrap();
    assert_eq!(parameters.accounted_bytes(), exact);
}
