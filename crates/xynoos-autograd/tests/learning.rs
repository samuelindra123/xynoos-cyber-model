use std::fmt::Write;

use xynoos_autograd::{ParameterId, ParameterSet, ResourceLimits, Sgd, Tape, ValueId};
use xynoos_tensor::Tensor;

const X: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
const Y: [f32; 5] = [-1.0, 1.0, 3.0, 5.0, 7.0];
const STEPS: usize = 100;
const RATE: f32 = 0.05;
const _: () = assert!(STEPS <= ResourceLimits::MAX_EXPERIMENT_STEPS);

fn scalar(value: f32) -> Tensor {
    Tensor::from_data(&[], &[value], Default::default()).unwrap()
}

fn objective(tape: &mut Tape<'_>, w: ParameterId, b: ParameterId) -> ValueId {
    let w = tape.parameter(w).unwrap();
    let b = tape.parameter(b).unwrap();
    // The samples are constants. Scaling implements w*x with no broadcast.
    let mut total = None;
    for (x, y) in X.into_iter().zip(Y) {
        let wx = tape.scale(w, x).unwrap();
        let prediction = tape.add(wx, b).unwrap();
        let target = tape.input(&scalar(y)).unwrap();
        let sample = tape.mse(prediction, target).unwrap();
        total = Some(total.map_or(sample, |sum| tape.add(sum, sample).unwrap()));
    }
    tape.scale(total.unwrap(), 1.0 / X.len() as f32).unwrap()
}

#[test]
fn native_full_batch_sgd_matches_101_independent_observations() {
    let mut parameters = ParameterSet::new(ResourceLimits::default()).unwrap();
    let w = parameters.insert(scalar(0.0)).unwrap();
    let b = parameters.insert(scalar(0.0)).unwrap();
    let initial_revision = parameters.revision();
    let sgd = Sgd::new(RATE).unwrap();
    let mut trace = String::new();
    let mut previous = f32::INFINITY;
    let mut initial = 0.0;
    let mut max_difference = [0.0_f64; 3];
    let mut count = 0;
    let mut final_state = [0.0; 3];
    for (step, row) in include_str!("fixtures/learning.txt").lines().enumerate() {
        let mut fields = row.split_ascii_whitespace();
        assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), step);
        let reference: Vec<f64> = fields.map(|field| field.parse().unwrap()).collect();
        assert_eq!(reference.len(), 3);
        let weight = parameters.value(w).unwrap().data()[0];
        let bias = parameters.value(b).unwrap().data()[0];
        let mut tape = Tape::new(&parameters).unwrap();
        let root = objective(&mut tape, w, b);
        let loss = tape.value(root).unwrap().data()[0];
        if step == 0 {
            assert_eq!((weight, bias, loss), (0.0, 0.0, 17.0));
            initial = loss;
        }
        assert!(loss.is_finite() && loss < previous, "loss at step {step}");
        previous = loss;

        let actual = [weight, bias, loss];
        for (i, (&a, &r)) in actual.iter().zip(&reference).enumerate() {
            let absolute = (f64::from(a) - r).abs();
            max_difference[i] = max_difference[i].max(absolute);
            let limit = if i < 2 {
                1e-6
            } else {
                // Bound the residual displacement from both parameter differences
                // and binary32 prediction rounding. Then bound the mean squares.
                let residual_delta = 3.25e-6;
                2.0 * r.sqrt() * residual_delta
                    + residual_delta * residual_delta
                    + 6.0 * f64::from(f32::EPSILON) * r
            };
            assert!(
                absolute <= limit,
                "step {step}, coordinate {i}: {actual:?} vs {reference:?}"
            );
        }
        write!(
            &mut trace,
            " {:08x}:{:08x}:{:08x}",
            weight.to_bits(),
            bias.to_bits(),
            loss.to_bits()
        )
        .unwrap();
        if matches!(step, 0 | 1 | 10 | 25 | 50 | 75 | 100) {
            println!("LEARNING step={step} w={weight:.9e} b={bias:.9e} loss={loss:.9e}");
        }
        count += 1;
        final_state = actual;
        if step == STEPS {
            break;
        }
        tape.backward(root).unwrap();
        let gradients = tape.into_parameter_gradients().unwrap();
        parameters.accumulate_gradients(gradients).unwrap();
        sgd.step(&mut parameters).unwrap();
    }
    assert_eq!(
        include_str!("fixtures/learning.txt").lines().count(),
        STEPS + 1
    );
    assert_eq!(count, STEPS + 1);
    assert_eq!(parameters.revision(), initial_revision + STEPS as u64);
    let [weight, bias, loss] = final_state;
    assert!(loss < 1e-8 && loss < initial * 1e-6);
    assert!((weight - 2.0).abs() <= 1e-5);
    assert!((bias - 3.0).abs() <= 1e-4);
    let [dw, db, dl] = max_difference;
    println!("LEARNING updates={STEPS} observations={count} rate={RATE} max_abs_ref_difference w={dw:.9e} b={db:.9e} loss={dl:.9e}");
    println!("TRAJECTORY{trace}");
}
