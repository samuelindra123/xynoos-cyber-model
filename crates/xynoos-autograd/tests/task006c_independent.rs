#![forbid(unsafe_code)]

use xynoos_autograd::{
    AdamW, AdamWConfig, ParameterSet, ResourceLimits, Tape, WeightedGradientAccumulator,
};
use xynoos_tensor::Tensor;

fn tensor(shape: &[usize], value: f32) -> Tensor {
    Tensor::from_data(
        shape,
        &vec![value; shape.iter().product()],
        Default::default(),
    )
    .unwrap()
}

fn gradient(
    parameters: &ParameterSet,
    ids: &[xynoos_autograd::ParameterId],
    values: &[f32],
) -> xynoos_autograd::GradientBatch {
    let mut tape = Tape::new(parameters).unwrap();
    let mut root = None;
    for (&id, &value) in ids.iter().zip(values) {
        let parameter = tape.parameter(id).unwrap();
        let scalar = tape.sum(parameter).unwrap();
        let term = tape.scale(scalar, value).unwrap();
        root = Some(match root {
            Some(previous) => tape.add(previous, term).unwrap(),
            None => term,
        });
    }
    tape.backward(root.unwrap()).unwrap();
    tape.into_parameter_gradients().unwrap()
}

fn reference_step(
    state: &mut [f32; 3],
    gradient: f32,
    rank_two: bool,
    config: [f32; 5],
    powers: [f64; 2],
) {
    // Deliberately spell out the frozen operations and each binary32 narrowing.
    let [beta1, beta2, epsilon, decay, rate] = config;
    let [beta1_power, beta2_power] = powers;
    let b1 = f64::from(beta1);
    let b2 = f64::from(beta2);
    let g = f64::from(gradient);
    let next_first = b1 * f64::from(state[1]) + (1.0 - b1) * g;
    let next_second = b2 * f64::from(state[2]) + (1.0 - b2) * (g * g);
    state[1] = next_first as f32;
    state[2] = next_second as f32;
    let corrected_first = f64::from(state[1]) / (1.0 - beta1_power);
    let corrected_second = f64::from(state[2]) / (1.0 - beta2_power);
    let decay_scale = if rank_two {
        1.0 - f64::from(rate) * f64::from(decay)
    } else {
        1.0
    };
    let decayed = decay_scale * f64::from(state[0]);
    let adaptive =
        f64::from(rate) * corrected_first / (corrected_second.sqrt() + f64::from(epsilon));
    state[0] = (decayed - adaptive) as f32;
}

#[test]
fn independent_f64_adamw_reference_checks_multiple_zero_and_extreme_updates() {
    let beta1 = f32::from_bits(0x3f7f_ffff);
    let beta2 = f32::from_bits(0x3f7f_ffff);
    let epsilon = 1e-30_f32;
    let decay = 0.2_f32;
    let rate = 0.01_f32;
    let config = AdamWConfig::new(beta1, beta2, epsilon, decay).unwrap();
    let mut parameters = ParameterSet::new(ResourceLimits::default()).unwrap();
    let matrix = parameters.insert(tensor(&[1, 1], 1.25)).unwrap();
    let gain = parameters.insert(tensor(&[1], -0.75)).unwrap();
    let mut optimizer = AdamW::new(&parameters, config).unwrap();
    let mut matrix_state = [1.25_f32, 0.0, 0.0];
    let mut gain_state = [-0.75_f32, 0.0, 0.0];
    let mut p1 = 1.0_f64;
    let mut p2 = 1.0_f64;

    for gradients in [[2.0_f32, -1.0_f32], [0.0, 0.0], [-0.5, 0.25]] {
        let batch = gradient(&parameters, &[matrix, gain], &gradients);
        parameters.accumulate_gradients(batch).unwrap();
        p1 *= f64::from(beta1);
        p2 *= f64::from(beta2);
        let config = [beta1, beta2, epsilon, decay, rate];
        let powers = [p1, p2];
        reference_step(&mut matrix_state, gradients[0], true, config, powers);
        reference_step(&mut gain_state, gradients[1], false, config, powers);
        optimizer.step(&mut parameters, rate).unwrap();

        assert_eq!(
            parameters.value(matrix).unwrap().data()[0].to_bits(),
            matrix_state[0].to_bits()
        );
        assert_eq!(
            parameters.value(gain).unwrap().data()[0].to_bits(),
            gain_state[0].to_bits(),
            "native={} reference={} gradient={gradients:?}",
            parameters.value(gain).unwrap().data()[0],
            gain_state[0]
        );
        assert_eq!(
            optimizer.first_moment(0).unwrap()[0].to_bits(),
            matrix_state[1].to_bits()
        );
        assert_eq!(
            optimizer.first_moment(1).unwrap()[0].to_bits(),
            gain_state[1].to_bits()
        );
        assert_eq!(
            optimizer.second_moment(0).unwrap()[0].to_bits(),
            matrix_state[2].to_bits()
        );
        assert_eq!(
            optimizer.second_moment(1).unwrap()[0].to_bits(),
            gain_state[2].to_bits()
        );
        assert_eq!(optimizer.beta1_power().to_bits(), p1.to_bits());
        assert_eq!(optimizer.beta2_power().to_bits(), p2.to_bits());
    }

    // A negative control: applying decoupled decay to the rank-1 gain disagrees.
    let incorrectly_decayed_gain =
        ((1.0 - f64::from(rate) * f64::from(decay)) * f64::from(-0.75_f32)) as f32;
    assert_ne!(incorrectly_decayed_gain.to_bits(), (-0.75_f32).to_bits());
    assert_ne!(
        parameters.value(gain).unwrap().data()[0].to_bits(),
        incorrectly_decayed_gain.to_bits()
    );
    assert_eq!(optimizer.step_count(), 3);
}

#[test]
fn independent_native_weighting_distinguishes_equal_microbatch_averaging() {
    let mut parameters = ParameterSet::new(ResourceLimits::default()).unwrap();
    let id = parameters.insert(tensor(&[1], 0.0)).unwrap();
    let mut accumulator = WeightedGradientAccumulator::new(&parameters, 2).unwrap();
    accumulator
        .add(&parameters, gradient(&parameters, &[id], &[1.0]), 2)
        .unwrap();
    accumulator
        .add(&parameters, gradient(&parameters, &[id], &[4.0]), 1)
        .unwrap();
    let (batch, metrics) = accumulator.finish(&parameters, None).unwrap();
    let actual = batch.gradient(id).unwrap().unwrap().data()[0];
    let independently_derived = ((2.0_f64 * 1.0 + 1.0_f64 * 4.0) / 3.0) as f32;
    let equal_microbatch_negative_control = ((1.0_f64 + 4.0) / 2.0) as f32;
    assert_eq!(actual.to_bits(), independently_derived.to_bits());
    assert_eq!(actual.to_bits(), 2.0_f32.to_bits());
    assert_eq!(metrics.supervised_tokens, 3);
    assert_eq!((actual - equal_microbatch_negative_control).abs(), 0.5);
}
