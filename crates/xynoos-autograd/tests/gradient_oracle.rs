use xynoos_autograd::{ParameterSet, Tape, ValueId};
use xynoos_tensor::Tensor;

struct Case {
    name: &'static str,
    groups: &'static [(&'static [usize], &'static [f32])],
    objective: fn(&mut Tape<'_>, &[ValueId]) -> ValueId,
}

fn tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor::from_data(shape, values, Default::default()).unwrap()
}

fn polynomial(tape: &mut Tape<'_>, x: &[ValueId]) -> ValueId {
    let x = x[0];
    let two = tape.input(&tensor(&[], &[2.0])).unwrap();
    let triple = tape.scale(x, 3.0).unwrap();
    let affine = tape.add(triple, two).unwrap();
    let square = tape.multiply(affine, affine).unwrap();
    let xx = tape.multiply(x, x).unwrap();
    let repeated = tape.add(xx, x).unwrap();
    tape.add(square, repeated).unwrap()
}

fn vector(tape: &mut Tape<'_>, x: &[ValueId]) -> ValueId {
    let target = tape.input(&tensor(&[3], &[0.25, -0.75, 1.5])).unwrap();
    let product = tape.multiply(x[0], x[1]).unwrap();
    let prediction = tape.add(product, x[0]).unwrap();
    let loss = tape.mse(prediction, target).unwrap();
    tape.scale(loss, -1.5).unwrap()
}

fn matrix(tape: &mut Tape<'_>, x: &[ValueId]) -> ValueId {
    let upstream = tape
        .input(&tensor(
            &[2, 4],
            &[0.5, -1.25, 2.0, 0.75, -1.5, 0.25, 0.5, -2.0],
        ))
        .unwrap();
    let product = tape.matmul(x[0], x[1]).unwrap();
    let weighted = tape.multiply(product, upstream).unwrap();
    tape.sum(weighted).unwrap()
}

const CASES: [Case; 3] = [
    Case {
        name: "polynomial",
        groups: &[(&[], &[-0.375])],
        objective: polynomial,
    },
    Case {
        name: "vector",
        groups: &[(&[3], &[0.75, -0.5, 1.25]), (&[3], &[-1.0, 0.5, 2.0])],
        objective: vector,
    },
    Case {
        name: "matrix",
        groups: &[
            (&[2, 3], &[0.5, -1.0, 2.0, -0.75, 1.5, 0.25]),
            (
                &[3, 4],
                &[
                    1.0, -2.0, 0.5, 0.25, 0.75, 0.5, -1.0, 2.0, -0.5, 1.25, 1.5, -0.75,
                ],
            ),
        ],
        objective: matrix,
    },
];

fn evaluate(case: &Case, values: &[f32], differentiate: bool) -> (f64, Vec<f64>) {
    let mut parameters = ParameterSet::new(Default::default()).unwrap();
    let mut start = 0;
    let ids: Vec<_> = case
        .groups
        .iter()
        .map(|(shape, group)| {
            let end = start + group.len();
            let id = parameters
                .insert(tensor(shape, &values[start..end]))
                .unwrap();
            start = end;
            id
        })
        .collect();
    let mut tape = Tape::new(&parameters).unwrap();
    let leaves: Vec<_> = ids.iter().map(|&id| tape.parameter(id).unwrap()).collect();
    let root = (case.objective)(&mut tape, &leaves);
    let objective = f64::from(tape.value(root).unwrap().data()[0]);
    let mut gradients = Vec::new();
    if differentiate {
        tape.backward(root).unwrap();
        for leaf in leaves {
            gradients.extend(
                tape.gradient(leaf)
                    .unwrap()
                    .unwrap()
                    .data()
                    .iter()
                    .map(|&x| f64::from(x)),
            );
        }
    }
    (objective, gradients)
}

fn central(case: &Case, original: &[f32], i: usize, step_scale: f64) -> f64 {
    let x = f64::from(original[i]);
    let h = step_scale * 1e-3 * x.abs().max(1.0);
    let (mut plus, mut minus) = (original.to_vec(), original.to_vec());
    plus[i] = (x + h) as f32;
    minus[i] = (x - h) as f32;
    let span = f64::from(plus[i]) - f64::from(minus[i]);
    assert!(span > 0.0);
    (evaluate(case, &plus, false).0 - evaluate(case, &minus, false).0) / span
}

fn fixture(name: &str) -> Vec<(f64, f64)> {
    include_str!("fixtures/gradients.txt")
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_ascii_whitespace();
            (fields.next().unwrap() == name).then(|| {
                let i = fields.next().unwrap().parse::<usize>().unwrap();
                let analytical = fields.next().unwrap().parse().unwrap();
                let numerical = fields.next().unwrap().parse().unwrap();
                assert!(fields.next().is_none());
                (i, (analytical, numerical))
            })
        })
        .enumerate()
        .map(|(index, (fixture_index, value))| {
            assert_eq!(index, fixture_index);
            value
        })
        .collect()
}

fn close(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= atol + rtol * actual.abs().max(expected.abs())
}

fn numerical_agrees(actual: f64, expected: f64) -> bool {
    close(actual, expected, 3e-3, 3e-2)
}

#[test]
fn python_analytic_and_numerical_references_match_native_gradients() {
    let mut coordinates = 0;
    let (mut analytical_error, mut python_fd_error, mut native_fd_error) =
        (0.0_f64, 0.0_f64, 0.0_f64);
    for case in &CASES {
        let values: Vec<_> = case
            .groups
            .iter()
            .flat_map(|(_, data)| data.iter().copied())
            .collect();
        let actual = evaluate(case, &values, true).1;
        let expected = fixture(case.name);
        assert_eq!(actual.len(), values.len());
        assert_eq!(actual.len(), expected.len());
        for (i, (&actual, &(analytical, numerical))) in actual.iter().zip(&expected).enumerate() {
            assert!(
                close(actual, analytical, 2e-6, 2e-5),
                "{}[{i}] analytical",
                case.name
            );
            assert!(
                numerical_agrees(actual, numerical),
                "{}[{i}] Python finite difference",
                case.name
            );
            analytical_error = analytical_error.max((actual - analytical).abs());
            python_fd_error = python_fd_error.max((actual - numerical).abs());
            for scale in [1.0, 0.5] {
                let native = central(case, &values, i, scale);
                assert!(
                    numerical_agrees(actual, native),
                    "{}[{i}] tape finite difference at {scale}",
                    case.name
                );
                native_fd_error = native_fd_error.max((actual - native).abs());
            }
            coordinates += 1;
        }
    }
    assert_eq!(coordinates, 25);
    println!("GRADIENTS {coordinates} coordinates, max absolute errors: analytical {analytical_error:.9e}, Python FD {python_fd_error:.9e}, native FD {native_fd_error:.9e}");
}

#[test]
fn checker_detects_all_three_negative_controls() {
    let case = &CASES[2];
    let values: Vec<_> = case
        .groups
        .iter()
        .flat_map(|(_, data)| data.iter().copied())
        .collect();
    let good = evaluate(case, &values, true).1;
    let references: Vec<_> = fixture(case.name)
        .into_iter()
        .map(|(_, numerical)| numerical)
        .collect();
    let rejects = |candidate: &[f64]| {
        candidate
            .iter()
            .zip(&references)
            .any(|(&actual, &expected)| !numerical_agrees(actual, expected))
    };
    assert!(!rejects(&good));
    let mut sign = good.clone();
    sign[0] = -sign[0];
    assert!(rejects(&sign), "sign-flipped gradient");
    let mut zero = good.clone();
    zero[0] = 0.0;
    assert!(rejects(&zero), "zeroed nonzero coordinate");
    let mut permuted = good.clone();
    for row in 0..2 {
        for col in 0..3 {
            permuted[col * 2 + row] = good[row * 3 + col];
        }
    }
    assert!(
        rejects(&permuted),
        "incorrectly permuted rectangular gradient"
    );
    println!(
        "GRADIENT CONTROLS detected 3/3: sign flip, zeroed coordinate, rectangular permutation"
    );
}
