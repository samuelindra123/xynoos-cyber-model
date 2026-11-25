#![forbid(unsafe_code)]

use xynoos_tensor::{Tensor, TensorLimits};

const LIMITS: TensorLimits = TensorLimits::new(1024);
const ATOL: f64 = 1e-6;
const RTOL: f64 = 1e-5;
const FIXTURE: &str = include_str!("fixtures/oracle.tsv");

fn parse_list<T>(field: &str, parse: impl Fn(&str) -> T) -> Vec<T> {
    if field == "-" {
        Vec::new()
    } else {
        field.split(',').map(parse).collect()
    }
}

fn shape(field: &str) -> Vec<usize> {
    parse_list(field, |x| x.parse().unwrap())
}

fn values(field: &str) -> Vec<f32> {
    parse_list(field, |x| {
        f32::from_bits(u32::from_str_radix(x, 16).unwrap())
    })
}

fn agrees(actual: f32, expected: f32) -> bool {
    if expected.is_nan() {
        return actual.is_nan();
    }
    if expected.is_infinite() {
        return actual == expected;
    }
    let (actual, expected) = (f64::from(actual), f64::from(expected));
    actual.is_finite() && (actual - expected).abs() <= ATOL + RTOL * expected.abs()
}

fn compare(actual: &[f32], expected: &[f32], line: usize, op: &str) {
    assert_eq!(actual.len(), expected.len(), "line {line}: {op}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            agrees(actual, expected),
            "line {line}: {op} output {index}: expected {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn independent_binary32_fixtures() {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("# xynoos-oracle-v1 seed=0x58594e4f"));
    let mut cases = 0;
    let mut outputs = 0;
    for (line, record) in (2..).zip(lines) {
        let fields: Vec<_> = record.split('\t').collect();
        let &[op, a_shape, a, b_shape, b, out_shape, expected] = fields.as_slice() else {
            panic!("line {line}: invalid fixture record");
        };
        let a = Tensor::from_vec(&shape(a_shape), values(a), LIMITS).unwrap();
        let out_shape = shape(out_shape);
        let expected = values(expected);
        cases += 1;
        outputs += expected.len();

        let output = match op {
            "sum" => {
                assert!(out_shape.is_empty());
                compare(&[a.sum()], &expected, line, op);
                continue;
            }
            "dot" => {
                let b = Tensor::from_vec(&shape(b_shape), values(b), LIMITS).unwrap();
                assert!(out_shape.is_empty());
                compare(&[a.dot(&b).unwrap()], &expected, line, op);
                continue;
            }
            "transpose" => a.transpose(LIMITS),
            "scale" => {
                let &[scalar] = values(b).as_slice() else {
                    panic!("line {line}: missing scale factor");
                };
                a.scale(scalar, LIMITS)
            }
            "add" | "subtract" | "multiply" | "matmul" => {
                let b = Tensor::from_vec(&shape(b_shape), values(b), LIMITS).unwrap();
                match op {
                    "add" => a.add(&b, LIMITS),
                    "subtract" => a.subtract(&b, LIMITS),
                    "multiply" => a.multiply(&b, LIMITS),
                    "matmul" => a.matmul(&b, LIMITS),
                    _ => unreachable!(),
                }
            }
            _ => panic!("line {line}: unknown operation {op}"),
        }
        .unwrap_or_else(|error| panic!("line {line}: {op}: {error}"));
        assert_eq!(output.shape(), out_shape, "line {line}: {op}");
        compare(output.data(), &expected, line, op);
    }
    assert!(cases > 0 && outputs > 0 && outputs <= 10_000);
}
