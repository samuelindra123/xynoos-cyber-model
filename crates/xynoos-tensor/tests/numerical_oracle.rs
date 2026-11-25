#![forbid(unsafe_code)]

use xynoos_tensor::{Tensor, TensorLimits};

const LIMITS: TensorLimits = TensorLimits::new(1024);
const ATOL: f64 = 1e-6;
const RTOL: f64 = 1e-5;
const FIXTURE: &str = include_str!("fixtures/oracle.tsv");
const HEADER: &str = "# xynoos-oracle-v1 seed=0x58594e4f";
const CONTROL_RECORD: &str = "add\t-\t3f800000\t-\t40000000\t-\t40400000";

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
        f32::from_bits(u32::from_str_radix(x, 16).expect("invalid binary32 fixture value"))
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
    assert_eq!(
        actual.len(),
        expected.len(),
        "line {line}: {op} output count"
    );
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            agrees(actual, expected),
            "line {line}: {op} output {index}: expected {expected:?}, got {actual:?}"
        );
    }
}

fn verify_fixtures(fixture: &str) {
    let mut lines = fixture.lines();
    assert_eq!(lines.next(), Some(HEADER));
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
        assert_eq!(output.shape(), out_shape, "line {line}: {op} output shape");
        compare(output.data(), &expected, line, op);
    }
    assert!(cases > 0 && outputs > 0 && outputs <= 10_000);
}

#[test]
fn independent_binary32_fixtures() {
    verify_fixtures(FIXTURE);
}

fn verify_control_record(record: &str) {
    verify_fixtures(&format!("{HEADER}\n{record}\n"));
}

#[test]
fn accepts_hand_calculated_scalar_addition() {
    verify_control_record(CONTROL_RECORD);
}

#[test]
#[should_panic(expected = "line 2: invalid fixture record")]
fn rejects_record_missing_a_field() {
    let (record, _) = CONTROL_RECORD.rsplit_once('\t').unwrap();
    verify_control_record(record);
}

#[test]
#[should_panic(expected = "line 2: unknown operation unknown")]
fn rejects_unknown_operation() {
    verify_control_record(&CONTROL_RECORD.replacen("add\t", "unknown\t", 1));
}

#[test]
#[should_panic(expected = "invalid binary32 fixture value")]
fn rejects_invalid_binary32_encoding() {
    verify_control_record(&CONTROL_RECORD.replace("40400000", "not-hex"));
}

#[test]
#[should_panic(expected = "line 2: add output shape")]
fn rejects_wrong_output_shape() {
    verify_control_record(&CONTROL_RECORD.replace("\t-\t40400000", "\t1\t40400000"));
}

#[test]
#[should_panic(expected = "line 2: add output count")]
fn rejects_wrong_output_count() {
    verify_control_record(&CONTROL_RECORD.replace("40400000", "40400000,40400000"));
}

#[test]
#[should_panic(expected = "line 2: add output 0: expected 4.0, got 3.0")]
fn rejects_value_outside_tolerance() {
    verify_control_record(&CONTROL_RECORD.replace("40400000", "40800000"));
}
