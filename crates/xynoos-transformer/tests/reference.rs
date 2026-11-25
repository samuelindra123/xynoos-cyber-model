#![forbid(unsafe_code)]

use serde_json::Value;
use xynoos_autograd::{AutogradError as A, Pass, ResourceLimits, Stage, Tape};
use xynoos_data::{BatchConfig, DatasetLimits, ManifestValidator, Split, StreamConfig, Tail};
use xynoos_tensor::{Tensor, TensorError};
use xynoos_tokenizer::{Boundaries, Tokenizer, TokenizerArtifact, TokenizerLimits};
use xynoos_transformer::{Config, Error, ReferenceBatch, Transformer};

const SEED: u64 = 0x58594e4f4f530005;
const LIMITS: TokenizerLimits = TokenizerLimits::MAX;
const REFERENCE: &str = include_str!("fixtures/reference.json");
const PARAMETERS: &str = include_str!("fixtures/parameters.json");
const TASK004: &str = include_str!("../../xynoos-data/tests/fixtures/task004.json");

type Fixture = (Value, Value, OwnedBatch);

fn fixture() -> Fixture {
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    let parameters = serde_json::from_str(PARAMETERS).unwrap();
    let batch = &reference["batch"];
    let owned = OwnedBatch {
        shape: serde_json::from_value(batch["shape"].clone()).unwrap(),
        input: serde_json::from_value(batch["input"].clone()).unwrap(),
        target: serde_json::from_value(batch["target"].clone()).unwrap(),
        valid: serde_json::from_value(batch["valid"].clone()).unwrap(),
        loss: serde_json::from_value(batch["loss"].clone()).unwrap(),
    };
    (reference, parameters, owned)
}

fn bpe() -> Tokenizer {
    let data: Value = serde_json::from_str(TASK004).unwrap();
    TokenizerArtifact::read(data["artifact"].as_str().unwrap().as_bytes(), LIMITS).unwrap()
}

struct OwnedBatch {
    shape: [usize; 2],
    input: Vec<u32>,
    target: Vec<u32>,
    valid: Vec<u8>,
    loss: Vec<u8>,
}

impl OwnedBatch {
    fn view(&self) -> ReferenceBatch<'_> {
        ReferenceBatch {
            shape: self.shape,
            input: &self.input,
            target: &self.target,
            valid: Some(&self.valid),
            loss: Some(&self.loss),
        }
    }

    fn row(&self, sample: usize, len: usize) -> ReferenceBatch<'_> {
        let start = sample * self.shape[1];
        let range = start..start + len;
        ReferenceBatch {
            shape: [1, len],
            input: &self.input[range.clone()],
            target: &self.target[range.clone()],
            valid: Some(&self.valid[range.clone()]),
            loss: Some(&self.loss[range]),
        }
    }
}

fn decode(text: &Value) -> f32 {
    let bytes = text.as_str().unwrap();
    assert_eq!(bytes.len(), 8);
    let raw = u32::from_str_radix(bytes, 16).unwrap();
    f32::from_bits(raw.swap_bytes())
}

fn values(list: &Value) -> Vec<f32> {
    list.as_array().unwrap().iter().map(decode).collect()
}

fn parameters(records: &Value, order: &Value) -> Vec<Vec<f32>> {
    order
        .as_array()
        .unwrap()
        .iter()
        .map(|name| values(&records["parameters"][name.as_str().unwrap()]["value"]))
        .collect()
}

fn fixture_coordinate(spec: xynoos_transformer::ParameterSpec, i: usize) -> f32 {
    let p = spec.index();
    if spec.shape().len() == 1 {
        1. + (((i + p) % 9) as f32 - 4.) / 32.
    } else {
        (((i * 67 + i / 11 * 19 + p * 43) % 257) as f32 - 128.) / if p == 0 { 2048. } else { 768. }
    }
}

fn explicit(config: Config, tokenizer: &Tokenizer) -> Transformer {
    Transformer::from_coordinates(config, tokenizer, Default::default(), fixture_coordinate)
        .unwrap()
}

#[derive(Default)]
struct Comparison {
    count: usize,
    abs: f64,
    rel: f64,
}

impl Comparison {
    fn compare(&mut self, a: f32, e: f32, abs: f64, rel: f64) {
        let (a, e) = (f64::from(a), f64::from(e));
        let error = (a - e).abs();
        assert!(error <= abs + rel * e.abs(), "{a} vs {e}, error {error}");
        self.abs = self.abs.max(error);
        if e != 0. {
            self.rel = self.rel.max(error / e.abs());
        }
        self.count += 1;
    }

    fn slice(&mut self, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected) {
            self.compare(a, e, 2e-5, 2e-5);
        }
    }

    fn report(&self, name: &str) {
        println!(
            "TASK005 {name}: n={} max_abs={:.9e} max_rel_nonzero={:.9e}",
            self.count, self.abs, self.rel
        );
    }
}

fn rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|s| {
        s.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

#[test]
fn mathematical_full_forward_backward_and_tying() {
    let (reference, records, batch) = fixture();
    let config = Config::MATHEMATICAL;
    let tokenizer = Tokenizer::byte();
    let supplied = parameters(&records, &reference["parameter_order"]);
    let mut model =
        Transformer::from_coordinates(config, &tokenizer, Default::default(), |spec, i| {
            supplied[spec.index()][i]
        })
        .unwrap();
    assert_eq!(model.parameter_count(), 7_536);
    let baseline = model.parameters().accounted_bytes();
    let revision = model.parameters().revision();
    for (spec, _) in model.inventory() {
        let name = spec.to_string();
        assert_eq!(reference["parameter_order"][spec.index()], name);
        assert_eq!(
            records["parameters"][name]["shape"],
            serde_json::json!(spec.shape())
        );
    }
    let run = model.evaluate(batch.view()).unwrap();
    assert_eq!(run.logits().shape(), &[2, 5, 260]);
    assert_eq!(
        run.metrics().unwrap().snapshot_work,
        2 * model.parameter_count()
    );
    let mut output = Comparison::default();
    output.slice(
        run.logits().data(),
        &values(&reference["outputs"]["logits"]),
    );
    output.slice(&[run.loss()], &values(&reference["outputs"]["loss"]));
    output.report("mathematical logits and loss");
    println!("TASK005 mathematical loss {:.9}", run.loss());
    let forward_bytes = model.parameters().accounted_bytes();
    let forward_peak = model.parameters().peak_accounted_bytes();
    let (gradients, metrics) = run.gradients().unwrap();
    let transfer_bytes = model.parameters().accounted_bytes();
    model.accumulate_gradients(gradients).unwrap();
    assert_eq!(model.parameters().revision(), revision);
    let mut gradients = Comparison::default();
    for (spec, id) in model.inventory() {
        let expected = values(&records["parameters"][spec.to_string()]["gradient"]);
        gradients.slice(
            model.parameters().gradient(id).unwrap().unwrap().data(),
            &expected,
        );
    }
    gradients.report("mathematical analytic gradient");
    let lookup = values(&reference["embedding_paths"]["lookup"]);
    let projection = values(&reference["embedding_paths"]["output"]);
    assert!(
        lookup[3 * config.width..4 * config.width]
            .iter()
            .any(|&x| x != 0.)
    );
    let expected: Vec<_> = lookup
        .iter()
        .zip(&projection)
        .map(|(&a, &b)| (f64::from(a) + f64::from(b)) as f32)
        .collect();
    let mut tied = Comparison::default();
    let embedding = model
        .parameters()
        .gradient(model.parameters().id_at(0).unwrap())
        .unwrap()
        .unwrap();
    tied.slice(embedding.data(), &expected);
    tied.report("embedding sum of lookup and output");
    println!(
        "TASK005 mathematical resources init_work={} base={baseline} forward_live={forward_bytes} forward_peak={forward_peak} transfer_live={transfer_bytes} retained={} peak={} rss_kib={:?} {metrics:?}",
        model.initialization_work(),
        model.parameters().accounted_bytes(),
        model.parameters().peak_accounted_bytes(),
        rss_kib()
    );
}

#[test]
fn seeded_initialization_bits() {
    let (reference, _, _) = fixture();
    let config = Config::MATHEMATICAL;
    let tok = Tokenizer::byte();
    let first = Transformer::new(config, &tok, SEED, Default::default()).unwrap();
    let second = Transformer::new(config, &tok, SEED, Default::default()).unwrap();
    assert_eq!(first.seed(), Some(SEED));
    let mut checked = 0;
    for ((spec, a), (_, b)) in first.inventory().zip(second.inventory()) {
        let observed = first.parameters().value(a).unwrap().data();
        let repeated = second.parameters().value(b).unwrap().data();
        let expected = values(&reference["initialization"][spec.to_string()]);
        for (i, &value) in observed.iter().enumerate() {
            assert_eq!(value.to_bits(), repeated[i].to_bits());
            assert_eq!(value.to_bits(), expected[i].to_bits(), "{spec}[{i}]");
            checked += 1;
        }
    }
    println!("TASK005 initialization matched {checked} independent binary32 values");
}

#[test]
fn finite_differences_at_both_steps() {
    let (reference, records, batch) = fixture();
    let tok = Tokenizer::byte();
    let config = Config::MATHEMATICAL;
    let base = parameters(&records, &reference["parameter_order"]);
    let create = |perturb: Option<(usize, usize, f64)>| {
        Transformer::from_coordinates(config, &tok, Default::default(), |spec, i| {
            let value = base[spec.index()][i];
            match perturb {
                Some((p, j, step)) if p == spec.index() && j == i => {
                    (f64::from(value) + step) as f32
                }
                _ => value,
            }
        })
        .unwrap()
    };
    let mut model = create(None);
    let (gradients, _) = model.evaluate(batch.view()).unwrap().gradients().unwrap();
    model.accumulate_gradients(gradients).unwrap();
    for step in [0.001, 0.003] {
        let mut rust = Comparison::default();
        let mut python = Comparison::default();
        for row in reference["finite_differences"].as_array().unwrap() {
            if row["step"].as_f64().unwrap() != step {
                continue;
            }
            let name = row["name"].as_str().unwrap();
            let (spec, id) = model
                .inventory()
                .find(|(entry, _)| entry.to_string() == name)
                .unwrap();
            let index = row["index"].as_u64().unwrap() as usize;
            let run = |delta| {
                create(Some((spec.index(), index, delta)))
                    .evaluate(batch.view())
                    .unwrap()
                    .loss()
            };
            let fd = ((f64::from(run(step)) - f64::from(run(-step))) / (2. * step)) as f32;
            let exact = model.parameters().gradient(id).unwrap().unwrap().data()[index];
            rust.compare(exact, fd, 2e-3, 2e-2);
            python.compare(exact, row["value"].as_f64().unwrap() as f32, 2e-3, 2e-2);
        }
        rust.report(&format!("native finite differences step={step}"));
        python.report(&format!("independent finite differences step={step}"));
        assert_eq!(rust.count, 23);
    }
}

#[test]
fn four_block_forward_and_gradient_coordinates() {
    let (reference, _, batch) = fixture();
    let mut model = explicit(Config::INTEGRATION, &bpe());
    assert_eq!(model.parameter_count(), 62_880);
    let expected = &reference["integration"];
    let evaluated = model.evaluate(batch.view()).unwrap();
    let mut output = Comparison::default();
    output.slice(evaluated.logits().data(), &values(&expected["logits"]));
    output.slice(&[evaluated.loss()], &values(&expected["loss"]));
    output.report("four-block logits and loss");
    let (grads, _) = evaluated.gradients().unwrap();
    model.accumulate_gradients(grads).unwrap();
    let mut gradients = Comparison::default();
    for (spec, id) in model.inventory() {
        let entry = &expected["gradients"][spec.to_string()];
        let coord = entry["indexes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize);
        let actual = model.parameters().gradient(id).unwrap().unwrap().data();
        for (i, value) in coord.zip(values(&entry["value"])) {
            gradients.compare(actual[i], value, 2e-5, 2e-5);
        }
    }
    assert_eq!(gradients.count, 76);
    gradients.report("four-block analytic gradient coordinates");
    let tok = bpe();
    for step in [0.001, 0.003] {
        let mut native = Comparison::default();
        let mut oracle = Comparison::default();
        for row in expected["finite_differences"].as_array().unwrap() {
            if row["step"].as_f64().unwrap() != step {
                continue;
            }
            let (spec, id) = model
                .inventory()
                .find(|(p, _)| p.to_string() == row["name"].as_str().unwrap())
                .unwrap();
            let coordinate = row["index"].as_u64().unwrap() as usize;
            let evaluate = |delta| {
                Transformer::from_coordinates(
                    Config::INTEGRATION,
                    &tok,
                    Default::default(),
                    |p, i| {
                        let value = fixture_coordinate(p, i);
                        if p.index() == spec.index() && i == coordinate {
                            (f64::from(value) + delta) as f32
                        } else {
                            value
                        }
                    },
                )
                .unwrap()
                .evaluate(batch.view())
                .unwrap()
                .loss()
            };
            let fd =
                ((f64::from(evaluate(step)) - f64::from(evaluate(-step))) / (2. * step)) as f32;
            let exact = model.parameters().gradient(id).unwrap().unwrap().data()[coordinate];
            native.compare(exact, fd, 2e-3, 2e-2);
            oracle.compare(exact, row["value"].as_f64().unwrap() as f32, 2e-3, 2e-2);
        }
        assert_eq!(native.count, 4);
        native.report(&format!("four-block native finite differences step={step}"));
        oracle.report(&format!(
            "four-block independent finite differences step={step}"
        ));
    }
}

#[test]
fn causal_padding_and_batch_invariance() {
    let (_, _, mut batch) = fixture();
    let config = Config::MATHEMATICAL;
    let model = explicit(config, &Tokenizer::byte());
    let baseline = model.parameters().accounted_bytes();
    let evaluated = model.evaluate(batch.view()).unwrap();
    let v = config.vocabulary;
    for (row, logits) in evaluated.logits().data().chunks_exact(v).enumerate() {
        if batch.valid[row] == 0 {
            assert!(logits.iter().all(|x| x.to_bits() == 0));
        }
    }
    let mut compare = Comparison::default();
    let mut both_losses = 0.;
    for (sample, valid) in [(0, 4), (1, 3)] {
        let long = model.evaluate(batch.row(sample, 5)).unwrap();
        let short = model.evaluate(batch.row(sample, valid)).unwrap();
        let start = sample * 5 * v;
        compare.slice(
            &evaluated.logits().data()[start..start + valid * v],
            short.logits().data(),
        );
        compare.slice(&long.logits().data()[..valid * v], short.logits().data());
        compare.slice(&[long.loss()], &[short.loss()]);
        both_losses += short.loss() / 2.;
    }
    compare.slice(&[evaluated.loss()], &[both_losses]);
    let old = evaluated.logits().data().to_vec();
    drop(evaluated);
    assert_eq!(model.parameters().accounted_bytes(), baseline);
    batch.input[2] = 15;
    batch.target[1] = 15;
    let changed = model.evaluate(batch.view()).unwrap();
    assert_eq!(&old[..2 * v], &changed.logits().data()[..2 * v]);
    assert_eq!(&old[5 * v..], &changed.logits().data()[5 * v..]);
    assert_ne!(&old[2 * v..3 * v], &changed.logits().data()[2 * v..3 * v]);
    compare.report("padding and batch invariance");

    let gradients = |len| {
        let mut model = explicit(config, &Tokenizer::byte());
        let (gradients, _) = model
            .evaluate(batch.row(0, len))
            .unwrap()
            .gradients()
            .unwrap();
        model.accumulate_gradients(gradients).unwrap();
        model
            .inventory()
            .flat_map(|(_, id)| {
                model
                    .parameters()
                    .gradient(id)
                    .unwrap()
                    .unwrap()
                    .data()
                    .to_vec()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(gradients(4), gradients(5));
}

struct DatasetFixture(std::path::PathBuf);

impl DatasetFixture {
    fn new() -> Self {
        use std::fs;
        let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("task005-batch-{}", std::process::id()));
        fs::create_dir_all(root.join("payload")).unwrap();
        let data: Value = serde_json::from_str(TASK004).unwrap();
        for (path, text) in data["payloads"].as_object().unwrap() {
            fs::write(root.join(path), text.as_str().unwrap()).unwrap();
        }
        for (file, field) in [
            ("manifest.json", "manifest"),
            ("sources.jsonl", "sources"),
            ("documents.jsonl", "documents"),
        ] {
            fs::write(root.join(file), data[field].as_str().unwrap()).unwrap();
        }
        Self(root)
    }
}

impl Drop for DatasetFixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn task004_stream_partial_and_fixed_final_batch() {
    let files = DatasetFixture::new();
    let dataset = ManifestValidator::open(&files.0, DatasetLimits::MAX).unwrap();
    let fit = dataset.fit_bpe(Split::Train, 32, LIMITS).unwrap();
    let mut model =
        Transformer::new(Config::INTEGRATION, fit.model(), SEED, Default::default()).unwrap();
    let baseline = model.parameters().accounted_bytes();
    let stream = |fixed| {
        dataset
            .fitted_batches(
                Split::Validation,
                &fit,
                StreamConfig {
                    batch: BatchConfig {
                        sequence_length: 8,
                        stride: 8,
                        batch_size: 3,
                        tail: Tail::Pad,
                        pad_final_batch: fixed,
                    },
                    boundaries: Boundaries::default(),
                    tokenizer_limits: LIMITS,
                },
            )
            .unwrap()
    };
    let batch = stream(true).next_batch().unwrap().unwrap();
    let run = model.evaluate_batch(&batch).unwrap();
    let forward_live = model.parameters().accounted_bytes();
    let forward_peak = model.parameters().peak_accounted_bytes();
    let (gradient, metrics) = run.gradients().unwrap();
    let transfer_live = model.parameters().accounted_bytes();
    model.accumulate_gradients(gradient).unwrap();
    assert!(
        model
            .inventory()
            .all(|(_, id)| model.parameters().gradient(id).unwrap().is_some())
    );
    println!(
        "TASK005 dataset resources shape={:?} init_work={} base={baseline} forward_live={forward_live} forward_peak={forward_peak} transfer_live={transfer_live} retained={} peak={} rss_kib={:?} {metrics:?}",
        batch.shape(),
        model.initialization_work(),
        model.parameters().accounted_bytes(),
        model.parameters().peak_accounted_bytes(),
        rss_kib()
    );
    let mut partial = stream(false);
    let mut fixed = stream(true);
    let v = model.config().vocabulary;
    let mut saw_partial = false;
    while let Some(batch) = partial.next_batch().unwrap() {
        let padded = fixed.next_batch().unwrap().unwrap();
        let short = model.evaluate_batch(&batch).unwrap();
        let long = model.evaluate_batch(&padded).unwrap();
        let [b, t] = batch.shape();
        assert_eq!(short.logits().shape(), &[b, t, v]);
        assert_eq!(long.logits().shape(), &[3, t, v]);
        assert_eq!(short.loss().to_bits(), long.loss().to_bits());
        assert_eq!(short.logits().data(), &long.logits().data()[..b * t * v]);
        assert!(
            long.logits().data()[b * t * v..]
                .iter()
                .all(|&x| x.to_bits() == 0)
        );
        saw_partial |= b < 3;
    }
    assert!(saw_partial);
    assert!(fixed.next_batch().unwrap().is_none());
}

fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
    Tensor::from_data(shape, data, Default::default()).unwrap()
}

#[test]
fn hand_attention_rope_rmsnorm_and_swiglu() {
    use xynoos_autograd::ParameterSet;
    let (reference, _, _) = fixture();
    let inputs = [1, 2, 258, 258];
    let targets = [2, 3, 258, 258];
    let mask = [1, 1, 0, 0];
    let batch = ReferenceBatch {
        shape: [2, 2],
        input: &inputs,
        target: &targets,
        valid: Some(&mask),
        loss: Some(&mask),
    };
    let mut set = ParameterSet::new(Default::default()).unwrap();
    let q = set
        .insert(tensor(&[2, 2, 2], &[1., 0., 0., 1., 2., 3., 4., 5.]))
        .unwrap();
    let gain = set.insert(tensor(&[2], &[2., 3.])).unwrap();
    let mut t = Tape::for_reference(&set, batch, 260, 1).unwrap();
    let q = t.parameter(q).unwrap();
    let gain = t.parameter(gain).unwrap();
    let v = t
        .input(&tensor(&[2, 2, 2], &[2., 0., 0., 4., 9., 9., 9., 9.]))
        .unwrap();
    let attention = t.causal_attention(q, q, v).unwrap();
    let mut comparison = Comparison::default();
    let mut expected = values(&reference["hand_attention"]);
    expected.extend([0.; 4]);
    comparison.slice(t.value(attention).unwrap().data(), &expected);
    let rope = t.rope(q).unwrap();
    let (sin, cos) = (1_f64.sin() as f32, 1_f64.cos() as f32);
    assert_eq!(&t.value(rope).unwrap().data()[..4], &[1., 0., -sin, cos]);
    let norm = t.rms_norm(q, gain).unwrap();
    let r = (0.5_f64 + 1e-5).sqrt();
    comparison.slice(
        t.value(norm).unwrap().data(),
        &[(2. / r) as f32, 0., 0., (3. / r) as f32, 0., 0., 0., 0.],
    );
    let glu = t.swiglu(q, v).unwrap();
    let silu_one = (1_f64 / (1. + (-1_f64).exp())) as f32;
    comparison.slice(
        &t.value(glu).unwrap().data()[..4],
        &[silu_one * 2., 0., 0., silu_one * 4.],
    );
    let root = t.sum(norm).unwrap();
    t.backward(root).unwrap();
    comparison.slice(
        t.gradient(gain).unwrap().unwrap().data(),
        &[(1. / r) as f32; 2],
    );
    comparison.report("hand attention, RoPE, RMSNorm, SwiGLU tables");
}

#[test]
fn stable_cross_entropy_and_exact_unsupervised_zero() {
    use xynoos_autograd::ParameterSet;
    let cases = [
        (
            vec![10_000., 9999., -10_000.],
            -10_000.,
            (-1_f64).exp().ln_1p() as f32,
        ),
        (vec![f32::MAX; 3], f32::MAX, 260_f32.ln()),
        (
            vec![f32::MAX, f32::from_bits(f32::MAX.to_bits() - 1), -f32::MAX],
            -f32::MAX,
            0.,
        ),
    ];
    for (values, fill, expected) in cases {
        let batch = ReferenceBatch {
            shape: [1, 2],
            input: &[1, 0],
            target: &[0, 258],
            valid: None,
            loss: Some(&[1, 0]),
        };
        let mut data = vec![fill; 520];
        data[..3].copy_from_slice(&values);
        let mut set = ParameterSet::new(Default::default()).unwrap();
        let id = set.insert(tensor(&[1, 2, 260], &data)).unwrap();
        let mut tape = Tape::for_reference(&set, batch, 260, 1).unwrap();
        let logits = tape.parameter(id).unwrap();
        let loss = tape.next_token_loss(logits).unwrap();
        let observed = tape.value(loss).unwrap().data()[0];
        Comparison::default().compare(observed, expected, 2e-5, 2e-5);
        tape.backward(loss).unwrap();
        let gradient = tape.gradient(logits).unwrap().unwrap().data();
        assert!(gradient[..260].iter().all(|x| x.is_finite()));
        assert!(gradient[260..].iter().all(|x| x.to_bits() == 0));
        assert!(
            gradient[..260]
                .iter()
                .map(|&x| f64::from(x))
                .sum::<f64>()
                .abs()
                < 1e-7
        );
        println!("TASK005 stable extreme cross entropy {:.9}", observed);
    }
}

fn autograd(error: A) -> Error {
    Error::Autograd(error)
}

#[test]
fn input_controls_validate_including_unsupervised_tokens() {
    let tok = Tokenizer::byte();
    let model = explicit(Config::MATHEMATICAL, &tok);
    let baseline = model.parameters().accounted_bytes();
    let batch = ReferenceBatch {
        shape: [1, 3],
        input: &[1, 2, 258],
        target: &[2, 258, 258],
        valid: Some(&[1, 1, 0]),
        loss: Some(&[1, 0, 0]),
    };
    assert!(model.evaluate(batch).is_ok());
    let cases = [
        (
            ReferenceBatch {
                input: &[1, 2, 260],
                ..batch
            },
            A::InvalidToken {
                index: 2,
                target: false,
            },
        ),
        (
            ReferenceBatch {
                target: &[2, 258, 260],
                ..batch
            },
            A::InvalidToken {
                index: 2,
                target: true,
            },
        ),
        (
            ReferenceBatch {
                input: &[259, 2, 258],
                ..batch
            },
            A::InvalidToken {
                index: 0,
                target: false,
            },
        ),
        (
            ReferenceBatch {
                target: &[2, 259, 258],
                ..batch
            },
            A::InvalidToken {
                index: 1,
                target: true,
            },
        ),
        (
            ReferenceBatch {
                input: &[258, 2, 258],
                ..batch
            },
            A::InvalidToken {
                index: 0,
                target: false,
            },
        ),
        (
            ReferenceBatch {
                valid: Some(&[1, 2, 0]),
                ..batch
            },
            A::InvalidMask { index: 1 },
        ),
        (
            ReferenceBatch {
                loss: Some(&[1, 2, 0]),
                ..batch
            },
            A::InvalidMask { index: 1 },
        ),
        (
            ReferenceBatch {
                valid: Some(&[1, 0, 1]),
                ..batch
            },
            A::InvalidMask { index: 1 },
        ),
        (
            ReferenceBatch {
                target: &[1, 258, 258],
                ..batch
            },
            A::InvalidTargetShift { index: 0 },
        ),
        (
            ReferenceBatch {
                loss: Some(&[1, 1, 0]),
                ..batch
            },
            A::InvalidMask { index: 1 },
        ),
        (
            ReferenceBatch {
                loss: Some(&[1, 0, 1]),
                ..batch
            },
            A::InvalidMask { index: 2 },
        ),
        (
            ReferenceBatch {
                loss: Some(&[0, 0, 0]),
                ..batch
            },
            A::NoSupervisedTargets,
        ),
        (
            ReferenceBatch {
                valid: Some(&[1, 0]),
                ..batch
            },
            A::InvalidReferenceShape,
        ),
    ];
    for (input, expected) in cases {
        let error = model.evaluate(input).unwrap_err();
        println!("TASK005 validation control {error}");
        assert_eq!(error, autograd(expected));
        assert_eq!(model.parameters().accounted_bytes(), baseline);
    }
    assert_eq!(
        model
            .evaluate(ReferenceBatch {
                shape: [0, 0],
                ..batch
            })
            .unwrap_err(),
        Error::InvalidBatchShape
    );
}

#[test]
fn invalid_dimensions_and_aggregate_inventory_rejected_before_generation() {
    let tok = Tokenizer::byte();
    let config = Config::MATHEMATICAL;
    assert_eq!(
        Transformer::new(Config::INTEGRATION, &tok, SEED, Default::default()).unwrap_err(),
        Error::VocabularyMismatch {
            model: 292,
            tokenizer: 260
        }
    );
    for invalid in [
        Config {
            width: 15,
            ..config
        },
        Config { heads: 0, ..config },
        Config {
            layers: 0,
            ..config
        },
        Config { ffn: 0, ..config },
        Config {
            context: 0,
            ..config
        },
        Config {
            vocabulary: 773,
            ..config
        },
    ] {
        assert_eq!(invalid.parameter_count(), Err(Error::InvalidConfiguration));
    }
    assert_eq!(
        Config {
            layers: usize::MAX,
            ..config
        }
        .parameter_count(),
        Err(autograd(A::ArithmeticOverflow))
    );
    assert!(matches!(
        Config {
            width: 8192,
            ..config
        }
        .parameter_count(),
        Err(Error::Autograd(A::Tensor(
            TensorError::PayloadLimitExceeded { .. }
        )))
    ));
    let limit = ResourceLimits::new(1024, 100, 100, ResourceLimits::MAX_WORK);
    let error = Transformer::from_coordinates(config, &tok, limit, |_, _| panic!("budget bypass"))
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Autograd(A::MemoryLimit { limit: 1024, .. })
    ));
    println!("TASK005 aggregate inventory control {error}");
    let model = explicit(
        Config {
            context: 1024,
            ..config
        },
        &tok,
    );
    let error = model
        .evaluate(ReferenceBatch {
            shape: [64, 1024],
            input: &[],
            target: &[],
            valid: None,
            loss: None,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Autograd(A::Tensor(TensorError::PayloadLimitExceeded { .. }))
    ));
    println!("TASK005 tensor limit control {error}");
}

#[test]
fn tight_limits_and_forward_nonfinite_cleanup() {
    use xynoos_autograd::ParameterSet;
    let batch = ReferenceBatch {
        shape: [1, 1],
        input: &[1],
        target: &[2],
        valid: None,
        loss: None,
    };
    let config = Config::MATHEMATICAL;
    let tok = Tokenizer::byte();
    for (nodes, ops, expected) in [
        (29, 19, A::ValueLimit { limit: 29 }),
        (30, 18, A::OperationLimit { limit: 18 }),
    ] {
        let limits = ResourceLimits::new(
            ResourceLimits::MAX_BYTES,
            nodes,
            ops,
            ResourceLimits::MAX_WORK,
        );
        let model = Transformer::new(config, &tok, SEED, limits).unwrap();
        let baseline = model.parameters().accounted_bytes();
        let error = model.evaluate(batch).unwrap_err();
        assert_eq!(error, autograd(expected));
        assert_eq!(model.parameters().accounted_bytes(), baseline);
        println!("TASK005 node or operation control {error}");
    }
    let limits = ResourceLimits::new(ResourceLimits::MAX_BYTES, 100, 100, 28);
    let mut set = ParameterSet::new(limits).unwrap();
    let id = set.insert(tensor(&[1, 1, 2], &[1., 1.])).unwrap();
    let before = set.accounted_bytes();
    let mut tape = Tape::for_reference(&set, batch, 260, 1).unwrap();
    let error = tape.parameter(id).unwrap_err();
    assert_eq!(
        error,
        A::WorkLimit {
            pass: Pass::Forward,
            requested: 29,
            limit: 28
        }
    );
    println!("TASK005 forward work control {error}");
    drop(tape);
    assert_eq!(set.accounted_bytes(), before);
    let model = Transformer::from_coordinates(config, &tok, Default::default(), |spec, _| {
        if spec.index() == 0 { 1. } else { f32::MAX }
    })
    .unwrap();
    let before = model.parameters().accounted_bytes();
    let revision = model.parameters().revision();
    let error = model.evaluate(batch).unwrap_err();
    assert!(matches!(
        error,
        Error::Autograd(A::NonFinite {
            stage: Stage::Forward,
            ..
        })
    ));
    assert_eq!(model.parameters().accounted_bytes(), before);
    assert_eq!(model.parameters().revision(), revision);
    assert!(
        model
            .inventory()
            .all(|(_, id)| model.parameters().gradient(id).unwrap().is_none())
    );
    println!("TASK005 finite-input overflow control {error}");
    let error = Transformer::from_coordinates(config, &tok, Default::default(), |_, _| f32::NAN)
        .unwrap_err();
    assert_eq!(
        error,
        autograd(A::NonFinite {
            stage: Stage::Parameter,
            index: 0
        })
    );
    println!("TASK005 NaN injection control {error}");
}

#[test]
#[ignore = "bounded hosted release candidate feasibility experiment"]
fn candidate_b1_t32() {
    let config = Config::RESEARCH_CANDIDATE;
    assert_eq!(config.parameter_count().unwrap(), 914_400);
    let model = Transformer::new(config, &bpe(), SEED, Default::default()).unwrap();
    let baseline = model.parameters().accounted_bytes();
    let input = std::array::from_fn::<_, 32, _>(|i| i as u32);
    let target = std::array::from_fn::<_, 32, _>(|i| i as u32 + 1);
    let run = model
        .evaluate(ReferenceBatch {
            shape: [1, 32],
            input: &input,
            target: &target,
            valid: None,
            loss: None,
        })
        .unwrap();
    assert_eq!(run.logits().shape(), &[1, 32, 292]);
    let metrics = run.metrics().unwrap();
    let forward_peak = model.parameters().peak_accounted_bytes();
    let forward_live = model.parameters().accounted_bytes();
    let loss = run.loss();
    let error = run.gradients().unwrap_err();
    assert!(matches!(
        error,
        Error::Autograd(A::WorkLimit {
            pass: Pass::Backward,
            limit: 100_000_000,
            ..
        })
    ));
    assert_eq!(model.parameters().accounted_bytes(), baseline);
    assert!(
        model
            .inventory()
            .all(|(_, id)| model.parameters().gradient(id).unwrap().is_none())
    );
    println!(
        "TASK005 candidate B=1 T=32 P={} loss={loss:.9} init_work={} base={baseline} forward_live={forward_live} forward_peak={forward_peak} peak={} rss_kib={:?} {metrics:?} backward={error}",
        model.parameter_count(),
        model.initialization_work(),
        model.parameters().peak_accounted_bytes(),
        rss_kib()
    );
}

#[test]
fn reverse_memory_and_work_fail_without_gradients() {
    let (_, _, batch) = fixture();
    let tok = Tokenizer::byte();
    let config = Config::MATHEMATICAL;
    let all = ResourceLimits::MAX_WORK;
    let reference = Transformer::new(config, &tok, SEED, Default::default()).unwrap();
    let evaluation = reference.evaluate(batch.view()).unwrap();
    let forward_peak = reference.parameters().peak_accounted_bytes();
    drop(evaluation);
    for limits in [
        ResourceLimits::new(forward_peak, 30, 19, all),
        ResourceLimits::new(ResourceLimits::MAX_BYTES, 30, 19, 300_000),
    ] {
        let model = Transformer::new(config, &tok, SEED, limits).unwrap();
        let baseline = model.parameters().accounted_bytes();
        let revision = model.parameters().revision();
        let failure = model
            .evaluate(batch.view())
            .unwrap()
            .gradients()
            .unwrap_err();
        assert_eq!(model.parameters().accounted_bytes(), baseline);
        assert_eq!(model.parameters().revision(), revision);
        assert!(
            model
                .inventory()
                .all(|(_, id)| model.parameters().gradient(id).unwrap().is_none())
        );
        if limits.max_work() == all {
            assert!(
                matches!(failure, Error::Autograd(A::MemoryLimit {limit, ..}) if limit == forward_peak)
            );
        } else {
            assert!(matches!(
                failure,
                Error::Autograd(A::WorkLimit {
                    pass: Pass::Backward,
                    requested: 547_964,
                    limit: 300_000
                })
            ));
        }
        println!("TASK005 reverse resource control {failure}");
    }
}

#[test]
fn operation_tables_match_independent_values_and_vjps() {
    let (reference, records, batch) = fixture();
    let operation_fixture: Value =
        serde_json::from_str(include_str!("fixtures/operations.json")).unwrap();
    let arrays = parameters(&records, &reference["parameter_order"]);
    let config = Config::MATHEMATICAL;
    let model =
        Transformer::from_coordinates(config, &Tokenizer::byte(), Default::default(), |spec, i| {
            arrays[spec.index()][i]
        })
        .unwrap();
    let mut tape = Tape::for_reference(
        model.parameters(),
        batch.view(),
        config.vocabulary,
        config.heads,
    )
    .unwrap();
    let snapshots: Vec<_> = model
        .inventory()
        .map(|(_, id)| tape.parameter(id).unwrap())
        .collect();
    let [e, ag, wq, wk, wv, wo, fg, wg, wu, wd, final_gain] = snapshots.try_into().unwrap();
    let x = tape.embedding(e).unwrap();
    let u = tape.rms_norm(x, ag).unwrap();
    let q = tape.project(u, wq, false).unwrap();
    let k = tape.project(u, wk, false).unwrap();
    let v = tape.project(u, wv, false).unwrap();
    let qr = tape.rope(q).unwrap();
    let kr = tape.rope(k).unwrap();
    let at = tape.causal_attention(qr, kr, v).unwrap();
    let projected = tape.project(at, wo, false).unwrap();
    let xa = tape.masked_residual(x, projected).unwrap();
    let r = tape.rms_norm(xa, fg).unwrap();
    let gate = tape.project(r, wg, false).unwrap();
    let value = tape.project(r, wu, false).unwrap();
    let activated = tape.swiglu(gate, value).unwrap();
    let down = tape.project(activated, wd, false).unwrap();
    let block = tape.masked_residual(xa, down).unwrap();
    let hidden = tape.rms_norm(block, final_gain).unwrap();
    let logits = tape.project(hidden, e, true).unwrap();
    let loss = tape.next_token_loss(logits).unwrap();
    tape.backward(loss).unwrap();
    let mut act = Comparison::default();
    let mut backward = Comparison::default();
    for (name, key) in [
        ("embedding", x),
        ("attention_norm", u),
        ("q", q),
        ("k", k),
        ("v", v),
        ("q_rotated", qr),
        ("k_rotated", kr),
        ("attended", at),
        ("projected", projected),
        ("attention_residual", xa),
        ("ffn_norm", r),
        ("gate", gate),
        ("value", value),
        ("activated", activated),
        ("down", down),
        ("block_output", block),
        ("final_hidden", hidden),
        ("logits", logits),
        ("loss", loss),
    ] {
        let expected = &operation_fixture["records"][name];
        let value = tape.value(key).unwrap();
        let grad = tape.gradient(key).unwrap().unwrap();
        assert_eq!(serde_json::json!(value.shape()), expected["shape"]);
        assert_eq!(value.shape(), grad.shape());
        act.slice(value.data(), &values(&expected["value"]));
        backward.slice(grad.data(), &values(&expected["gradient"]));
    }
    act.report("operation tables forward");
    backward.report("operation tables adjoints");
}
