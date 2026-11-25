use std::time::Instant;

use xynoos_autograd::GradientBatch;
use xynoos_transformer::CandidateReferenceExperimentPolicy as Policy;

use super::*;

const CONFIG: Config = Config::RESEARCH_CANDIDATE;

fn model(config: Config, limits: ResourceLimits) -> Transformer {
    Transformer::new(config, &bpe(), SEED, limits).unwrap()
}

fn candidate_fixture() -> Value {
    serde_json::from_str(include_str!("../fixtures/candidate.json")).unwrap()
}

fn batch(record: &Value) -> OwnedBatch {
    OwnedBatch {
        shape: serde_json::from_value(record["shape"].clone()).unwrap(),
        input: serde_json::from_value(record["input"].clone()).unwrap(),
        target: serde_json::from_value(record["target"].clone()).unwrap(),
        valid: serde_json::from_value(record["valid"].clone()).unwrap(),
        loss: serde_json::from_value(record["loss"].clone()).unwrap(),
    }
}

fn checksum(data: &[f32]) -> String {
    let mut checksum = 0xcbf29ce484222325_u64;
    for value in data {
        for byte in value.to_le_bytes() {
            checksum = (checksum ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    format!("{checksum:016x}")
}

fn summary(data: &[f32], expected: &Value, cmp: &mut Comparison) {
    let mut l1 = 0.;
    let mut l2_squared = 0.;
    let mut max_abs = 0_f64;
    for &value in data {
        assert!(value.is_finite());
        let value = f64::from(value);
        l1 += value.abs();
        l2_squared += value * value;
        max_abs = max_abs.max(value.abs());
    }
    for (name, actual) in [("l1", l1), ("l2_squared", l2_squared), ("max_abs", max_abs)] {
        let expected = expected[name].as_f64().unwrap();
        let error = (actual - expected).abs();
        assert!(
            error <= 1e-10 + 2e-5 * expected.abs(),
            "{name}: {actual} vs {expected}"
        );
        cmp.abs = cmp.abs.max(error);
        if expected != 0. {
            cmp.rel = cmp.rel.max(error / expected.abs());
        }
        cmp.count += 1;
    }
}

fn embedded(batch: &GradientBatch, model: &Transformer, expected: &Value, label: &str) {
    let id = model.parameters().id_at(0).unwrap();
    let data = batch.gradient(id).unwrap().unwrap().data();
    let indexes: Vec<usize> = serde_json::from_value(expected["indexes"].clone()).unwrap();
    let lookup = values(&expected["lookup"]);
    let output = values(&expected["output"]);
    let combined = values(&expected["combined"]);
    let mut cmp = Comparison::default();
    for (j, index) in indexes.into_iter().enumerate() {
        let sum = (f64::from(lookup[j]) + f64::from(output[j])) as f32;
        assert_eq!(sum, combined[j]);
        cmp.compare(data[index], sum, 2e-5, 2e-5);
    }
    cmp.report(label);
    let mut aggregates = Comparison::default();
    summary(data, &expected["summary"], &mut aggregates);
    aggregates.report(&format!("{label} aggregates"));
}

fn finite_differences(model: &Transformer, tok: &Tokenizer, fixture: &Value, batch: &OwnedBatch) {
    let params: Vec<_> = model
        .inventory()
        .map(|(_, id)| model.parameters().value(id).unwrap().data())
        .collect();
    let records = fixture["parameters"].as_array().unwrap();
    let mut covered = std::collections::BTreeSet::new();
    for step in [0.001_f64, 0.003] {
        let mut cmp = Comparison::default();
        for sample in fixture["finite_differences"].as_array().unwrap() {
            let name = sample["name"].as_str().unwrap();
            let entry = records
                .iter()
                .position(|record| record["name"] == name)
                .unwrap();
            let index = sample["index"].as_u64().unwrap() as usize;
            let check = &records[entry];
            let position = check["indexes"]
                .as_array()
                .unwrap()
                .iter()
                .position(|i| i.as_u64() == Some(index as u64))
                .unwrap();
            let expected = decode(&check["gradient"][position]);
            let id = model.parameters().id_at(entry).unwrap();
            let analytic = model.parameters().gradient(id).unwrap().unwrap().data()[index];
            let original = f64::from(params[entry][index]);
            let hi = (original + step) as f32;
            let lo = (original - step) as f32;
            assert!(f64::from(lo) < original && original < f64::from(hi));
            let loss = |at| {
                let perturbed =
                    Transformer::from_coordinates(CONFIG, tok, Default::default(), |spec, i| {
                        if spec.index() == entry && i == index {
                            at
                        } else {
                            params[spec.index()][i]
                        }
                    })
                    .unwrap();
                perturbed.evaluate(batch.view()).unwrap().loss()
            };
            let fd = (f64::from(loss(hi)) - f64::from(loss(lo))) / (f64::from(hi) - f64::from(lo));
            assert!(
                (f64::from(analytic) - fd).abs() <= 2e-3 + 2e-2 * fd.abs(),
                "{name} {index}: {analytic} vs {fd}"
            );
            let error = (f64::from(expected) - fd).abs();
            assert!(
                error <= 2e-3 + 2e-2 * fd.abs(),
                "{name} {index}: {expected} vs {fd}"
            );
            cmp.abs = cmp.abs.max(error);
            if fd != 0. {
                cmp.rel = cmp.rel.max(error / fd.abs());
            }
            cmp.count += 1;
            covered.insert(name);
        }
        assert_eq!(cmp.count, 12);
        cmp.report(&format!("candidate FD h={step}"));
    }
    println!("TASK005-R candidate FD tensors {covered:?}");
}

#[test]
#[ignore = "bounded hosted release candidate oracle, reverse and transfer"]
fn full_candidate_b1_t32() {
    let fixture = candidate_fixture();
    let tok = bpe();
    assert_eq!(tok.vocabulary_size(), 292);
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["architecture_version"], 1);
    assert_eq!(fixture["seed"], format!("0x{SEED:016x}"));
    for (name, value) in [
        ("v", CONFIG.vocabulary),
        ("d", CONFIG.width),
        ("n", CONFIG.layers),
        ("h", CONFIG.heads),
        ("f", CONFIG.ffn),
        ("context", CONFIG.context),
    ] {
        assert_eq!(fixture["config"][name], value);
    }
    let original = batch(&fixture["batch"]);
    assert_eq!(original.shape, [1, 32]);
    assert!(original.input.iter().copied().eq(0..32));
    assert!(original.target.iter().copied().eq(1..33));
    let start = Instant::now();
    let mut model = Transformer::new(CONFIG, &tok, SEED, Default::default()).unwrap();
    let initialization = start.elapsed();
    let base = model.parameters().accounted_bytes();
    let revision = model.parameters().revision();
    assert_eq!(base, 3_669_304);
    assert_eq!(model.initialization_work(), 20_118_280);
    assert_eq!(model.parameter_count(), 914_400);
    assert_eq!(model.inventory().len(), 74);

    let start = Instant::now();
    let eval = model.evaluate_candidate(original.view(), Policy).unwrap();
    let forward = start.elapsed();
    let forward_peak = model.parameters().peak_accounted_bytes();
    let forward_live = model.parameters().accounted_bytes();
    let loss = eval.loss();
    assert_eq!(eval.logits().shape(), &[1, 32, 292]);
    assert_eq!(loss.to_bits(), 5.736_882_f32.to_bits());
    assert_eq!(forward_live, 9_658_268);
    assert_eq!(forward_peak, 9_658_268);
    assert_eq!(eval.metrics().unwrap().forward_work, 66_894_756);
    let mut forward_cmp = Comparison::default();
    forward_cmp.slice(eval.logits().data(), &values(&fixture["logits"]));
    forward_cmp.compare(loss, decode(&fixture["loss"]), 2e-5, 2e-5);
    forward_cmp.report("candidate forward");

    let start = Instant::now();
    let reverse = eval.backward().unwrap();
    let backward = start.elapsed();
    let backward_peak = model.parameters().peak_accounted_bytes();
    assert!(backward_peak > forward_peak && backward_peak <= ResourceLimits::MAX_BYTES);
    let start = Instant::now();
    let (grads, metrics) = reverse.into_gradients().unwrap();
    let transfer = start.elapsed();
    let transfer_peak = model.parameters().peak_accounted_bytes();
    let transferred = model.parameters().accounted_bytes();
    assert_eq!(metrics.values, 198);
    assert_eq!(metrics.operations, 124);
    assert_eq!(metrics.backward_work, 152_581_015);
    assert!(metrics.backward_work <= Policy::BACKWARD_WORK);
    assert_eq!(metrics.gradient_transfer_work, 3_658_070);
    assert_eq!(grads.revision(), revision);

    let mut analytic = Comparison::default();
    let mut aggregates = Comparison::default();
    let mut fingerprints = Vec::new();
    let mut gradient_cells = 0;
    let records = fixture["parameters"].as_array().unwrap();
    assert_eq!(records.len(), model.inventory().len());
    for ((spec, id), record) in model.inventory().zip(records) {
        assert_eq!(spec.to_string(), record["name"].as_str().unwrap());
        let shape: Vec<usize> = serde_json::from_value(record["shape"].clone()).unwrap();
        assert_eq!(spec.shape(), shape);
        let value = model.parameters().value(id).unwrap();
        let weight_hash = checksum(value.data());
        assert_eq!(
            weight_hash,
            record["init_checksum"].as_str().unwrap(),
            "{spec}"
        );
        let grad = grads.gradient(id).unwrap().unwrap();
        assert!(model.parameters().gradient(id).unwrap().is_none());
        assert_eq!(grad.shape(), shape);
        gradient_cells += grad.numel();
        let wanted = values(&record["gradient"]);
        let indexes: Vec<usize> = serde_json::from_value(record["indexes"].clone()).unwrap();
        assert_eq!(indexes.len(), wanted.len());
        for (i, wanted) in indexes.into_iter().zip(wanted) {
            analytic.compare(grad.data()[i], wanted, 2e-5, 2e-5);
        }
        summary(grad.data(), &record["summary"], &mut aggregates);
        fingerprints.push((weight_hash, checksum(grad.data())));
    }
    assert_eq!(gradient_cells, 914_400);
    assert_eq!(analytic.count, 225);
    assert_eq!(aggregates.count, 222);
    analytic.report("candidate analytic");
    aggregates.report("candidate gradient aggregates");
    embedded(&grads, &model, &fixture["embedding"], "candidate shared E");

    let start = Instant::now();
    let accumulation_work = model.accumulate_gradients_with_work(grads).unwrap();
    let accumulation = start.elapsed();
    let accumulated = model.parameters().accounted_bytes();
    assert_eq!(accumulation_work, 914_548);
    assert_eq!(model.parameters().revision(), revision);
    for ((_, id), (weight, gradient)) in model.inventory().zip(&fingerprints) {
        assert_eq!(
            checksum(model.parameters().value(id).unwrap().data()),
            *weight
        );
        assert_eq!(
            checksum(model.parameters().gradient(id).unwrap().unwrap().data()),
            *gradient
        );
    }
    assert_eq!(
        accumulated - base,
        4 * gradient_cells + 131 * size_of::<usize>()
    );
    assert!(transferred > accumulated && transferred < forward_live);

    let repeated = batch(&fixture["repeated"]["batch"]);
    assert_eq!(repeated.input.iter().filter(|&&id| id == 3).count(), 2);
    let other = Transformer::new(CONFIG, &tok, SEED, Default::default()).unwrap();
    let other_base = other.parameters().accounted_bytes();
    let eval = other.evaluate_candidate(repeated.view(), Policy).unwrap();
    let mut cmp = Comparison::default();
    cmp.compare(
        eval.loss(),
        decode(&fixture["repeated"]["loss"]),
        2e-5,
        2e-5,
    );
    cmp.report("candidate repeated loss");
    let (foreign, _) = eval.gradients().unwrap();
    embedded(
        &foreign,
        &other,
        &fixture["repeated"]["embedding"],
        "candidate repeated shared E",
    );
    assert_eq!(
        model.accumulate_gradients(foreign),
        Err(autograd(A::WrongParameterSet))
    );
    let cleaned = other.parameters().accounted_bytes();
    assert_eq!(cleaned, other_base);
    assert_eq!(model.parameters().revision(), revision);
    assert_eq!(model.parameters().accounted_bytes(), accumulated);
    for ((_, id), (weight, gradient)) in model.inventory().zip(&fingerprints) {
        assert_eq!(
            checksum(model.parameters().value(id).unwrap().data()),
            *weight
        );
        assert_eq!(
            checksum(model.parameters().gradient(id).unwrap().unwrap().data()),
            *gradient
        );
    }
    println!(
        "TASK005-R candidate P={gradient_cells} entries=74 loss={loss:.9} revision={revision} finite_cells=914400 violations=0 nonfinite=0 foreign=WrongParameterSet base={base} forward_live={forward_live} forward_peak={forward_peak} backward_peak={backward_peak} transfer_peak={transfer_peak} transferred={transferred} accumulated={accumulated} cleaned={cleaned} VmHWM_KiB={:?} init_work={} backward_requested={} accumulation_work={accumulation_work} {metrics:?}",
        rss_kib(),
        model.initialization_work(),
        metrics.backward_work,
    );
    println!(
        "TASK005-R candidate elapsed_ms init={:.3} forward={:.3} backward={:.3} transfer={:.3} accumulation={:.3}",
        initialization.as_secs_f64() * 1000.,
        forward.as_secs_f64() * 1000.,
        backward.as_secs_f64() * 1000.,
        transfer.as_secs_f64() * 1000.,
        accumulation.as_secs_f64() * 1000.
    );
    let start = Instant::now();
    finite_differences(&model, &tok, &fixture, &original);
    println!(
        "TASK005-R candidate FD elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.
    );
}

#[test]
fn candidate_shape_and_configuration_are_bound() {
    let tiny = Tokenizer::byte();
    let wrong_configs = [
        Config {
            vocabulary: 260,
            ..CONFIG
        },
        Config {
            width: 88,
            ..CONFIG
        },
        Config {
            layers: 7,
            ..CONFIG
        },
        Config { heads: 8, ..CONFIG },
        Config { ffn: 240, ..CONFIG },
        Config {
            context: 127,
            ..CONFIG
        },
    ];
    let input = std::array::from_fn::<_, 32, _>(|i| i as u32);
    let target = std::array::from_fn::<_, 32, _>(|i| i as u32 + 1);
    let candidate_batch = ReferenceBatch {
        shape: [1, 32],
        input: &input,
        target: &target,
        valid: None,
        loss: None,
    };
    let bpe = bpe();
    for config in wrong_configs {
        let tok = if config.vocabulary == 260 {
            &tiny
        } else {
            &bpe
        };
        let model =
            Transformer::from_coordinates(config, tok, Default::default(), |_, _| 0.).unwrap();
        let baseline = model.parameters().accounted_bytes();
        let revision = model.parameters().revision();
        assert_eq!(
            model
                .evaluate_candidate(candidate_batch, Policy)
                .unwrap_err(),
            Error::InvalidCandidateConfiguration { actual: config }
        );
        assert_eq!(
            xynoos_autograd::ReferenceTransformerEvaluation::record_candidate(
                model.parameters(),
                candidate_batch,
                Policy,
            )
            .unwrap_err(),
            A::InvalidCandidateInventory,
        );
        assert_eq!(model.parameters().accounted_bytes(), baseline);
        assert_eq!(model.parameters().revision(), revision);
        assert!(
            model
                .inventory()
                .all(|(_, id)| model.parameters().gradient(id).unwrap().is_none())
        );
        println!("TASK005-R rejected candidate config {config:?}");
    }
    let model = Transformer::from_coordinates(CONFIG, &bpe, Default::default(), |_, _| 0.).unwrap();
    let baseline = model.parameters().accounted_bytes();
    for actual in [[2, 32], [1, 31]] {
        assert_eq!(
            model
                .evaluate_candidate(
                    ReferenceBatch {
                        shape: actual,
                        ..candidate_batch
                    },
                    Policy
                )
                .unwrap_err(),
            Error::InvalidCandidateBatchShape { actual }
        );
        assert_eq!(model.parameters().accounted_bytes(), baseline);
        println!("TASK005-R rejected candidate shape {actual:?}");
    }
}

#[test]
fn malformed_mask_and_target_shift_combinations() {
    let model = explicit(Config::MATHEMATICAL, &Tokenizer::byte());
    let baseline = model.parameters().accounted_bytes();
    let revision = model.parameters().revision();
    let good = ReferenceBatch {
        shape: [1, 4],
        input: &[1, 2, 3, 258],
        target: &[2, 3, 258, 258],
        valid: Some(&[1, 1, 1, 0]),
        loss: Some(&[1, 0, 0, 0]),
    };
    let accepted = model.evaluate(good).unwrap();
    assert!(accepted.loss().is_finite());
    drop(accepted);
    let cases = [
        (
            ReferenceBatch {
                target: &[2, 258, 258, 258],
                ..good
            },
            A::InvalidTargetShift { index: 1 },
        ),
        (
            ReferenceBatch {
                target: &[2, 2, 258, 258],
                ..good
            },
            A::InvalidTargetShift { index: 1 },
        ),
        (
            ReferenceBatch {
                target: &[2, 3, 258, 3],
                ..good
            },
            A::InvalidToken {
                index: 3,
                target: true,
            },
        ),
        (
            ReferenceBatch {
                input: &[1, 2, 3, 3],
                ..good
            },
            A::InvalidToken {
                index: 3,
                target: false,
            },
        ),
        (
            ReferenceBatch { loss: None, ..good },
            A::InvalidMask { index: 2 },
        ),
        (
            ReferenceBatch {
                valid: None,
                ..good
            },
            A::InvalidToken {
                index: 3,
                target: false,
            },
        ),
        (
            ReferenceBatch {
                loss: Some(&[1, 0, 0]),
                ..good
            },
            A::InvalidReferenceShape,
        ),
    ];
    for (bad, expected) in cases {
        assert_eq!(model.evaluate(bad).unwrap_err(), autograd(expected));
        assert_eq!(model.parameters().accounted_bytes(), baseline);
        assert_eq!(model.parameters().revision(), revision);
        assert!(
            model
                .inventory()
                .all(|(_, id)| model.parameters().gradient(id).unwrap().is_none())
        );
    }
    println!("TASK005-R seven malformed batch controls preserve state");
}

#[test]
#[ignore = "bounded hosted release check of tightened work limit and policy scope"]
fn candidate_policy_preserves_tighter_work_and_default_entry() {
    let work = ResourceLimits::MAX_WORK - 1;
    let limits = ResourceLimits::new(
        ResourceLimits::MAX_BYTES,
        ResourceLimits::MAX_VALUES,
        ResourceLimits::MAX_OPERATIONS,
        work,
    );
    let model = model(CONFIG, limits);
    let base = model.parameters().accounted_bytes();
    let fixture = candidate_fixture();
    let batch = batch(&fixture["batch"]);
    let eval = model.evaluate_candidate(batch.view(), Policy).unwrap();
    let peak = model.parameters().peak_accounted_bytes();
    assert_eq!(
        eval.gradients().unwrap_err(),
        autograd(A::WorkLimit {
            pass: Pass::Backward,
            requested: 152_581_015,
            limit: work,
        })
    );
    assert_eq!(model.parameters().accounted_bytes(), base);
    assert_eq!(model.parameters().peak_accounted_bytes(), peak);
}
