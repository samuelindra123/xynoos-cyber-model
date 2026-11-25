# XYNOOS Cyber Model Versioning

**Status:** owner-approved naming and versioning policy for the initial development period.

## Identity

- Project name: **XYNOOS Cyber-Model**.
- Public model family: **XYNOOS Cyber**.
- First experimental model candidate: **XYNOOS Cyber 0.1**.
- Canonical model ID: `xynoos-cyber-0.1`.

XYNOOS Cyber 0.1 is an experimental architecture candidate awaiting language-model training. It is not a trained, intelligent, production-ready, or publicly released model. Randomly initialized weights are not a completed model release and must not be published as one.

Use numeric-only model names during the initial development period: XYNOOS Cyber 0.1, 0.2, 0.3, 1.0, 1.1, 2.0, and so on. Do not add descriptive model-family suffixes such as Flux, Veril, Quantum, or Genesis.

## Version semantics

- **0.x** identifies experimental research generations. It does not imply a public release or demonstrated language-model capability.
- **1.0** requires explicit owner approval of model release qualification and completion of every required release gate.
- **Minor versions** represent meaningful, documented improvements within a model generation. A parameter-count change alone is insufficient.
- **Major versions** represent significant architecture changes or capability changes established by evaluation evidence.
- **Artifact revisions** are tracked separately from capability versions. Repacking, metadata fixes, or byte-for-byte replacements do not automatically create a new model version; each artifact still needs its own checksum and provenance.
- **Engine versions** are independent of model versions. An engine optimization or compatible runtime change does not automatically create a new model version.
- **Checkpoint identity** is independent of marketing version. Identify a checkpoint by its immutable content digest and the training run, source, data, and configuration that produced it.
- **Quantization** does not automatically create a new model generation. Record quantization method, precision, calibration or conversion provenance, and artifact checksum as artifact metadata.

## Release evidence

Every model release requires documented:

1. Architecture and actual parameter count.
2. Tokenizer identity and vocabulary/configuration.
3. Data and artifact provenance, rights, and applicable licensing.
4. Training source revision, configuration, seed policy, and training-token accounting.
5. Evaluation methods, results, limitations, and safety evidence appropriate to the release.
6. Artifact format, checksums, and a separate licensing review.
7. Runtime compatibility and measured resource claims, if such claims are made.

No model may be presented as a completed release until the owner approves qualification and the required evidence is available. Public naming does not establish capability.

## Research scale possibilities

Future architecture research may consider approximately 5M, 10M, 50M, 125M, 350M, 1B, 3B, 8B, or 14B and higher parameter targets. These are research possibilities, not delivery commitments or automatic version mappings. Do not assign a future model version solely from parameter count. Scaling decisions depend on measured training quality, efficiency, data coverage, resource availability, and explicit owner approval.
