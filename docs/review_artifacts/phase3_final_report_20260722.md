# CINELINGUS PHASE 3 FINAL REPORT

**Phase:** Dialogue Function and Function-Preserving Translation  
**Date:** 2026-07-22  
**Closeout:** **ENGINEERING COMPLETE — HUMAN CALIBRATION AND SUBJECTIVE REVIEW DEFERRED**

## Executive finding

Phase 3 delivered an inspectable, local dialogue-function subsystem and a bounded Function-Preserving Translation path. In the selected Mega Man → short-excerpt proof, the first function donor failed acoustic preflight and was not rendered. A second already-legal donor was selected from a bounded audit, confirmed in isolation, passed preflight, survived the final render, reclassified as the intended function, eliminated the comparison variant's failed rendered placement, and raised average rendered word coverage from **68.65% to 87.30%**. No destination-voice residue was detected.

The result is technically useful but not human-calibrated. The classifier remains experimental and disabled by default. No claim is made that the taxonomy is validated by people or that the output is subjectively preferred.

Primary evidence:

- `evaluation/phase3_function_screen_mega_man_to_excerpt_provisional_20260722/function_schedule_screen.json`
- `evaluation/phase3_function_acoustic_preflight_mega_man_to_excerpt_provisional_20260722/function_acoustic_preflight.json`
- `evaluation/phase3_function_donor_audit_mega_man_to_excerpt_20260722/function_donor_audit.json`
- `evaluation/phase3_function_acoustic_preflight_repaired_mega_man_to_excerpt_20260722/function_acoustic_preflight.json`
- `evaluation/phase3_render_proof_mega_man_to_excerpt_provisional_20260722/function_render_proof.json`
- `evaluation/phase3_classifier_runtime_benchmark_20260722.json`

## 1. Implementation summary

### Major components

- Versioned three-axis taxonomy with definitions, examples, counterexamples, ambiguity rules, abstention rules, migration policy, and prohibited claim scope.
- Deterministic local English classifier with evidence-bearing rules, confidence, multi-label interaction output, ambiguity, and abstention.
- Separate `dialogue_function_bundle_v2` artifacts with entity-level resume, provenance, context signatures, taxonomy/classifier identity, and validation.
- DialogueTurn aggregation and ordered Performance function sequences that preserve turn order and speaker structure without flattening.
- Independent function compatibility scoring and four modes: disabled, report-only, assisted, and preserving.
- Four-way schedule screen and an internal zero-weight invariant control.
- Function-specific acoustic preflight, donor audit, rendered-transcript reclassification, repair selection, render-gated acceptance, and rollback support.
- Developer commands for taxonomy, bundles, calibration, screening, acoustic preflight, rendered verification, and proof rendering.
- Schema validation and automated tests for every new contract layer.
- Final regression suite: **635 passed in 23.01 seconds**.

### Dependency changes

No runtime dependency was added. The classifier uses Python rules only. Existing Whisper-medium, FFmpeg, semantic embeddings, scheduling, residue verification, and rendering machinery were reused.

Triton was unavailable during some Whisper word-alignment calls. Whisper used its slower fallback kernels; this affected speed, not proof acceptance.

### Deviations

- Human annotation was declined and is explicitly deferred.
- Human preference review was not conducted. A blinded A/B/C/D package was prepared but remains unopened.
- Seven of eight audited FilmModels contain no normalized DialogueTurns, so turn and sequence behavior remains a partial capability.
- The proof demonstrates one successful changed placement, not universal function correctness. Two earlier placements retain rendered function mismatches.

## 2. Final taxonomy

Canonical artifact: `src/cinelingus/dialogue_function/taxonomy_v1.json`.

### Axis A — surface form

| Label | Operational definition |
|---|---|
| `declarative` | Statement-like clause. |
| `interrogative` | Question form or information-seeking syntax. |
| `imperative` | Directive grammatical form. |
| `exclamatory` | Explicitly emphatic or exclamatory form. |
| `fragment` | Syntactically incomplete lexical utterance. |
| `non_lexical` | Vocalization without dependable lexical content. |
| `unknown` | Surface form cannot be supported. |

### Axis B — interaction function

| Label | Operational definition |
|---|---|
| `provide_information` | Supplies a proposition or fact. |
| `request_information` | Seeks information. |
| `request_action` | Politely or indirectly asks another to act. |
| `acknowledgment` | Registers receipt or recognition. |
| `agreement` | Expresses concurrence. |
| `disagreement` | Expresses non-concurrence. |
| `refusal` | Declines a request, action, or proposition. |
| `command` | Directly instructs another to act. |
| `warning` | Alerts another to danger or harmful consequence. |
| `accusation` | Attributes fault or wrongdoing. |
| `defense` | Denies or answers blame. |
| `explanation` | Supplies an explicit reason or causal account. |
| `reassurance` | Explicitly reduces stated concern. |
| `confession` | Admits responsibility or concealed conduct. |
| `threat` | States intended harm or an adverse contingent consequence. |
| `revelation` | Marks information as newly disclosed or previously hidden. |
| `interruption` | Explicitly breaks or stops another contribution. |
| `deflection` | Redirects or avoids the current demand/topic. |
| `narration` | Recounts events in temporal order. |
| `greeting_or_address` | Greets or directly addresses a participant. |
| `clarification_or_repair` | Corrects, restates, or requests conversational clarification. |
| `discourse_management` | Manages turn order, topic, or procedure. |
| `evaluation_or_reaction` | Evaluates or reacts to an observed proposition/event. |
| `unknown` | Function cannot be supported. |
| `ambiguous` | Multiple incompatible functions remain plausible. |
| `not_applicable` | No interaction function applies, including supported non-lexical material. |

### Axis C — sequence position

| Label | Operational definition |
|---|---|
| `initiating` | Begins a supported ordered exchange. |
| `responding` | Directly responds within supported ordered structure. |
| `continuing` | Continues an established contribution or sequence. |
| `closing` | Closes a supported exchange. |
| `standalone` | Structurally supported material outside an exchange. |
| `unavailable` | Ordered evidence is absent or insufficient. |

### Merges and removals

No requested label was removed or merged. The three axes remain separate because surface form, interaction function, and sequence position are not interchangeable. State labels remain explicit rather than being forced into substantive classes.

## 3. Annotation report

The prepared calibration set contains **23 real, provenance-linked passages** drawn from animation, live action, and the short-form excerpt. Proposal sampling covers all 23 substantive interaction labels with no proposal-coverage gap. Ten selected passages include `request_information`; most other target labels occur once, with `evaluation_or_reaction` and `greeting_or_address` occurring twice.

Human-reviewed samples: **0/23**. Therefore:

- agreement is not measured;
- human ambiguity rate is not measured;
- class balance describes classifier proposals only, not ground truth;
- per-label precision/recall and a human confusion matrix are unavailable;
- confidence calibration against people is unavailable.

The annotation finalizer is implemented and will preserve multiple annotators, ambiguity, disagreement, per-axis metrics, per-label precision/recall/F1, confidence bins, and abstention errors if review is resumed later.

## 4. Classifier report

### Selected approach

`dialogue_function_rules_v3_calibration_refinement`: ordered deterministic lexical, punctuation, and bounded-context rules. It is local, inspectable, dependency-free, reproducible, and able to abstain.

### Rejected approaches

- General-purpose LLM: prohibited by contract and unnecessary for the bounded baseline.
- Embedding-only classification: semantic similarity does not distinguish conversational function.
- Supervised neural classifier: rejected because no human-labeled Phase 3 training set exists.
- Broad scene/narrative inference: rejected because it would exceed transcript and declared structural evidence.
- Character, emotion, relationship, irony, or narrative-purpose classification: out of scope.

### Provenance

Each result records classifier version, taxonomy version/signature, configuration signature, source passage/provenance, transcript signature, context signature, evidence rules, confidence, ambiguity, and abstention.

### Runtime and memory

- 3,946 passages classified in **1.622 seconds**.
- Throughput: **2,432.16 passages/second**.
- Peak traced Python memory: **21.068 MB**.
- Four-way rendered proof: **157.772 seconds** total.
- Render-proof peak traced Python memory: **1,569.498 MB**; this excludes FFmpeg and native Whisper kernel peaks.

### Confidence and abstention

- Classified: 3,752/3,946.
- Abstained: 194/3,946 (**4.92%**).
- Animation: 168 abstentions among 1,067 passages (**15.75%**).
- Live action: 26 abstentions among 2,875 passages (**0.90%**).
- Short-form excerpt: 0 abstentions among 4 passages.

Low-confidence and ambiguous evidence is weakened toward neutral during scheduling. These thresholds are provisional because no human calibration was completed.

## 5. Classification quality

No human-ground-truth confusion matrix or per-label accuracy claim is available. The implemented evaluator can produce both once annotations exist.

Observed provisional findings:

- Generic declaratives frequently fall back to `provide_information`; this is useful but over-broad.
- Short or clipped utterances can shift among acknowledgment, evaluation/reaction, and provide-information.
- Request-action versus command remains lexically fragile.
- Transcript metadata can be displaced by one adjacent clip even when the audio itself is complete.
- Rendered words sometimes change the classification from the stored donor classification, demonstrating why schedule metadata is not proof.

Passage-only classification is available for all 3,946 passages. Context-aware adjacent-passage mode is implemented, but no human comparison was performed. Valid DialogueTurn context exists only for the four-passage short excerpt.

Animation produced substantially more abstentions, consistent with its higher frequency of short, exclamatory, non-lexical, or transcription-fragile material. This is a coverage observation, not a claim that animation classification is less accurate.

## 6. Turn and sequence findings

- Audited SpeechPassages: 3,946.
- Normalized DialogueTurns: 4.
- Models with DialogueTurns: 1/8.
- Models with no DialogueTurns: 7/8.
- Passage classifications with sequence position available: 4/3,946 (**0.10%**).
- Ordered Performance function sequences: 1.

Turn aggregates preserve ordered passage references, speaker references, per-axis distributions, confidence, and ambiguity. Performance sequences preserve the ordered series of turn distributions and speaker structure; they are never reduced to one label.

The capability remains partial. Missing order is represented as `sequence_position.unavailable` and contributes no penalty.

## 7. Function compatibility

Scoring policy: `dialogue_function_axis_compatibility_v1`.

- Surface form weight: 0.30.
- Interaction function weight: 0.60.
- Sequence position weight: 0.10 when supported; zero when unavailable.
- Low confidence and ambiguity weaken the contribution toward neutral.
- Function contribution is applied only after existing candidate legality and technical constraints.
- Proof configuration used semantic weight 0.05 and function weight 0.15.

Function evidence remains separate from semantic evidence. Semantic similarity is retained as a secondary contributor during function repair.

Counterexamples found in the real screen:

- Three high-cosine candidates had function compatibility 0.333: semantic similarity 0.818, 0.797, and 0.811.
- The initial lower-cosine/right-function alternative traded semantic score `-0.087889` for function score `+0.666667`, but failed acoustic preflight and was rejected.
- The accepted replacement `c000120` retained normalized semantic contribution 0.871 and function contribution 1.000, then survived rendering.

## 8. Function-Preserving Translation proof

| Variant | Schedule role | Changed from legacy | Mean scheduled function preservation | Result |
|---|---:|---:|---:|---|
| Legacy control | Function disabled | 0 | unavailable | Rendered control. |
| Semantic-only | Semantic weight 0.05 | 2 | unavailable | Rendered comparison. |
| Function report-only | Zero influence | 2 | 0.500 | Byte-identical to semantic-only. |
| Function-preserving screen | Function weight 0.15 | 3 | 0.667 | One changed placement; first donor rejected at preflight. |
| Function-preserving repaired | Confirmed legal replacement | 3 | bounded changed placement = 1.000 | Accepted after rendered verification. |

Report-only produced the same donor schedule, audio SHA-256, and video SHA-256 as semantic-only. This proves report-only and zero influence did not alter output.

The changed placement moved from a high-semantic/low-function donor to `c000120`, “But we humans learn from our mistakes.” The replacement was acoustically confirmed, classified as `provide_information`, and rendered into a destination passage classified as `provide_information`.

Full-render function reclassification remains mixed: the repaired variant contains **2 verified and 2 mismatched** placements. The bounded repair improved the targeted placement; it did not correct the two earlier semantic donor mismatches.

## 9. Technical quality

| Variant | Average word coverage | Failed mappings | Warning mappings | Residue |
|---|---:|---:|---:|---|
| Legacy | 69.95% | 1 | 3 | None detected |
| Semantic-only | 68.65% | 1 | 2 | None detected |
| Function report-only | 68.65% | 1 | 2 | None detected |
| Function-preserving repaired | **87.30%** | **0** | 3 | None detected |

The repaired fourth placement reached 85.71% rendered word coverage and retained the intended function. The overall proof passed technical non-regression. All four audio and video artifacts were produced.

Delivery result: **ACCEPTED AS PROVISIONAL TECHNICAL EVIDENCE**. It is not accepted as calibrated human-preference evidence.

## 10. Function repair

Attempts:

1. `c000073` — rejected before render. Metadata claimed “Fortunately, the building has been evacuated,” while Whisper observed an adjacent line at 16.67% metadata coverage and low confidence.
2. Bounded audit of 12 already-legal, function-compatible donors.
3. `c000120` — selected by function-first, semantics-second policy; observed transcript matched metadata at 100% in the audit and was confirmed again in isolation.

Results:

- Successful repair proposals: 1.
- Accepted after render: 1.
- Acoustic preflight failures: 1.
- Render rollbacks: 0.
- Unresolved non-target function mismatches: 2.
- Context-dependent/turn-sequence repairs: 0; structural coverage was insufficient and no sequence was fabricated.

## 11. Human review

Human review was explicitly deferred.

- Function judgment: not measured.
- Semantic judgment: not measured.
- Performance judgment: not measured.
- Intelligibility judgment: automated word coverage only; no human judgment.
- Overall preference: not measured.

A seeded blinded four-variant package exists at `evaluation/phase3_render_proof_mega_man_to_excerpt_provisional_20260722/deferred_blinded_review/`. Its questions remain separated into function fit, semantic fit, performance fit, completeness, and overall preference.

## 12. Known taxonomy and classifier weaknesses

- No human calibration or inter-annotator agreement.
- English-only rules; non-English passages are unavailable rather than guessed.
- Broad `provide_information` fallback can hide finer distinctions.
- Multi-function utterances are detected by independent rules rather than a learned joint model.
- Punctuation and ASR truncation can change surface/function labels.
- DialogueTurn and sequence coverage is extremely low.
- No active-speaker localization, mouth-motion attribution, or listener/reaction identity.
- Function fit cannot determine whether the visible person is the one speaking.
- Function preservation does not imply semantic, performance, visual, comedic, or narrative success.

## 13. Default enablement recommendation

**Recommendation: `FUNCTION_PRESERVING_AVAILABLE_BY_EXPLICIT_SELECTION`.**

- Normal/default production behavior: `FUNCTION_DISABLED`.
- Developer diagnostics: `FUNCTION_REPORT_ONLY` is safe and proven zero-influence.
- `FUNCTION_ASSISTED` and `FUNCTION_PRESERVING`: experimental, explicit opt-in only.
- Do not enable by default until human calibration and broader rendered proofs exist.

## 14. Phase 4 recommendation

The most useful next evidence is not broader language modeling; it is visual speaker-placement evidence:

1. Per-face mouth-activity tracks synchronized to destination speech windows.
2. Active-speaker attribution with explicit confidence and unavailable states.
3. Listener/reaction versus speaker-role evidence.
4. Action-conflict evidence for running, fighting, wide shots, and off-screen dialogue.
5. Occlusion, face visibility, shot scale, and camera-cut continuity.

Active-speaker evidence is necessary for resolving the observed cases where one voice spans two visible conversational participants or dialogue plays over visibly non-speaking action. It should remain a soft or report-only contributor until calibrated, with hard rejection reserved for high-confidence contradictions.

Likely visual-placement failures currently misread as function failures include visible mouth silence during inserted speech, speech continuing across speaker changes, dialogue over action-only shots, and visually apparent exchanges represented by a single donor voice.

Recommended Phase 4 scope: visual tracks and active-speaker attribution only, with no emotion, character identity, relationship, or narrative inference.

## Acceptance-criteria disposition

- Criteria 1–25 and 27–31: engineering implementation/evidence complete within the stated provisional claim scope.
- Criterion 20: satisfied by the `c000120` repair surviving acoustic and rendered verification.
- Criteria 24–25: real high-cosine/wrong-function and lower-cosine/right-function evidence recorded.
- Criterion 26 and all requested human-preference findings: **deferred**.
- Human confidence calibration implied by the work order: **deferred**.

Final disposition: **Phase 3 engineering closeout accepted; human-validation closeout remains open and must not be inferred from this report.**
