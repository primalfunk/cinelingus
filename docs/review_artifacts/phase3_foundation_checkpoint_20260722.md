# Phase 3 — Dialogue Function foundation checkpoint

Date: 2026-07-22

## State

Phase 3 is in progress. The taxonomy and provisional classification/scheduling
foundation are implemented. The next contract gate is human review of the calibration
set; classifier thresholds and taxonomy merges must not be treated as final before that
evidence exists.

## Implemented

- Versioned three-axis taxonomy with 39 stable labels: seven surface forms, 26
  interaction-function states, and six sequence-position states.
- Definitions, inclusion/exclusion rules, positive examples, counterexamples,
  multi-label policy, ambiguity/abstention rules, and migration policy.
- Local deterministic English rules classifier with qualified label IDs, inspectable
  rule evidence, per-label confidence, multi-label output, ambiguity, and abstention.
- SpeechPassage classification independent of DialogueTurn availability.
- Adjacent-passage and DialogueTurn context signatures; sequence position remains
  `unavailable` without ordered turn evidence.
- Separately cached, provenance-preserving function bundles with bounded interruption
  checkpoints, entity-level reuse, validation, and human-readable reports.
- Explicit English-only calibrated scope; non-English passages abstain rather than
  receiving fabricated function labels.
- Separate `dialogue_function_compatibility` contributor with per-axis distributions,
  confidence weakening, neutral unavailable sequence evidence, and distinct taxonomy/
  classifier identity.
- `FUNCTION_DISABLED`, `FUNCTION_REPORT_ONLY`, `FUNCTION_ASSISTED`, and
  `FUNCTION_PRESERVING` scheduling modes at the service layer.
- Shared scheduler integration that only reranks already-legal candidates. Report-only
  and zero-weight equivalence tests pass.
- Developer commands for taxonomy validation, bundle build/validation/reporting, and
  calibration preparation/finalization.
- Human calibration package that preserves classifier proposals as non-ground-truth,
  validates human labels, and records ambiguity and multi-annotator disagreement.

## Real-corpus evidence

All eight Phase 2 audit FilmModels were processed. The five principal scheduling films
account for 1,544 passages. The expanded eight-film calibration pool contains
animation, live action, short/long passages, non-lexical material, and the only four
passages with valid ordered-turn evidence.

The provisional 23-sample calibration package represents all 23 non-state interaction
labels requested for positive classification. Its first draft exposed over-broad
defense, narration, and command rules; those were narrowed and the classifier/cache
version advanced before issuing the current review package.

Artifacts:

- `evaluation/phase3_function_calibration_20260722/calibration_manifest.json`
- `evaluation/phase3_function_calibration_20260722/calibration_annotations.json`
- `evaluation/phase3_function_calibration_20260722/calibration_review.md`
- `evaluation/phase3_calibration_corpus_manifest_20260722.json`

## Gate before continuation

The contract requires a human-reviewed calibration set before classifier behavior is
finalized. The reviewer must assign surface form, interaction function(s), sequence
position, confidence, ambiguity, and notes for each sample. Proposals shown in the
worksheet are advisory and must not be accepted automatically.

After review, the next implementation sequence is:

1. finalize calibration statistics and revise/merge operationally weak labels;
2. freeze classifier thresholds/version;
3. run function counterfactual and four-way schedule screening;
4. implement rendered-function attribution and function-mismatch repair;
5. prove at least one surviving repair and prepare blinded five-question review;
6. close the 31 acceptance criteria and required builder report.

## Regression status

Full suite: **626 passed**. Source media remains read-only.
