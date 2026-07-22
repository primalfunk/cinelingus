# Phase 3 implementation checkpoint — 2026-07-22

## State

Phase 3 engineering is complete through the provisional four-way rendered proof and one accepted render-verified function repair. Human calibration and subjective review remain explicitly deferred.

## Implemented

- Versioned three-axis dialogue-function taxonomy with explicit unknown, ambiguous, not-applicable, and unavailable states.
- Local deterministic English classifier with evidence, bounded context disclosure, confidence, ambiguity, and abstention.
- Separate provenance-preserving `dialogue_function_bundle_v2` cache for every SpeechPassage.
- DialogueTurn aggregation with ordered passage references, per-axis distributions, disagreement/ambiguity, confidence, and cache signatures.
- Ordered Performance function sequences preserving turn order and speaker structure without flattening.
- Four modes: `FUNCTION_DISABLED`, `FUNCTION_REPORT_ONLY`, `FUNCTION_ASSISTED`, and `FUNCTION_PRESERVING`.
- Confidence-aware function compatibility independent of semantic similarity and subordinate to existing legal/technical constraints.
- Four-way schedule screen plus zero-weight invariant: legacy, semantic-only, report-only, preserving, and internal zero-weight control.
- Rendered transcript reclassification that distinguishes intended preservation, donor function, rendered function, technical failure, and unverifiable evidence.
- High-confidence function-mismatch repair using already-legal donors, function-primary/semantic-secondary ranking, technical gates, uncertainty retention, render-gated commit, and candidate-level rollback.
- Human calibration package and finalizer with per-axis and per-label metrics, confusion records, confidence bins, abstention analysis, ambiguity, and annotator disagreement.
- Developer commands for taxonomy validation, bundle build/validation/report, calibration preparation/finalization, four-way schedule screening, and rendered-function verification.

## Real corpus findings

- Function bundles: 3,946 passages across eight Phase 2 FilmModels.
- Normalized DialogueTurns: four, all in the short-form excerpt; seven of eight models have zero normalized turns.
- Ordered function sequences: one available Performance sequence in the short-form excerpt. Sequence position remains explicitly unavailable and neutral elsewhere.
- Mega Man → short-excerpt provisional screen at semantic weight `0.05` and function weight `0.15`:
  - semantic/report-only mean function preservation: `0.500000`
  - function-preserving mean function preservation: `0.666667`
  - changed from report-only: one of four placements
  - technical regressions detected by schedule screen: zero
  - distinct donors retained: four
  - high-cosine/wrong-function candidates: three
  - lower-cosine/right-function candidate: one, with semantic delta `-0.087889` and function delta `+0.666667`
  - selection state: `BLOCKED_PENDING_REVIEWED_CALIBRATION`

The schedule result is candidate evidence only. It is not a rendered-quality or human-preference claim.

## Verification

- Full automated suite: **635 passed** in 23.01 seconds.
- Source media remained read-only.
- No new runtime dependency or external model was introduced.

## Required gate to finish Phase 3

1. Complete `evaluation/phase3_function_calibration_20260722/calibration_annotations.json` using the accompanying `calibration_review.md`.
2. Run `finalize-function-calibration`; inspect per-label errors and adjust/freeze thresholds only from reviewed evidence.
3. Re-run the schedule screen with calibration state `COMPLETE`.
4. Render the nominated legacy, semantic-only, report-only, and function-preserving variants under identical settings.
5. Run acoustic/residue, word-coverage, technical, and rendered-function verification.
6. Exercise at least one function-specific repair and require it to survive the second render; otherwise roll it back.
7. Prepare the blinded comparison and collect separate function, semantic, performance, completeness, and overall judgments.
8. Issue the required end-of-phase report without extending claims beyond measured evidence.
