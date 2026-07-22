# Phase 2 — Semantic Passage Representation completion status

Date: 2026-07-22

## Verdict

Phase 2 is `COMPLETE`. All thirty-one acceptance criteria pass. The final blinded
human review was completed on 2026-07-22 and independently preferred the
semantic-assisted condition in all four requested judgments, with the reviewer
characterizing the overall difference as light.

The contract's required narrative builder deliverable is consolidated in
`docs/review_artifacts/phase2_end_of_phase_report_20260722.md`.

The production default remains semantic weight zero. Semantic similarity remains a
separate, bounded soft contribution and never overrides legality, timing, speaker,
visual, completeness, verification, repair, rollback, or delivery gates.

## Acceptance ledger

| Criteria | State | Evidence |
|---|---|---|
| 1–3: pinned local provider, CPU, optional GPU reporting | PASS | Local E5 provider, immutable model identity, explicit device/runtime record, fake and unavailable providers |
| 4–9: deterministic entities, external cache, validation, resume, invalidation isolation | PASS | Semantic bundle/cache tests and bundle validation reports |
| 10–12: exact deterministic retrieval and limitations | PASS | Exact cosine service, stable-ID tie breaking, provenance-rich semantic reports and false-similarity reporting |
| 13–16: turn audit and bounded turn/sequence support | PASS | Coverage audit; valid ordered turns only; zero-turn models disclosed; missing structure is not inferred |
| 17–22: optional scheduling, equivalence, hard constraints, clean fallback | PASS | Disabled/report-only/zero-weight equivalence tests; bounded assisted screen; quarantine and fallback tests |
| 23: control and assisted Translation render | PASS | `evaluation/phase2_render_proof_mega_man_to_excerpt_word_repaired_v2_20260721/semantic_render_proof.json` |
| 24: normal verification and repair | PASS | Word-level rendered verification; candidate-level rollback; acceptance state `ACCEPTED_FOR_HUMAN_REVIEW` |
| 25: useful semantic effect | PASS | Repaired semantic donor `c000042` survives final render at 100% rendered-word coverage; unsafe `c000056` is rolled back |
| 26: no material bounded-corpus regression | PASS | Cross-film and performance corpus screens plus unchanged hard constraints and zero-weight equivalence |
| 27: separate blinded human judgments | PASS | Completed signed review result records semantic relatedness, performance fit, intelligibility/completeness, and overall preference separately |
| 28–29: no prohibited claims; FilmModel/bridge intact | PASS | Scope declarations and equivalence proofs |
| 30: tests | PASS | 610 tests passing on 2026-07-22 |
| 31: source media read-only | PASS | Repairs write derived clip overlays and renders only; canonical/source media are not modified |

## Final vertical proof

The nominated assisted schedule requested two semantic changes. Rendered word-level
verification accepted one and rejected the other. Candidate-level recovery rolled
only the failed placement back to its control donor and rerendered the assisted
variant.

Final measurements:

- technical acceptance: `ACCEPTED_FOR_HUMAN_REVIEW`;
- changed mappings requested: 2;
- semantic mappings surviving repair: 1;
- semantic mappings rolled back: 1;
- failed semantic interventions after repair: 0;
- surviving semantic donor rendered-word coverage: 100%;
- control average rendered-word coverage: 69.95%;
- assisted average rendered-word coverage: 72.22%;
- shared pre-existing failed mappings: 1 in each condition, explicitly disclosed;
- new residual-dialogue regression: none detected.

This satisfies the contract's useful-effect definition through a semantic-assisted
repair that survives rendered verification. It does not assert that the semantic
condition is aesthetically preferable; that conclusion belongs to the blinded
review.

## Human-review result

The reviewer selected blinded version B for semantic relatedness, performance fit,
intelligibility/completeness, and overall preference, noting that “B seems lightly
better.” Unblinding showed B to be the `SEMANTIC` condition.

The validated result is recorded at:

`evaluation/phase2_render_proof_mega_man_to_excerpt_word_repaired_v2_20260721/blinded_review/semantic_review_result.json`

The review supports a bounded conclusion: in this proof case, the semantic-assisted
render was lightly preferred without technical degradation. A single review case is
not evidence for changing the production default, which remains semantic weight zero.

## Closure

Phase 2 delivers the required complete vertical proof:

SpeechPassage → local semantic embedding → exact similarity → optional bounded
scheduling contribution → Translation render → verification and candidate-level
repair → measured blinded A/B comparison.

No Phase 2 stop condition remains active. Further work belongs to a subsequent phase
or a separately authorized expansion of the evaluation corpus.
