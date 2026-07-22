# Phase 2 performance-corpus checkpoint — 2026-07-21

## Outcome

The performance-fill semantic screen now has a validated four-case corpus aggregate. It does not yet establish a quality improvement or justify enabling semantic assistance by default.

- Cases: 4
- Selected mappings: 18
- Selected mappings with semantic evidence: 17 (94.44%)
- Full-coverage cases: 3
- Partial-coverage cases: 1 (Wallace & Gromit, 3/4 mappings)
- Zero-influence invariant failures: 0
- Cases with a semantic selection change at weights 0.05–0.20: 0
- Render nominees: 0

The durable report is `evaluation/phase2_performance_corpus_screen_20260721/semantic_corpus_screen.json`.

## Reference reconciliation correction

Raw clip libraries can retain `e...` dialogue-event IDs while FilmModel speech passages point to filtered `w...` window IDs. Semantic admission now resolves records in this order:

1. direct source reference;
2. exact canonical media start;
3. conservative normalized transcript equality/containment (at least two tokens).

The fallback only links equivalent source records. It does not create candidates or alter similarity scores. This correction raised Magic School Bus coverage from 0/4 to 4/4 and Wallace & Gromit from 0/4 to 3/4. Wallace's remaining uncovered mapping retains the neutral legacy score and is reported as incomplete coverage.

## Interpretation

Mega Man, WKYK, and Magic School Bus are valid restraint cases: semantic evidence was available for every selected placement, all zero-influence invariants passed, and weights through 0.20 did not displace the performance-first selections.

Wallace & Gromit is not counted as a full restraint case because one selected mapping lacks admissible semantic evidence.

The earlier best-fit render proof remains rejected: its semantic variant reduced rendered word coverage and increased failed dialogue mappings. Human review was correctly withheld. The next render nominee must first pass selected-donor acoustic preflight; transcript metadata completeness alone is insufficient.

## Current gate

Semantic scheduling remains disabled by default. Advancing the claim requires either:

- a performance-fill case where semantics changes a legal, conflict-free selection and the selected donor passes acoustic preflight; or
- broader destination-performance coverage demonstrating where the tested weight range has useful influence without weakening performance, speaker, duration, visual, or completeness constraints.
