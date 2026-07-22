# Phase 2 cross-film performance matrix — 2026-07-21

## Outcome

Four full-corpus performance-fill pairings were screened with semantic weights 0.00, 0.05, 0.10, 0.15, and 0.20:

- Mega Man → Magic School Bus
- WKYK → Wallace & Gromit
- Wallace & Gromit → Mega Man
- Magic School Bus → WKYK

Across the maximum-weight schedules:

- Scheduled mappings: 1,170
- Semantically covered mappings: 859 (73.42%)
- Full-coverage cases: 2
- Partial-coverage cases: 2
- Zero-influence invariant failures: 0
- Cases with semantic selection changes: 3
- Cases with changed-selection compatibility conflicts: 3
- Render nominees: 0

The durable aggregate is `evaluation/phase2_crossfilm_corpus_screen_20260721/semantic_corpus_screen.json`.

## Case findings

### Magic School Bus → WKYK

All 255 selected mappings had semantic evidence. No selections changed through weight 0.20. This is valid restraint evidence.

### Mega Man → Magic School Bus

All 256 selected mappings had semantic evidence. At weight 0.20, two placements changed. One change traded a semantic contribution increase of 0.034668 for lower performance compatibility (-0.0069), transcript completeness (-0.0367), and legacy candidate score (-0.0043). It was correctly withheld from rendering.

### WKYK → Wallace & Gromit

Only 207 of 376 selected mappings had semantic evidence (55.05%). At weight 0.20, two placements changed and one conflict reduced performance compatibility by 0.0081. This case is ineligible both for partial coverage and compatibility conflict.

### Wallace & Gromit → Mega Man

Only 141 of 283 selected mappings had semantic evidence at weight 0.20 (49.82%). All positive weights changed many placements; each variant produced dozens of conflicts affecting performance, duration, speaker, or transcript completeness. The strong availability asymmetry makes this evidence diagnostically useful but unsuitable for quality claims.

## Safety correction

Schedule-screen render nomination now requires all of the following:

1. assisted mode with positive semantic weight;
2. at least one changed placement;
3. zero compatibility conflicts;
4. semantic evidence for every selected placement.

Partial semantic coverage can therefore be measured, but it cannot nominate a render. This prevents availability bias from being mistaken for semantic improvement.

## Interpretation

The semantic subsystem is functioning and influential, but current performance-fill evidence does not show a safe quality gain. At low weights it is commonly restrained by performance-first ranking. Where it becomes influential, the observed changes either weaken established compatibility evidence or occur under incomplete semantic coverage.

The next refinement should be a counterfactual opportunity audit over already-legal candidates: identify placements where a more semantically related donor is Pareto-safe on performance, duration, speaker, visual fit, and completeness before allowing semantics to alter the schedule. Increasing semantic weight is not supported by this evidence.
