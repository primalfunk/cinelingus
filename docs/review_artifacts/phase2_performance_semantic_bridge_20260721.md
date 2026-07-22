# Phase 2 performance semantic bridge checkpoint — 2026-07-21

## Implemented

Speech-passage embeddings can now represent a whole source or destination performance when a clip/window has no direct semantic match. Explicit `speech_passage_references` are preferred; otherwise the bridge uses deterministic temporal overlap inside the FilmModel performance bounds. Direct clip/window evidence always wins.

Fallback provenance is retained as `performance_passage_aggregate`. Schedule-screen variants disclose the number of selected mappings that use it, and corpus reports now separate direct semantic coverage from aggregate-assisted coverage.

The counterfactual opportunity audit also records the selected-source, candidate-source, and destination evidence scopes. This prevents a broad performance representation from being mistaken for exact clip-level evidence.

## Fixed cross-film corpus result

The current four-pair aggregate contains 1,169 selected mappings:

- 1,097 have semantic representation (93.84%).
- 856 have direct placement-level representation (73.23%).
- 241 use the performance-passage aggregate fallback.
- Three cases have full aggregate-assisted coverage; Wallace & Gromit → Mega Man remains partial.
- Zero invariant failures and zero render nominees.

Per case:

| Pair | Mappings | Represented | Direct | Aggregate fallback | State |
|---|---:|---:|---:|---:|---|
| Mega Man → Magic School Bus | 256 | 256 | 256 | 0 | changed with conflicts |
| WKYK → Wallace & Gromit | 376 | 376 | 207 | 169 | changed with conflicts |
| Wallace & Gromit → Mega Man | 282 | 210 | 138 | 72 | partial coverage |
| Magic School Bus → WKYK | 255 | 255 | 255 | 0 | safe restraint |

## Counterfactual finding (superseded by placement-level audit)

The initial performance-level audit reported 14 globally admissible alternatives. Subsequent guarded-admission implementation showed that this statement was too broad: performance-level direct evidence does not guarantee that the actual selected clip has direct evidence. The corrected placement-level and mapping-score audit is recorded in `phase2_guarded_pareto_admission_20260721.md`.

## Interpretation and next gate

The bridge substantially reduces missing semantic evidence without disguising its granularity. It does not solve the remaining 72 unrepresented mappings and it does not establish quality improvement.

The guarded Pareto-admission scheduling path has now been implemented. It emits a variant only when the full placement-level gate succeeds.

Durable aggregate: `evaluation/phase2_crossfilm_corpus_screen_20260721/semantic_corpus_screen.json`
