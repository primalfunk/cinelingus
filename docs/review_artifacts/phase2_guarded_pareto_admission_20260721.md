# Phase 2 guarded Pareto admission checkpoint — 2026-07-21

## Implemented capability

Semantic schedule screening now includes a fail-closed guarded admission path for performance-fill schedules.

An opportunity can influence a guarded schedule only when:

1. the existing hard constraints and placement validation pass;
2. protected performance, duration, speaker, visual, completeness, scheduler-tier, soft-constraint, and legacy scores do not regress;
3. full-schedule source reuse is directly admissible, or a positive two-cycle is independently admissible;
4. selected and candidate mappings have direct passage evidence at the actual placement level;
5. the semantic gain is positive;
6. candidate donors are reserved and displaced donors are quarantined across both performance selection and fallback filling.

The scheduler rebuilds admitted choices through the normal placement path and raises an error if an audited donor is no longer legal or placeable. It does not patch mappings after the fact.

## Defects found and closed during the real-corpus run

- A displaced donor could re-enter the later undercoverage fallback pool and cause unrelated destination changes. Quarantine now covers the entire build.
- Performance-level direct evidence could conceal a placement using `performance_passage_aggregate`. Admission now requires direct evidence on every affected mapping.
- The two-cycle audit compared performance similarity but omitted mapping-level `legacy_candidate_score`. That score is now a protected Pareto axis.

## Fixed-corpus result

After rerunning all four cross-film screens:

| Pair | Local Pareto-safe | Globally reusable | Guarded admissions |
|---|---:|---:|---:|
| Mega Man → Magic School Bus | 0 | 0 | 0 |
| WKYK → Wallace & Gromit | 0 | 0 | 0 |
| Wallace & Gromit → Mega Man | 11 | 8 | 0 |
| Magic School Bus → WKYK | 0 | 0 | 0 |

The strongest Wallace & Gromit → Mega Man proposal was a two-cycle between destination performances `p000026` and `p000031`. It was directly evidenced and had a positive net semantic delta, but the second leg reduced mapping-level legacy score by 0.0158. The corrected audit rejects it as `protected_axis_regression`.

The subsequent direct-passage bridge reduced this set further. The remaining positive globally reusable cycle still relies on performance-aggregate evidence for `c000094`; it is not eligible for guarded admission. See `phase2_direct_passage_bridge_20260721.md`.

## Decision

No `pareto_guarded` schedule is emitted for the fixed corpus, no render nominee is created, and acoustic preflight is intentionally not run. This is valid restraint evidence and confirms that the admission mechanism can distinguish a promising performance-level counterfactual from a fully authorized placement.

Durable aggregate: `evaluation/phase2_crossfilm_corpus_screen_20260721/semantic_corpus_screen.json`
