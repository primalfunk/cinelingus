# Phase 2 semantic opportunity audit — 2026-07-21

## Purpose

The cross-film matrix showed that increasing semantic weight causes compatibility regressions before it yields a renderable improvement. The counterfactual opportunity audit therefore asks a narrower question without changing the schedule:

> Among candidates already admitted by performance-fill hard constraints, is there a more semantically related donor that remains no worse on every protected compatibility axis?

## Admission sequence

For each report-only destination performance, the audit:

1. starts from candidates that passed performance class, speaker-count, and acceptance-floor constraints;
2. considers only candidates with higher semantic contribution than the legacy winner;
3. runs the candidate's actual turn-placement plan;
4. requires semantic evidence for every resulting placement;
5. requires no lower scheduler tier or additional soft-constraint violations;
6. requires no regression in legacy performance score, performance compatibility, duration, speaker pattern, visual fit, or transcript completeness;
7. checks the completed schedule for source-performance reuse conflicts.

The audit is report-only. It does not influence candidate ranking or mutate the selected schedule.

## Cross-film findings

Across the four full-corpus pairings:

- Higher-semantic alternatives examined: 848
- Placement-valid alternatives: 586
- Fully semantically covered alternatives: 495
- Locally Pareto-safe alternatives: 1
- Globally admissible alternatives: 0

The single local opportunity occurred in Wallace & Gromit → Mega Man:

- Destination performance: `p000024`
- Counterfactual donor: `p000091` / clip `c000147`
- Semantic contribution improvement: +0.021844
- Protected-axis deltas: all 0.0

However, donor performance `p000091` is already selected at destination `p000026`. The opportunity therefore fails full-schedule source-reuse admission and cannot be promoted without a downstream reassignment or swap proof.

## Conclusion

There is currently no globally admissible, Pareto-safe semantic substitution in the measured matrix. This explains why semantics is either restrained or produces conflicts under direct weighted reranking.

The next useful experiment is a bounded two-placement swap audit: determine whether the legacy donor freed at `p000024` can legally and non-regressively replace `p000091` at `p000026`. Any larger rescheduling search should remain out of scope until a two-cycle demonstrates value.
