# Phase 2 two-cycle semantic swap audit — 2026-07-21

## Question

The counterfactual opportunity audit found one locally Pareto-safe substitution in Wallace & Gromit → Mega Man, but its proposed donor was already used at another destination. The bounded follow-up tested one two-placement cycle:

1. move donor performance `p000091` from destination `p000026` to `p000024`;
2. move the displaced donor `p000032` from `p000024` to `p000026`.

No larger rescheduling search was performed.

## Admission requirements

The second leg must independently pass:

- performance-class and speaker-count hard constraints;
- actual consecutive-turn placement validation;
- acceptance-floor similarity;
- complete semantic coverage;
- no scheduler-tier or soft-constraint regression;
- no regression in legacy performance score, performance compatibility, duration, speaker pattern, visual fit, or transcript completeness;
- positive semantic gain across the complete two-placement cycle.

## Result

The second leg was technically and editorially non-regressive:

- Replacement at `p000026`: `p000032` / clip `c000110`
- Legacy performance score delta: 0.0
- Scheduler tier delta: 0
- Soft-constraint count delta: 0
- Performance delta: 0.0
- Duration delta: 0.0
- Speaker delta: 0.0
- Visual delta: 0.0
- Transcript-completeness delta: 0.0

However, the combined semantic delta across both destinations was `-0.015903`. The locally positive first leg was outweighed by the semantic loss at the second destination.

State: `REJECTED`

Reason: `net_semantic_gain_not_positive`

## Conclusion

The only locally Pareto-safe opportunity in the measured cross-film matrix does not survive a two-cycle global allocation test. There are therefore still zero globally admissible semantic substitutions in this corpus evidence.

This is a useful stopping boundary for weighted performance-fill semantics: neither raising weight nor introducing a general rescheduler is supported. The next Phase 2 refinement should improve passage/turn representation or candidate coverage, then rerun the same fixed audit rather than expanding the allocation search.
