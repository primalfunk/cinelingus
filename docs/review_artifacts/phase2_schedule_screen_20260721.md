# Phase 2 Semantic Schedule Screen — 2026-07-21

## Outcome

The first durable real-model schedule screen nominates `SEMANTIC_ASSISTED` at weight `0.05` for a bounded render proof. This is a render candidate, not a quality or preference result.

Case: Mega Man animation donor passages to the established 15-second short-form destination excerpt. The screen used pinned local `intfloat/multilingual-e5-small` bundles, exact cosine similarity, `best_fit` scheduling, lookahead 32, and soft shot-boundary scoring.

## Invariants

- Disabled and report-only selections are identical.
- Disabled and report-only scores are identical.
- Disabled and assisted-zero selections are identical.
- Disabled and assisted-zero scores are identical.
- Semantic scoring reranks only candidates already exposed by the scheduler.
- A new transcript-completeness gate prevents semantic assistance from selecting a candidate less complete than the exact legacy winner.

## Selected schedule effect

- Placements: 4
- Changed placements at weight 0.05: 2
- Mean selected raw cosine: `0.782622` control/report-only to `0.797048` assisted
- Mean transcript completeness: `1.0` to `1.0`
- Donor diversity: 4 unique donors; maximum reuse 1
- Measured performance, duration, speaker, visual, and completeness conflicts: 0
- Higher tested weights (`0.10`, `0.15`, `0.20`) produce the same selected schedule; therefore `0.05` is the conservative nominee.

The changed donors have legacy composite-score deltas of approximately `-0.0001` and `-0.0007`. These are disclosed soft-score tradeoffs. They require rendered verification and separate human judgments of semantic relatedness, performance fit, intelligibility/completeness, and overall preference.

## Negative comparison

The WKYK-to-excerpt case passes all zero-influence invariants but yields no schedule change after the completeness gate at weights through `0.20`. It is retained as restraint evidence rather than promoted to rendering.

## Evidence

Machine-readable schedules and comparison metrics are in `evaluation/phase2_schedule_screen_mega_man_to_excerpt_20260721/`.

No rendered-output, repair-survival, delivery, or human-preference claim is made by this screen.
