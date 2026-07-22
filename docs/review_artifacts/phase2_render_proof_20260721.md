# Phase 2 Semantic Render Proof — 2026-07-21

## Result

The first selected semantic schedule produced two technically complete 15.023-second videos, but the assisted variant is rejected before human review.

Both control and assisted renders used identical destination media, sample rate, channels, loudness target, fades, hard destination-dialogue suppression, background reconstruction, Whisper `medium` verification, and one permitted residue-correction pass. Both contained four mappings.

## Verification

| Measure | Control | Assisted 0.05 |
|---|---:|---:|
| Destination-voice residue | None detected | None detected |
| Residue correction passes | 0 | 0 |
| Rendered-dialogue status | Fail | Fail |
| Average word coverage | 72.22% | 51.34% |
| Failed mappings | 1 | 2 |

The semantic render therefore fails three admission conditions:

- rendered-dialogue verification does not pass;
- assisted failed-mapping count is worse than control;
- assisted word coverage is worse than control.

The automatically created blind package was moved to `blinded_review_WITHHELD_render_failure` and must not be presented as an eligible review case. The move is reversible; no render evidence was deleted.

## Interpretation

The schedule-only improvement in mean cosine did not survive rendered verification. This is useful negative evidence: transcript completeness estimated before rendering was insufficient to prevent actual word loss at the selected placements. Semantic gain cannot justify this degradation.

The case also used `best_fit`, so production editorial refinement did not run. It cannot satisfy the contract's repair-survival requirement. Follow-up performance-first screens for both Mega Man and WKYK preserve all zero-influence invariants and produce technically valid performance schedules, but neither changes selection at weights through 0.20. They are restraint evidence, not render candidates.

## Evidence

- `evaluation/phase2_render_proof_mega_man_to_excerpt_20260721/semantic_render_proof.json`
- `evaluation/phase2_performance_schedule_screen_mega_man_to_excerpt_20260721/semantic_schedule_screen.json`
- `evaluation/phase2_performance_schedule_screen_wkyk_to_excerpt_20260721/semantic_schedule_screen.json`

Phase 2 still requires a performance-first case where semantics changes an otherwise legal donor, survives render verification and repair, and is then admitted to blinded review.
