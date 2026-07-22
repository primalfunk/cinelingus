# Phase 2 — Render-informed semantic recovery checkpoint

Date: 2026-07-21

## Outcome

Phase 2 now has a deterministic recovery chain from semantic schedule selection through acoustic preflight, full render verification, and subsequent donor quarantine. Failed candidates no longer require one-off schedule edits and cannot silently re-enter later repair screens.

The Wallace & Gromit → Mega Man guarded nominee was rendered as a complete 23-minute control/semantic pair after one acoustic repair rescreen. The semantic render improved aggregate technical measurements but its two changed placements failed final rendered-dialogue verification, so human review was correctly withheld.

## Implemented

- Rejected acoustic preflight artifacts identify exact `source_performance_id` values.
- `screen-semantic-schedules --repair-preflight` quarantines those sources and selects the next direct-evidence, globally Pareto-safe alternatives.
- Two-cycle admissions are removed atomically when either donor is quarantined.
- Rejected full render proofs now expose changed-placement verification, failed intervention donors, and repair lineage.
- `screen-semantic-schedules --repair-render-proof` carries both the new rendered failures and all inherited quarantines into the next screen.
- Preflight artifacts preserve repair lineage across repeated recovery generations.
- Expressive ASR elongation is normalized narrowly: runs of three or more repeated characters align `Ah!` with `AAAAAAAH!`, while ordinary doubled letters are unchanged.
- Completed proof variants are reusable after interruption only when audio, video, final schedule, matching destination identity, residue evidence, and rendered-dialogue evidence are all present.
- Blinded case identity and context are now derived from the actual screen and destination rather than the earlier short-excerpt fixture.
- Pareto-guarded proof variants are reported as `SEMANTIC_ASSISTED` with their explicit selection policy.

## Full render result

Artifact: `evaluation/phase2_render_proof_wallace_to_mega_repair1_20260721/semantic_render_proof.json`

| Measure | Control | Semantic guarded |
| --- | ---: | ---: |
| Mapping count | 282 | 282 |
| Voice residue | NONE_DETECTED | NONE_DETECTED |
| Average rendered word coverage | 47.55% | 53.30% |
| Failed mappings | 153 | 138 |
| Initial editorial quality | 0.5287 | 0.5671 |
| Final editorial quality | 0.5665 | 0.5944 |
| Accepted repairs | 14 | 9 |
| Rejected placements | 212 | 201 |

The semantic path therefore reduced failures by 15 and raised average word coverage by 5.75 percentage points. This is useful system-level evidence, but it is not sufficient for semantic preference or Phase 2 completion.

## Intervention-specific result

The repaired schedule changed two placements:

- `p000115` / `c000158` — intended “Geronimo!”
- `p000163` / `c000249` — intended “Ah!”

Both passed selected-clip preflight at 100% word coverage. Both survived editorial repair, but both failed final in-context verification at their destination placements. The first was heard as “Confirmed”; the second produced no aligned speech. The proof was rejected for:

- `rendered_dialogue_verification_not_passing`
- `semantic_intervention_mapping_not_passing`

No blinded review package was released.

## Recovery trail

The rendered failures and prior acoustic failures produced a durable quarantine containing:

- `p000015`
- `p000115`
- `p000163`
- `p000196`

Subsequent candidates were rejected before rendering:

- `p000167` / “Ow!” — no speech detected
- `p000093` / “Oh” — no speech detected
- `p000131` / “Oh” — no speech detected

This establishes a recurring corpus defect among short-interjection candidates: transcript metadata exists while the exact scheduled WAV boundary contains no matching utterance.

## Current interpretation

The full render demonstrates that semantics can coexist with normal verification, rollback, and editorial repair and can improve aggregate technical measurements. It does not yet provide the required useful reviewed semantic effect because the actual changed placements failed in context.

The next corrective target should be opportunity-pool acoustic validation before guarded admission. Screening only the finally selected donor is safe but inefficient when a cluster of nearby short-interjection alternatives shares invalid clip boundaries. A bounded acoustic-health cache keyed by clip digest, trim range, Whisper configuration, intended transcript, and verifier version would allow the scheduler to exclude unsupported donor evidence before nomination without treating ASR as semantic understanding.

## Validation

- Full regression suite: **595 passed**.
- Source media remained read-only.
- No failed proof was released for human review.
