# Contract-driven media transformation pipeline

Cinelingus treats each run as a compiled media contract. The contract is the
single authority for input scope, output extent, audio behavior, repetition,
analysis capabilities, filter invariants, and final acceptance.

## Runtime sequence

1. **Describe media** using stream-level `ffprobe` facts rather than container
   duration alone.
2. **Compile a run contract** from the selected filter and every complete input
   file.
3. **Resolve one canonical media clock.** The anchor video is consumed from
   timestamp zero, and the output ends at the shortest required supporting
   audio stream. A film is never shortened for a preview or short-clip mode.
4. **Qualify the schedule before rendering.** Contract violations block the
   render. Exhausted candidate pools safely leave remaining windows unmodified
   unless the filter explicitly authorizes repetition.
5. **Render against the compiled extent.** Planning, audio construction, muxing,
   and acceptance use the same duration authority.
6. **Certify from evidence.** Schedule qualification, filter acceptance, and
   encoded stream/timing acceptance produce `CERTIFIED`, `DEGRADED`,
   `EXPERIMENTAL`, or `BLOCKED`; registry presence is not proof of reliability.

## Core artifacts

- `run_contract.json` records the immutable input and policy contract.
- `schedule_qualification.json` records pre-render checks and safe omissions.
- `filter_acceptance.json` records creative and audio-provenance checks.
- `montage_render_acceptance.json` records final stream and duration checks.
- `filter_certification.json` derives the operator-facing evidence state.

All artifacts are schema validated and written under
`output/contracts/<filter_id>/` where applicable.

## Analysis trust and cache compatibility

Speaker maps are classified as direct, inferred, weak, or failed. Filters that
require speaker identity accept only direct evidence. The speaker-cache
signature includes the backend, model, device, and trust-policy version, so a
weak fallback artifact cannot silently satisfy a later identity-dependent run.

## Extension boundary

Future filters, renderers, and analysis backends extend typed contract inputs
and evidence outputs. They should not introduce a second duration calculation,
implicit repetition, hidden fallback, or registry-only availability claim.
