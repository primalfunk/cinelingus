# Phase 2 selected-donor acoustic preflight — 2026-07-21

## Outcome

Changed semantic donor clips must now pass targeted Whisper-medium retranscription before `render-semantic-proof` will run. This closes the gap between transcript metadata completeness and the words acoustically present inside a selected WAV boundary.

The preflight:

1. compares the control and nominated assisted schedules by destination placement;
2. selects only donor mappings changed by semantics;
3. assembles their exact trimmed WAV regions into one silence-separated proof reel;
4. transcribes that reel once using the configured Whisper model (medium by default);
5. checks ordered word coverage, sentence beginning, sentence ending, and observed speech;
6. emits `ACCEPTED_FOR_RENDER` or `REJECTED_ACOUSTIC_INTEGRITY`.

The artifact is versioned and schema-validated. It is bound to both the schedule-screen signature and assisted variant, preventing reuse against a different experiment.

## Real negative control

The earlier Mega Man `assisted_005` best-fit nominee was screened again:

- Changed mappings: 2
- Accepted mappings: 1
- Rejected mappings: 1
- Mean word coverage: 69.23%
- Preflight state: `REJECTED_ACOUSTIC_INTEGRITY`

The rejected clip (`c000042`) intended:

> In order to modify your life cell, I must shut you down, Mega.

Whisper medium found only the beginning of that line amid adjacent audio. Ordered word coverage was 38.46%, and the sentence ending was absent. The other changed clip (`c000056`) achieved 100% coverage with both sentence boundaries present.

This predicts the already-observed rendered-dialogue failure without performing another full video render. The durable evidence is `evaluation/phase2_acoustic_preflight_mega_man_to_excerpt_20260721/semantic_acoustic_preflight.json`.

## Operational gate

`render-semantic-proof` now requires `--preflight` pointing to an accepted artifact for the same screen signature and semantic variant. A rejected, missing, mismatched, or stale preflight stops execution before rendering.

This preflight proves selected-donor transcript integrity only. It does not establish semantic preference, rendered voice separation, visual fit, or human quality.
