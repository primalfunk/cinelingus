# Phase 2 — Multi-role speech harmonization checkpoint

Date: 2026-07-21

## Outcome

FilmModel construction can now merge speech evidence from both cache roles instead of discarding one role's view. The primary role remains authoritative for reference binding, while supplemental role passages retain their own IDs, timestamps, transcripts, confidence, and provenance.

The Wallace & Gromit model was rebuilt from:

- `destination_video/timeline.json` — 370 destination speech windows
- `source_dialogue/dialogue_events.json` — 484 source utterances

The resulting FilmModel is `VALID` with 854 speech passages and two explicitly reported speech views. Its semantic bundle is `READY (VALID)`.

## Implementation

- `build-film-model` accepts repeatable `--include-speech-role` arguments for `source_dialogue` and `destination_video` cache directories.
- Supplemental speech artifacts participate in the deterministic build signature and are recorded as distinct source artifacts.
- Dialogue-event and timeline adapters can run more than once in a single build.
- Capability coverage aggregates passage and speech-view counts across roles.
- Local source IDs are first-binding: a supplemental role cannot overwrite a primary role's performance-reference mapping if both roles reuse the same ID.
- The older same-role rule remains intact: when raw dialogue events and a redundant timeline are supplied under the original unqualified keys, dialogue events remain the preferred canonical view.

## Fixed four-pair corpus result

Artifact: `evaluation/phase2_crossfilm_corpus_harmonized_20260721/semantic_corpus_screen.json`

| Measure | Before | Harmonized |
| --- | ---: | ---: |
| Mappings | 1,169 | 1,169 |
| Semantically represented | 1,122 | 1,169 |
| Weighted semantic coverage | 95.9795% | 100% |
| Directly represented | 1,067 | 1,167 |
| Weighted direct coverage | 91.2746% | 99.8289% |
| Exact direct joins | 891 | 1,167 |
| Boundary bridges | 111 | 0 |
| Text bridges | 65 | 0 |
| Performance aggregate fallbacks | 55 | 2 |
| Unrepresented mappings | 47 | 0 |
| Render nominees | 0 | 1 |

Every corpus case now reports full semantic coverage. The two remaining aggregate fallbacks are in Magic School Bus → WKYK; no placement is unrepresented.

## Newly admitted nominee

Wallace & Gromit → Mega Man now produces `ASSISTED_CANDIDATE_SELECTED` through the guarded Pareto schedule:

- 282/282 mappings have exact direct semantic evidence.
- Four placements change.
- All four admissions preserve the protected compatibility axes.
- Two changes form one globally safe two-cycle; two are direct conflict-free substitutions.
- Unique donor count remains 282, maximum donor reuse remains 1, and conflict count remains 0.

This is a candidate for rendering and human review, not an automatic quality claim. The evidence establishes full traceability and guarded scheduler admissibility; perceptual superiority still requires review.

## Validation

- FilmModel builder and CLI coverage tests: included in the full suite.
- Full regression suite: **591 passed**.
- Harmonized corpus state: `RENDER_NOMINEES_AVAILABLE`.
