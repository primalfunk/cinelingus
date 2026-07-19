# ADR-001: Montage Composition Foundation

- Status: Approved
- Product line: Cinelingus 0.2.x
- Decision date: 2026-07-15
- Affected areas: filter contracts, visual analysis, audio rendering, montage planning, acceptance

## Incorporated authority

This ADR is the durable implementation index for three approved design documents, which remain authoritative and versioned with it:

- [Montage Design Decisions](../montage_I_design.txt)
- [Montage Foundation Design and Product Authorization](../montage_II_foundation.txt)
- [Montage Foundation Implementation Formalization Addendum](../montage_III_addendum.txt)

The ADR summarizes their operative rules; where detail differs, the approved source documents govern. Initial montage artifact schemas are version `1.0`. Existing filter contracts remain at their current versions until each filter enters the montage-native migration cohort.

## Decision

Cinelingus is a cinematic composer, not a video splicer. The public unit of montage composition is the **cinematic moment**: one or more consecutive shots that express a coherent visual idea. Shots remain internal evidence and every selected moment retains source-film, scene, shot, visual-boundary, and audio-boundary provenance.

The implementation hierarchy is:

1. cinematic integrity;
2. speech intelligibility;
3. honest evidence and capability reporting;
4. deterministic planning;
5. filter-law compliance;
6. source participation and structural shape;
7. requested duration.

Target duration never authorizes an unsafe cut. Respect comes before surprise.

## Filter laws

Every montage-aware filter declares four independent laws:

- **Visual Law**: imagery that may be selected.
- **Temporal Law**: chronology and ordering.
- **Dialogue Law**: treatment of speech.
- **Audio Law**: treatment of production sound, ambience, music, and effects.

One governing montage relationship—such as identity, memory, cause, contrast, escalation, symmetry, repetition, irony, association, collision, or silence—controls selection and ordering. The remaining criteria support that relationship in this order: speech intelligibility, visual continuity, rhythm, source diversity, and novelty.

## Clean boundaries

Core analysis may claim only literal evidence: detected cuts and transitions, speech timing, optical motion, camera-motion magnitude, stillness, duration, and conservative grouping. It must not claim to understand completed gestures, gaze, intention, or human action.

Long takes may be divided only at internal candidates supported by literal evidence such as completed speech, silence, low optical flow, camera stabilization, stable framing, or a transition-like reset. If no candidate meets the safety threshold, the planner must:

1. use the complete take;
2. choose a different moment;
3. relax target duration;
4. invoke an explicit fallback or fail.

It may never insert an arbitrary duration cut or silently lower a safety threshold.

## Capability tiers

- **Core** requires no heavyweight learned visual model. It uses transitions, fades, dissolves, speech boundaries, motion, stillness, and conservative grouping. CPU operation remains supported.
- **Enhanced Vision** optionally adds face, mouth, gaze, pose, subject tracking, and related evidence.
- **Enhanced Audio** optionally adds local source separation, residual-vocal detection, and stem-aware rendering.

All analysis is offline-first. Large models are optional downloads outside the source tree and run-artifact directories. Every backend records name, version, license/source inventory, redistribution status, device, thresholds, confidence, fallback, and timestamp. Reduced capability must produce conservative behavior rather than fabricated confidence.

## Assertion provenance

Human annotations and machine assertions remain distinct. Every assertion uses one capability tag:

- `HUMAN_ANNOTATION`
- `CORE_HEURISTIC`
- `ENHANCED_VISION`
- `ENHANCED_AUDIO`
- `FALLBACK_INFERENCE`
- `CONTRACT_RULE`
- `PLANNER_DERIVATION`

Assertions record their name, value, confidence, backend/version, source artifact, supporting evidence IDs, fallback status, and explanatory note. An inferred safe boundary must retain the evidence from which it was derived.

## Audio laws and actual methods

Requested Audio Law and Actual Audio Method are separate fields. Approved actual methods are:

- `STEM_SEPARATED_REPLACEMENT`
- `MIXED_BED_REPLACEMENT`
- `ORIGINAL_REALITY`
- `GENERATED_OR_CONSTRUCTED_BED`
- `SILENCE_OR_MINIMAL_BED`

The long-term default is Dialogue Replacement plus Continuous Ambient Bed. Its fallback order is reliable separated stems, acceptably clean mixed-bed replacement, Original Reality when permitted, minimal/constructed bed, then candidate rejection. Mixed source audio must never be labeled as a clean ambient stem.

Partially intelligible or competing ghost speech fails acceptance for a claimed clean ambient bed. Residual-speech evaluation distinguishes none, non-speech vocal residue, unintelligible speech residue, partially intelligible ghost speech, and competing ghost speech.

## Target shape and source participation

- Standard: 90–150 seconds.
- Quick Preview: 30–60 seconds.
- Extended: 3–6 minutes.
- Experimental: unlimited.
- Typical moment: 4–12 seconds; average approximately 7 seconds.
- Standard outputs contain at least 8 distinct moments, preferably 12–25.
- Quick Preview may use 4–8 when its contract declares that fallback.

Every montage has Beginning, Development, Climax, and Resolution roles. Required films must remain visually recognizable. Default source shares are anchor 40–60%, secondary 20–40%, and tertiary 10–25%, normalized for film count and overridden only by an explicit filter law.

## Calibration corpus

The portable corpus manifest stores source IDs and hashes, resolver keys, timestamps, shot/scene IDs, labels, notes, confidence, split, and schema version. Machine-specific paths live only in an ignored local resolver that verifies media hashes.

Development, tuning, and held-out splits must independently represent clean and broken entrances/exits, speech, silence, cuts, dissolves, fades, long takes, reactions, J/L cuts, motion interruption, and ambiguity. Adjacent boundaries from one scene remain in one split. The initial minimum sizes are 40 development, 30 tuning, and 40 held-out distinct examples.

## Production readiness

A successful render never implies readiness. Formal evaluation compares Core planning with a naive complete-shot sampler and produces one verdict: `EXPERIMENTAL`, `PREVIEW`, or `PRODUCTION_READY`.

Initial safety targets include 98% avoidance of audible-word cuts, 95% avoidance of sentence-onset and transition cuts, 90% avoidance of severe subject/camera-motion interruption, 85% human acceptability of accepted boundaries, 80% retention of Core-detectable acceptable boundaries, and zero fabricated long-take boundaries. Fixed inputs and seeds reproduce the same plan-level result.

The baseline comparison targets at least 50% relative reduction in severe speech failures, 35% relative reduction in severe motion failures, and 30 percentage-point improvement in human-rated acceptability. Relevant held-out strata must be large enough to make each percentage meaningful before a production-ready verdict is permitted.

## Reproducibility

Identical media hashes, artifacts, backend versions, contract version, configuration, capabilities, seed, and planner version produce identical moment IDs, eligibility, selections, source assignments, ordering, visual/audio boundaries, overlaps, structural roles, laws, fallbacks, provenance, and acceptance decisions. Byte-identical encoded media is not required across FFmpeg, codec, operating-system, driver, encoder, or architecture changes.

## Phase 1 sequence

1. Artifact schemas and stable provenance objects.
2. Conservative Core moment analysis.
3. Deterministic planner, fallbacks, and source-balance enforcement.
4. Naive baseline and evaluation harness.
5. Self Shuffle migration.
6. Flashback, Foreshadow, Recollection, then Dream.

Phase 1 remains `EXPERIMENTAL` until held-out evidence satisfies the approved readiness requirements.

## Amendment policy

Later doctrine changes require a dated amendment or superseding ADR, identification of affected rules, and contract/schema version changes where necessary. Historical decisions are preserved rather than silently rewritten.

## Destination-intro non-privilege amendment

For filters that do not require chronology, the earliest eligible moment, timestamp zero, and the first detected scene receive no implicit opening preference. Opening selection is governed by structural role, governing relationship, cinematic integrity, and seeded diversity. Plans record the opening's eligible-timeline position and selection basis. Formal evaluation measures opening diversity across seeds or materially different inputs; repeated source-intro selection is reported as planner bias and prevents a production-ready verdict.

## Universal Cinelanguage segmentation amendment

Montage planning is the shared segmentation and sentence-forming layer of Cinelanguage, not a filter-specific feature. Every successful filter operation must author its visual, temporal, dialogue, and audio laws over a validated montage plan. Filter implementations may change selection relationships and chronology, but may not bypass cinematic-moment eligibility, plan provenance, structural-role assignment, opening-selection evidence, or encoded render acceptance. Compatibility entry points must route into the same planner rather than maintain parallel segmentation systems.

## Revision history

- 2026-07-15: ADR created; all three approved authority documents incorporated; Phase 1 marked `EXPERIMENTAL`.
- 2026-07-15: Destination-intro non-privilege rule added; planner and production-readiness evaluation must report source-start bias.
- 2026-07-15: Montage plans made the universal Cinelanguage segmentation layer for every filter operation.
- 2026-07-15: Continuous-audio rule added; source soundtracks form the montage bed and sustained dead air fails acceptance.
- 2026-07-15: Source-audio duration rule added; requested length became a ceiling, shortening replaced implicit repetition, and repetition authorization became plan provenance.
- 2026-07-15: Montage eligibility moved ahead of filter placement; structural analysis retains the full timeline while rendered placements are constrained to complete audio-safe moments.

## Continuous-audio amendment

Every rendered montage must carry meaningful continuous audio. Selected cinematic moments retain their source soundtrack as the base layer; filter-authored dialogue ducks that layer only across its replacement interval. An audio-free visual reel plus sparse dialogue is forbidden. Encoded acceptance rejects a sustained below-threshold run longer than 0.75 seconds. If meaningful source audio cannot cover a region, planning must select another moment or remove the silent region while preserving the montage boundary contract.

Eligibility is a scheduling input, not only a post-scheduling acceptance test. Filters may analyze all destination windows to establish their governing relationship, but they author rendered placements only in windows fully contained by audio-safe cinematic moments. A deterministic, law-preserving rescue is required when seeded selection skips every eligible placement.

## Source-audio duration and non-repetition amendment

Requested runtime is a ceiling. If eligible non-repeated source audio cannot support it, the planner shortens the selected video and may explicitly relax the configured minimum duration. It must not loop, repeat, or pad source material merely to meet a runtime request. Repetition is legal only when the active filter contract or an explicit filter-plan parameter authorizes it. Plans record requested, available, and resolved durations together with the authorization basis and every observed repeated source placement; unauthorized repetition is a planning failure.
