# Phase 1 Completion Report — Minimal Cinematic Model Core

Date: 2026-07-21  
Status: **COMPLETE**

Phase 1 now provides a deterministic, versioned FilmModel over existing Cinelingus evidence, plus a lossless bridge to the Phase 0 Translation schedule. It does not redirect or redefine the legacy pipeline. The final regression suite is **545 passed**, the eight-case bounded corpus is fully valid and deterministic, and the proof render is decoded-audio identical.

## A. Implementation summary

Added `src/cinelingus/cinematic_model/` modules for schema policy, model construction, artifact adapters, stable IDs, confidence, capabilities, provenance, serialization, validation, cache decisions, reports, lookup, schedule bridging, developer commands, and bounded-corpus measurement. Added the FilmModel and schedule-bridge JSON Schemas, CLI integration, focused contract/builder/corpus tests, evaluation artifacts, and architecture/operator documentation.

Major decisions:

- FilmModel is additive and opt-in; Phase 0 artifacts remain canonical and unmodified.
- The media hash plus inspection signature establishes film identity. Paths are trace references, never identity.
- Normalized entities carry stable film-local IDs and referenced provenance; source IDs remain reversibly mapped in migration reports.
- Missing optional evidence degrades capabilities to `PARTIAL`, `FALLBACK`, or `UNAVAILABLE`.
- The schedule bridge preserves the complete original schedule payload in an envelope, so reconstruction does not depend on a lossy normalized view.
- A mismatch between an archived editorial donor and cached schedule donor is reported as degraded traceability rather than reconciled by guesswork.

Deviations: the CLI accepts an artifact directory or cache-root/media-hash/role identity rather than raw media alone, because raw media is insufficient to build the evidence model and Phase 1 must not run analysis. The corpus-native 11-minute short-form item lacks analysis artifacts; acceptance uses an analyzed 15-second Phase 0 excerpt and records the native-cache gap below. No GUI controls were added.

## B. Final schema

The normative contracts are `schemas/film_model.schema.json` and `schemas/schedule_bridge.schema.json`, both declaring JSON Schema Draft 2020-12. FilmModel schema version is `1.0.0`; builder, adapter, validator, bridge, and ID-policy versions are independently recorded.

Required top-level FilmModel fields are: schema and builder versions, film ID, media, timeline, capabilities, all eight entity collections, provenance, confidence summary, source-artifact registry, validation state, and construction signature.

| Object | Required structural contract | Optional/evidence-dependent content |
| --- | --- | --- |
| MediaIdentity | film/media IDs, source reference kind, filename, duration, inspection signature | container, streams, rate, resolution, channels, corpus ID |
| Timeline | start/end/duration, time-base policy, indexes, tolerance | frame/sample rate, discontinuities |
| Shot | stable ID, time range, provenance | boundaries, transitions, performance/moment links, detector evidence |
| Transition | stable ID, time range, provenance | conservative classification, confidence, neighboring shots, guard evidence |
| SpeechPassage | stable ID, time range, provenance | original/normalized transcript, language, speaker and structural links |
| SpeakerCluster | stable ID, provenance | intervals, backend/fallback/stitching evidence, passage/turn/performance links |
| DialogueTurn | stable ID, time range, provenance | ordered passages, speaker candidates, neighbors, structural timing evidence |
| Performance | stable ID, time range, provenance | transcript/audio/visual source detail and all available cross-references |
| CinematicMoment | stable ID, time range, provenance | boundary, stillness, speech, transition, and linked-object evidence |
| EditorialObservation | stable ID, scope, provenance | placement/object references, failures, repair, verification, and final state |
| ProvenanceRecord | stable ID, media/artifact references, producer and builder versions | source range/IDs, configuration, construction, migration history |
| SourceArtifact | stable ID, logical type, provenance | locator, hashes, versions, signature, compatibility and requirement state |
| Capability | declared status | producer, configuration, coverage, confidence, limitations |

Missing is not treated as empty: absent evidence produces an empty entity collection plus an explicit degraded capability. Unknown scalar values are `null`; unsupported features are `UNAVAILABLE` with a reason. Minor-compatible additions may retain schema major version 1; breaking field or interpretation changes require a major schema increment and cache invalidation.

## C. Stable ID specification

Namespaces are `film`, `shot`, `transition`, `speech`, `speaker`, `turn`, `performance`, `moment`, `editorial`, `artifact`, `placement`, and `provenance`. IDs contain a namespace and the first 20 hex characters of a SHA-256 canonical evidence digest.

- Film input: ID-policy version, media hash, and movie-artifact inspection signature.
- Entity input: ID-policy version, namespace, film ID, canonical time/source evidence, source object identity, and relevant artifact/configuration identity.
- Collision behavior: `StableIdRegistry` retains the full digest and fails if two different digests produce the same truncated ID.
- Migration mapping: each build emits source-to-model ID mappings by artifact/object type.
- Evidence: reordered artifact dictionaries and changed volatile creation timestamps produce identical IDs, signatures, capability declarations, validation reports, and canonical serialization.

## D. Artifact migration report

Consumed logical artifacts are `movie`, `dialogue_events` or `timeline`, `shots`, `speaker_map`, `performance`, `cinematic_moments`, `clip_library`, `replacement_schedule`, and optional `editorial_report` or `editorial_decisions`. Redundant timeline and editorial views are deterministically de-duplicated in favor of the richer canonical view.

Preserved data includes raw artifact content signatures/locators, source object IDs, transcript text, timing, speaker backend/fallback state, shot/transition evidence, performance source detail, moment boundary language, schedule fields, editorial failures/recommendations/final states, configuration signatures, and producer versions. Normalization is limited to stable IDs, canonical millisecond time, normalized comparison text alongside original text, explicit references, confidence records, and capability status.

Intentionally omitted normalized claims are listed in section I. Unmodeled source fields remain recoverable through the registered original artifact rather than being destructively rewritten. Cross-media artifact hashes fail the build. Near-media-boundary overshoot is clamped only within the documented tolerance and records both the original range and migration action. Risks remain around legacy order-derived source IDs and artifact variants that omit ordered turns.

## E. Validation report

Validation covers structural, identity, temporal, referential, provenance, confidence, and capability integrity; the bridge separately validates schedule traceability and reconstruction signatures. Invalid models cannot be accepted for reproduction.

Resolved failures encountered during implementation:

- A 17 ms transcript overshoot was handled by a narrow, provenance-recorded media-boundary normalization policy.
- A candidate proof run had contradictory schedule/editorial donor evidence; it is now classified `DEGRADED` and was rejected as formal proof evidence.
- Both published schema files had an empty dialect-key caused by Windows shell interpolation. They now contain literal `$schema` keys and a regression test.

Bounded results: four smoke and four deterministic standard cases; **8/8 VALID**, zero warnings, deterministic rebuilds, cache hits, and unchanged source-media stat tuples. Cases cover live action, animation, short form, feature scale, strong diarization, partial evidence, and schedule-bearing artifacts. Full detail is in `evaluation/phase1_bounded_corpus_20260721.json`.

## F. Cache report

The construction signature covers media hash, FilmModel schema, builder, adapter, ID policy, timing policy, and relevant source-artifact content signatures. It excludes volatile timestamps, artifact enumeration order, and unrelated output files. A schema, builder, relevant artifact, media hash, or ID-policy change requires rebuild; `--force` overrides reuse. Compatible valid models return `CACHE_HIT`.

Measured reloads ranged from 0.6 ms for the short excerpt to 40.2 ms for the 83-minute feature in the smoke tier; all eight cases returned cache hits. Model bundles are written atomically and include model, build, migration, validation, human report, and signed manifest files.

## G. Translation equivalence report

Formal case: `smoke/case_002_transition_sentence_integrity_6a92c69a`.

- Source/destination FilmModels: VALID.
- Schedule: three placements, trace readiness READY, 18/18 equivalence checks, no differences, identical canonical reconstruction signature.
- Fresh control vs reconstructed-schedule render: 721,104 frames, 2,884,588 bytes, RMS 0.0, byte-identical.
- Archived-duration comparison: 720,000 decoded frames per path, RMS 0.0; container bytes differ only in WAV header/container data.
- Verification/editorial evidence: no new residue, verification, editorial, repair, acceptance, or delivery-state change. Decoded PCM identity gives both paths identical downstream evidence.
- Nondeterminism: none in model or schedule content. Runtime measurements are observational and intentionally outside canonical artifacts.

See `evaluation/phase1_translation_equivalence_20260721.json` and `docs/review_artifacts/phase1_translation_equivalence_proof_20260721.md`.

## H. Compatibility report

- Legacy filters: unchanged; FilmModel is not required for existing execution.
- Phase 0 editorial system: repair selection, thresholds, rollback, neighborhood commit, resume, acceptance, and delivery gates are unchanged.
- Corpus framework: existing plans/caches are read; no source-media sidecars or analysis jobs are created.
- Reporting: additive FilmModel bundle/report and trace/equivalence artifacts only.
- CLI: additive build, validate, report, compare, trace, reconstruct, and schedule-compare commands.
- GUI: no public changes.
- Regression status: **545 passed**.

## I. Fields intentionally omitted

Phase 1 does not model semantic embeddings, semantic similarity, dialogue functions, performance functions, character or actor identity, active-speaker attribution, relationships, semantic scene meaning, narrative events or causality, location identity, emotion understanding, or inferred character demographics. Capability records explicitly report the corresponding functions as unavailable.

## J. Runtime and storage impact

Times are one local Windows run and are baselines, not hard guarantees.

| Case | Duration | Build | Reload | Validate | Model | Peak traced memory | Model/source JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short excerpt | 15.0 s | 33.7 ms | 0.6 ms | 0.5 ms | 0.069 MiB | 0.516 MiB | 1.02x |
| live-action WKYK | 21.9 min | 401.5 ms | 11.1 ms | 3.1 ms | 1.569 MiB | 4.428 MiB | 2.58x |
| animation Mega Man | 23.9 min | 1.152 s | 12.4 ms | 6.6 ms | 2.939 MiB | 8.485 MiB | 2.24x |
| Wallace & Gromit feature | 83.4 min | 2.676 s | 40.2 ms | 10.3 ms | 5.295 MiB | 15.215 MiB | 2.59x |

Across the standard tier, the worst observed build was 4.729 s, reload 30.8 ms, validation 11.3 ms, model 4.446 MiB, and peak traced memory 27.972 MiB. Provenance/source registries account for roughly 42–53% of serialized model size; this is deliberate traceability overhead and the clearest future compaction target. Measured transcript duplicate payload was negligible except 0.6 KiB in the short excerpt. No Whisper, Pyannote, or visual analysis ran.

Practical Phase 1 budgets: under 6 seconds for an artifact-rich feature build, under 100 ms for cached model load, under 50 ms validation, under 10 MiB canonical model size, and under 64 MiB traced Python allocation for the present corpus scale.

## K. Known limitations

- Architectural: the normalized model indexes evidence but does not reason semantically; schedule reproduction uses a lossless legacy payload envelope.
- Data: the native corpus short-form item has inventory metadata but no analysis cache; a fully analyzed Phase 0 excerpt covers short-form mechanics.
- Artifact: several caches omit ordered performance turns or speaker maps, so dialogue-turn/diarization capabilities legitimately degrade or remain empty.
- Validation: validation proves contract and link integrity, not that detector evidence is artistically or semantically correct.
- Migration: legacy order-derived IDs are stable only while the source artifact is unchanged; reversible mappings mitigate but do not redefine them.
- Performance: provenance is 42–53% of model bytes and full in-memory canonicalization scales with object count. Current measured scale remains well within proposed budgets.

## L. Phase 2 recommendation

Begin with embeddings on `SpeechPassage`, using original transcript plus language and transcription provenance. It is the most consistently populated, narrowly scoped semantic unit. Do not make `DialogueTurn` the first embedding target: turns are structurally sound where ordered-turn evidence exists, but several representative caches produce zero normalized turns. First improve/standardize turn derivation coverage without semantic labels.

Introduce a separate immutable `TurnSequence` only after turn coverage is consistent. It should reference ordered turn IDs and performance context rather than duplicating transcripts. Semantic artifacts should inherit film/media ID, source artifact/provenance IDs, source object IDs, time range, transcript signature, language, model/version, configuration signature, calibration state, and construction signature.

Cache embeddings per film, semantic artifact type, entity ID, canonical input signature, language, model/version, and configuration. Detected/configured language is already retained per speech passage where present, but corpus language coverage has not been calibrated and multilingual quality must not be inferred from a code alone.

No FilmModel defect blocks a bounded Phase 2 speech-passage embedding prototype. Broad turn-sequence semantics should wait for the dialogue-turn coverage gap to be resolved. Character identity, active-speaker attribution, dialogue-function labels, scene meaning, and narrative inference remain separate later phases.

## Acceptance disposition

All 20 Phase 1 acceptance criteria are satisfied. The representative native short-form cache gap is explicitly resolved for mechanical acceptance through an analyzed short excerpt and remains disclosed as a data limitation. Phase 1 is closed; further semantic implementation belongs to Phase 2.
