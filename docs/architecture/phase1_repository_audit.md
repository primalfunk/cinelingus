# Phase 1 Repository Audit — Minimal Cinematic Model Core

Date: 2026-07-21

## Scope and compatibility boundary

Phase 1 will add a normalized, versioned `FilmModel` beside the existing artifact
graph. It will not replace Phase 0 artifacts, alter their schemas, or redirect
legacy filter, repair, render, verification, rollback, or resume paths until the
model bridge has demonstrated behavioral equivalence.

The first proof case is the completed Phase 0 performance-strategy case
`strategy_002_performance_mismatch_d0394ea4`. It contains both source and
destination cache manifests, the complete analysis graph, a replacement
schedule, editorial evidence, render verification, and a final output.

## Canonical source artifacts

| FilmModel area | Current canonical input | Notes |
| --- | --- | --- |
| media | `movie.json` | SHA-256 media hash is the primary identity; path is local traceability only. |
| timeline/speech | `timeline.json`, `dialogue_events.json`, filtered variants | Destination voice windows and source transcript events have similar but distinct contracts. |
| speakers | `speaker_map.json` | Explicit backend, alignment, fallback, and assignment evidence already exist. |
| shots/transitions | `shots.json` | Transitions and boundary stability are embedded in the shot artifact. |
| visual evidence | `visual_report.json`, `visual_performance.json` | Preserve measurements and current caveats; do not broaden interpretation. |
| performances | `performance.json`, `performance_library.json` | Rich structural/audio/visual evidence; current performance IDs are order-derived. |
| moments | `cinematic_moments.json` | Already uses evidence-hashed moment IDs where generated; not universal. |
| schedule | `replacement_schedule.json` | Required for the Translation equivalence bridge, not intrinsic film state. |
| editorial | `editorial_decisions.json`, `editorial_report.json`, repair artifacts | Run- or schedule-scoped evidence; optional in a film-only build. |
| discovery | cache `manifest.json` | Records logical artifact names and local paths, but not every producer signature. |

## Identity findings

- Media SHA-256 hashes are stable and suitable for deriving `film_id`.
- Existing moment IDs are evidence-hashed and should be preserved through a
  reversible source-ID mapping.
- Performance IDs (`p000000`), some turn fallbacks, speaker labels, and editorial
  placement IDs are derived from deterministic order. They are stable for an
  unchanged artifact, but not independently stable if membership changes.
- FilmModel entity IDs will therefore hash canonical evidence: namespace,
  `film_id`, canonical time range, source artifact identity, original object ID,
  and the relevant configuration signature. Source IDs remain in provenance and
  the migration ID map.
- ID generation is collision-checked and versioned. No ID is based solely on
  filesystem enumeration or in-memory list order.

## Timing findings and decision

Existing artifacts express seconds as JSON numbers and generally round detected
boundaries to milliseconds. FilmModel adopts seconds, half-open intervals
`[start, end)`, millisecond normalization, and a 1 ms comparison tolerance.
Zero-length intervals are retained only when explicitly supported by source
evidence and are reported by validation. Source values remain recoverable through
provenance.

The first live proof build exposed a 17 ms transcript overshoot beyond the
ffprobe duration. The adapter now clamps only end-boundary overshoot within the
larger of one frame or 50 ms. The unmodified source range and the normalization
policy are recorded in provenance; larger violations remain validation errors.

## Provenance and confidence findings

Provenance is currently distributed across media hashes, schema/tool versions,
configuration signatures, backend fields, source IDs, and artifact paths.
FilmModel will normalize these into referenced provenance records without
discarding the source artifacts.

Existing confidence fields are heterogeneous: detector scores, heuristic scores,
categorical statuses, and occasional calibrated-looking values share the same
numeric range. Phase 1 will preserve each value with its scale, interpretation,
evidence source, calibration state, and fallback state. It will not relabel a
heuristic score as a probability.

## Cache and serialization findings

- Current cache identity is `<media_hash>/<role>` with artifact freshness checked
  by per-stage `config_signature` values.
- `stable_hash` already provides canonical key ordering for signatures, and
  `write_json` provides atomic local writes.
- FilmModel adds a builder signature covering media identity, source artifact
  content/signatures, schema version, builder version, ID policy, timing policy,
  and adapter versions. Creation timestamps and local path spelling are excluded
  from semantic identity.
- Canonical serialization sorts object collections by stable ID and dictionary
  keys recursively. Reload must preserve the canonical byte representation.

## Adapter and migration decisions

- Adapters are artifact-specific and non-destructive. They do not rewrite cache
  inputs.
- Every migrated object records its source artifact and source object identity.
- Original fields that do not yet have normalized homes remain available through
  the artifact registry and a source snapshot/reference; normalization never
  becomes the only copy of evidence.
- Dialogue turns will be a structural normalized view of existing ordered
  performance turns. No dialogue-function semantics are inferred.
- Transition and moment capabilities are `PARTIAL` or `UNAVAILABLE` when their
  dedicated evidence is absent; absence is not represented as an empty,
  successful analysis.
- Editorial observations remain explicitly run/schedule scoped.

## Known limitations disclosed from the outset

Phase 1 does not provide semantic embeddings or similarity, dialogue-function
classification, character identity, active-speaker attribution, relationship
inference, semantic scene understanding, or narrative event understanding.
Capability records and human-readable reports must expose these exclusions.

## Implementation order

1. Versioned contracts and schema.
2. Artifact adapters and deterministic builder.
3. Structural, temporal, identity, provenance, confidence, and capability validation.
4. Deterministic serialization, cache integration, report, lookup API, and CLI.
5. Lossless schedule bridge and Translation equivalence proof.
6. Bounded live-action, animation, short-form, and feature-length corpus validation.
