# Alteration guarantee

Cinelingus treats operational completion and significant alteration as separate contracts. A valid MP4 is not a successful experiment when it is materially equivalent to the anchor film.

## Success boundary

The public executor accepts these altered outcomes:

- `REQUESTED_SUCCESS`: the requested filter passes duration and alteration acceptance.
- `NORMALIZED_SUCCESS`: normalized parameters still produce an accepted alteration.
- `PRIMARY_ONLY_SUCCESS`: the compatible primary filter produces an accepted alteration.
- `DURATION_REPAIRED_SUCCESS`: an altered requested result is capped to the audio-supported extent and revalidated.
- `ALTERED_FALLBACK_SUCCESS`: the requested path failed or was too weak, and the universal full-timeline renderer produced a measured alteration.

`UNALTERED_RECOVERY` returns playable media without surfacing a rendering exception, but it is explicitly not a successful experiment. It exists only for the case where both the requested path and the independent altered renderer fail.

## Requested-filter acceptance

Requested filters are evaluated from their machine-checkable `filter_acceptance.json` evidence. The v1 minimum is:

- filter acceptance passes;
- effective authored extent is at least 5% of the timeline;
- at least three source placements exist; and
- authored placements occupy at least two timeline regions.

Effective authored extent is the greater of declared dialogue coverage and mapped-dialogue duration divided by render duration. A technically successful but negligible filter result proceeds to the universal altered renderer.

## Universal renderer

For two or more films, the complete anchor video is paired with supporting audio for the full audio-supported extent. Every supporting film is included in a normalized mix when more than one is present. The original anchor audio is not mapped into this fallback.

For one film, the complete anchor video is retained while the entire audio track receives a deterministic FFmpeg transformation using high-pass, low-pass, equalization, echo, limiting, and AAC encoding.

Both modes first attempt video stream copy and then H.264 video transcoding. They always use the duration established by complete-input preflight.

## Independent evidence

Universal results must pass:

- final container, video, and audio duration checks;
- full-timeline audio-coverage declaration;
- non-anchor media-hash provenance for multi-input replacement, or a declared full-track filter chain for one input; and
- sampled decoded-audio comparison at five distributed positions, with at least 60% of comparable samples measurably different from the anchor.

The evidence is written as `alteration_acceptance/*.json` and embedded in `configuration_outcome.json`.

## Assurance boundary

The software guarantee assumes readable media with audio, working FFmpeg tools, writable storage, and enough free space. Power loss, process termination, codec defects, hardware faults, and exhausted storage remain environmental failures. Within those prerequisites, filter or analysis failure is contained by the altered renderer, and unchanged passthrough is never reported as successful alteration.
