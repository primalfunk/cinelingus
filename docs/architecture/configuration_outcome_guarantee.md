# Configuration outcome guarantee

Cinelingus separates cinematic-law quality, significant alteration, and operational completion. The strict filter pipeline may reject a schedule whose analysis, provenance, alteration, or output acceptance is inadequate. The user-facing executor contains those failures and attempts an independent full-timeline alteration before considering unaltered recovery.

## Altered success classes

- `REQUESTED_SUCCESS`: the requested filter completed and passed stream, duration, and alteration acceptance.
- `NORMALIZED_SUCCESS`: normalized parameters produced an accepted alteration.
- `PRIMARY_ONLY_SUCCESS`: an unproven stack was reduced to a compatible primary filter which produced an accepted alteration.
- `DURATION_REPAIRED_SUCCESS`: an altered requested result was capped to the audio-supported extent and revalidated.
- `ALTERED_FALLBACK_SUCCESS`: the requested result failed or was too weak, and the independent universal renderer produced a full-timeline measured alteration.

`UNALTERED_RECOVERY` is not an experimental success. It returns playable, duration-correct media only when both the requested and universal altered paths fail. Legacy `PASSTHROUGH_SUCCESS` records remain schema-readable, but v3 execution does not emit that status.

## Resolution and rescue rules

Configuration resolution is deterministic. Unknown parameters are ignored, invalid declared values use contract defaults, and every adjustment is serialized. Ordered stacks continue to use the compatibility compiler. A non-certified stack does not execute implicitly; its first implemented non-Bloom primary filter is attempted alone.

Preflight establishes the authoritative duration: the complete anchor video timeline limited by the shortest required supporting-audio stream. Final acceptance checks the container, video stream, and audio stream independently. The default tolerances are 250 ms for the container and video packet boundary and 50 ms for audio. Complete overlong output is capped and revalidated; short output proceeds to altered fallback.

Requested output must also pass the alteration contract. Insufficient authored extent is treated like a filter failure even when the MP4 and the filter's basic acceptance pass. The universal renderer replaces or transforms full-timeline audio, validates duration and provenance, and compares decoded audio samples with the anchor. Details live in [Alteration guarantee](alteration_guarantee.md).

If universal alteration fails, the final recovery publisher attempts a lossless stream copy, then H.264/AAC transcoding, then a same-container copy. This output is duration-validated but explicitly labeled `UNALTERED_RECOVERY`.

User cancellation remains cancellation and does not produce an unwanted recovery artifact.

## Assurance boundary

The guarantee assumes at least one readable video with audio, working FFmpeg tools, and a writable output volume with enough space. It cannot prevent power loss, hardware failure, process termination, storage exhaustion, or external codec defects. Within those prerequisites, requested-filter failure is converted into an altered fallback instead of being surfaced as a run-ending error.

The GUI and public `Pipeline.execute_configuration(...)` service use this outcome boundary. Strict developer methods remain available for tests and diagnosis.

Run the default-directory assurance audit with:

```powershell
python tools/certify_configuration_outcomes.py
```

The audit probes all discovered videos, verifies declared parameters, resolves every ordered filter pair, forces a real requested-filter failure, and validates the altered fallback's duration, provenance, and sampled audio difference.
