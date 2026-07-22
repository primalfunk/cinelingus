# Phase 1 Translation Equivalence Proof

Date: 2026-07-21

## Result

**PASS** for the paired benchmark smoke/case_002_transition_sentence_integrity_6a92c69a.

The existing Translation schedule was ingested through source and destination
FilmModels, traced, reconstructed, rendered, and compared without changing
selection, repair, acceptance, or legacy execution code.

## Model and trace evidence

- Source FilmModel: ilm_1bb1ad996dbc773153a8, validation VALID.
- Destination FilmModel: ilm_f696039d2fed48f3573a, validation VALID.
- Schedule placements: 3.
- Schedule trace: READY, with donor passage, donor performance, destination
  passage/performance, speaker, verification, and editorial references resolved.
- Original and reconstructed schedule signature:
  71334e62c34587e438d4ebce071891f5b3e8db841dcfc00a2f22bc8916b49fb8.
- Equivalence checks: 18/18 passed.
- Differences: 0.
- Behavioral or invalid differences: 0.

The checks cover placement count and IDs, ordering, destination and donor timing,
media identity, speaker and performance references, adaptation, suppression,
fades, render operations, configuration signature, canonical payload, and trace
completeness.

## Render evidence

A fresh control was rendered directly from the existing cached schedule. The
model path was rendered from the reconstructed schedule using the same media,
duration, suppression mask policy, ambience reconstruction, sample rate,
channels, loudness, and fade settings.

- Frames: 721,104 on both paths.
- File size: 2,884,588 bytes on both paths.
- PCM difference RMS: 0.0.
- Byte-identical WAV: yes.

The archived benchmark WAV uses the historical 15.000-second render boundary.
Rendering the reconstructed schedule at that same boundary produces:

- 720,000 frames on both paths.
- 48 kHz stereo on both paths.
- 2,880,172 bytes on both paths.
- PCM difference RMS: 0.0.
- Non-identical container bytes, attributable to WAV header/container bytes;
  decoded PCM is identical.

Because decoded PCM is identical, the existing verification evidence receives
the same audio:

- rendered-dialogue verification: WARN on both paths;
- voice residue: NONE_DETECTED;
- editorial report: PASS;
- repaired placements: 0;
- rejected placements: 0;
- filter acceptance: pass;
- montage render acceptance: PASS.

## Rejected initial proof candidate

phase0_performance_strategy/strategy_002_performance_mismatch_d0394ea4 was
useful for adapter development but is not claimed as the formal end-to-end
control. Its cached schedule retains donor c000005 for placement 3 while the
archived editorial decision and verification identify repaired donor c000002.
The bridge now reports this as contradictory evidence and degrades trace
readiness instead of silently presenting the artifacts as a matched control.

## Compatibility conclusion

The FilmModel bridge is additive. It preserves the exact legacy schedule payload,
reconstructs it without behavioral changes, and produces identical rendered PCM.
Source media and Phase 0 artifacts remain read-only, and no legacy scheduling,
repair, rendering, rollback, resume, verification, or acceptance path was
redirected.
