# Phase 2 — Bounded Clip-Boundary Recovery

## Outcome

Phase 2 now has a read-only-media boundary-recovery stage for acoustically rejected semantic donor clips. It searches quarter-second shifts while preserving the original clip duration, prevents a candidate from crossing neighboring dialogue events, and writes a derived clip-library overlay instead of altering the canonical cache.

A candidate must preserve the intended lexical content without adjacent words in a discovery context and an independently assembled confirmation context. Short breathy Whisper spellings such as `Sorryhh` may corroborate `Sorry`; substitutions and additional words do not.

The opportunity audit and selected-donor acoustic preflight now reject `adjacent_dialogue_before` and `adjacent_dialogue_after`. The opportunity health-cache version was advanced so earlier coverage-only acceptances cannot be reused.

## Wallace proof

Artifact: `evaluation/phase2_clip_boundary_repair_wallace_20260721/semantic_clip_boundary_repair.json`

- Rejected clips searched: 8
- Fixed-duration boundary candidates: 72
- Discovery matches: 8, all for `c000095` / `Sorry.`
- Independently confirmed candidates: 0
- Repaired clips admitted: 0
- Final state: `NO_REPAIR_CANDIDATE`

The Wallace evidence is appropriately conservative. The `Sorry` clip was heard as `Sorry` in the discovery reel, but the later, leading-context-reducing candidate did not survive independent transcription. The original window was also observed elsewhere as `Ooh, sorry`, which is now explicitly disqualifying. The other seven rejected clips never produced their intended lexical content under the bounded search.

This result means the repair mechanism works, but these eight source clips remain quarantined. It does not claim that the dialogue is absent from the movie; it establishes that the current coarse Whisper metadata cannot support a repeatable, contamination-free fixed-duration donor cut.

## Implementation

- `src/cinelingus/semantic/clip_boundary_repair.py`
- `src/cinelingus/semantic/opportunity_acoustics.py`
- `src/cinelingus/semantic/acoustic_preflight.py`
- `src/cinelingus/render_verification.py`
- `schemas/semantic_clip_boundary_repair.schema.json`
- Developer command: `repair-semantic-clip-boundaries`

## Verification

Full regression suite: **600 passed**.

## Remaining boundary work

The current search deliberately changes position but not duration. The next safe extension is a word-timestamp or phoneme/onset-assisted cut generator that can shorten a coarse 2–3 second Whisper segment, followed by the same two-context acoustic gate. Duration expansion should not be the next step because it increases the risk of importing neighboring dialogue.
