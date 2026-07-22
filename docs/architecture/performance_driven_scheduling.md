# Performance-Driven Scheduling

Translation uses detected performances as its default scheduling unit. Perception, judgment, and rendering remain separate: performance artifacts describe evidence; the scheduler makes deterministic choices; the renderer applies only accepted mappings.

## Performance evidence

Each performance records ordered turns, speech and silence intervals, speaker sequence, cadence, response timing, interruption rate, density, energy, scene category, shot boundaries, confidence, and declared adaptation capabilities. These fields extend the existing performance artifact and reuse the same Whisper, diarization, and shot evidence.

## Decision hierarchy

The scheduler evaluates all unused compatible donor performances for each destination performance and orders candidates by:

1. fallback tier;
2. local speaker-consistency violations;
3. active Matching-profile score, descending;
4. stable source performance ID.

The tiers are complete performance, adapted performance, consecutive turn sequence, legacy whole-line fallback, and preserve original. Terminal-turn removal is the only destructive adaptation in this phase. Discontinuous donor assembly is forbidden.

Monologue/dialogue class and incompatible speaker counts are hard rejections. Global speaker mappings are preferences during selection, while a stable local source-to-destination participant map is retained for every accepted performance.

## Matching profiles

Balanced, Rhythm, Deadpan, Volatile, Structural, and the retained creative profiles alter candidate weights before selection. Candidate scores and rejection reasons are recorded in `replacement_schedule.json`; profiles do not merely rescore a completed schedule.

## Rendering policy

Accepted speech spans use hard suppression with configurable padding and adaptive edge crossfades. The original destination bed remains untouched outside those spans. Failed and unmatched performances are not suppressed. The former attenuation behavior remains available as explicit `duck` mode.

The current background reconstruction strategy is neighboring non-speech continuity plus adaptive crossfades. Separated ambience stems and learned room-tone extension remain future refinements; the schedule does not claim that those sources exist when they do not.

## Diagnostics

Every accepted mapping records its performance pair, Matching profile, scheduler tier, similarity components, hard and soft constraint results, speaker mapping, timing operations, suppression mode, fallback reason, and candidate rejections. Run reports summarize complete couplings, adaptations, turn-sequence matches, linewise fallbacks, and preserved regions.

The scheduler is deterministic for identical artifacts and configuration. `whole_line_fill` remains callable directly for regression comparison and is also retained as Tier 4.

## Phase 2: rendered-soundtrack verification

Phase 2 closes the gap between a planned suppression policy and the audio actually produced.

- Every schedule retains the complete set of detected destination speech regions, including transcript evidence. This evidence is independent of whether a region receives a replacement.
- Hard suppression removes the destination bed across accepted speech regions. The renderer then selects the nearest interval outside all detected speech, repeats it only when necessary, and applies adaptive edge fades before mixing it beneath donor dialogue.
- Each reconstruction records its exact source and target bounds. If no safe non-speech source exists, the region remains silent and is counted as a silence fallback.
- With `verify_voice_residue` enabled, the completed WAV is transcribed with the configured Whisper model. Rendered words are contrasted with both the displaced destination transcript and the intended donor transcript.
- Verification reports `NONE_DETECTED`, `POSSIBLE_DESTINATION_SPEECH_DETECTED`, `INCONCLUSIVE`, or `UNAVAILABLE`. Transcript contrast is a review signal rather than source-separation proof.

The post-render pass intentionally costs an additional Whisper transcription in quality mode. It can be disabled explicitly, but the UI and reports will then say that residue was not tested.

Remaining calibration work requires held-out real media: tune speech-boundary padding, measure false positives and false negatives in transcript contrast, and establish acceptable silence-fallback rates by scene type.

## Phase 3: calibration and review gates

Phase 3 turns quality observations into repeatable acceptance evidence.

### Confidence-aware suppression

The configured suppression padding is now a minimum rather than a universal fixed value. Low-confidence regions receive additional protection, recovered filtered regions receive a further guard, short utterances receive attack and tail protection, and trailing padding is intentionally larger than leading padding. The complete decision is written to suppression_padding_report.

### Scored ambience selection

Every speech-safe ambience gap is evaluated for temporal proximity, available duration, preceding-side continuity, and prior reuse. The selected source records its component scores, whether looping was required, and the candidate count. Missing safe beds remain explicit silence fallbacks.

### Suspicious-region queue

The normal problem-region report now includes probable destination-speech residue, ambience silence fallbacks, and uncertain speech boundaries. Critical problems sort first. Existing problem-preview rendering therefore produces timed excerpts for these failures without a second review subsystem.

### Real-media corpus gate

The portable manifest format is demonstrated in config/performance_quality_corpus.example.json. Each case points to a completed run_report.json and may override corpus-wide thresholds.

Run the gate through the normal launcher:

    python run_cinelingus.py quality-corpus --corpus-manifest config/performance_quality_corpus.local.json --runs-root C:\path\to\completed-corpus-runs --output-dir output\quality-corpus

The command exits unsuccessfully when any case violates residue status, performance-first coverage, linewise fallback, preserved-original, silence-fallback, or problem-count limits. This makes the corpus suitable for regression and release gates once real cases have been curated.
