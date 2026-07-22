# Phase 2 Repository Audit — Semantic Passage Representation

Date: 2026-07-21

## Viability decision

Passage-only Phase 2 is viable without redesigning FilmModel. SpeechPassage has stable IDs, original transcript, normalized comparison text, language state, source time range, confidence, and complete provenance. Vectors can remain outside the core model and key their cache to the FilmModel construction signature plus passage input and provider configuration.

The semantic scheduler must be injected only after existing legality/candidate generation. Disabled, report-only, unavailable, invalid, and zero-weight configurations must pass the existing score and candidate-ordering path without arithmetic or sort-key changes.

## Runtime and model assets

- PyTorch and NumPy are installed.
- `transformers`, `tokenizers`, and `sentence-transformers` are not installed.
- No local cache for `intfloat/multilingual-e5-small` exists.
- Production rendering must never trigger a download. A strict developer build may prepare assets explicitly.
- The upstream model is 384-dimensional and declares a 512-token model limit. Phase 2 intentionally uses the work-order 256-token limit to bound runtime and cache identity.

The production implementation should use direct `transformers` inference with attention-mask mean pooling and L2 normalization. This avoids requiring the broader `sentence-transformers` runtime while preserving the model card's documented pooling behavior. The exact immutable revision and local asset digest must be recorded after assets are explicitly prepared.

## Scheduler insertion points

`schedule._score_candidate` is the shared base compatibility scorer used by normal scheduling and editorial alternatives. Phase 2 should wrap or extend this scorer with an explicit semantic context. It must not alter `_score_candidate` output when influence is disabled or zero. Candidate generation, duration/stretch legality, performance grouping, speaker constraints, reuse constraints, transition rules, and failure-specific repair strategies remain authoritative.

`pipeline.schedule_from_artifacts` owns the replacement-schedule cache signature and is the correct orchestration point for optional semantic bundle discovery. Semantic configuration must join the schedule signature only when report-only or assisted evaluation is requested; ordinary Translation must not load a provider or semantic bundle.

## DialogueTurn coverage

The reproducible audit is `scripts/phase2_turn_coverage_audit.py`; output is `evaluation/phase2_turn_coverage_20260721.json`.

Across the eight Phase 1 representative models:

- SpeechPassages: 3,946
- DialogueTurns: 4
- Passages assigned to turns: 4 (0.1014%)
- Models with zero turns: 7 of 8
- Performances with ordered normalized turns: 1

The sole covered case is the analyzed 15-second Phase 0 excerpt. Five other models already preserve deterministic performance-to-passage membership for all passages, but their performance artifacts do not contain `ordered_turns`. Two destination variants show a source-ID mismatch: performance artifacts reference `w…` speaking-window IDs while their canonical speech view exposes `e…` event IDs.

## Narrow corrective candidate

A possible structural correction is to derive one structural turn per referenced SpeechPassage, ordered by canonical time within a performance, only when:

1. the performance has no explicit ordered turns;
2. passage membership is already explicit or can be matched unambiguously by containment;
3. every derived turn has a stable passage ID, valid time range, and complete provenance; and
4. no passage receives contradictory performance membership.

This would not infer dialogue function or meaning. It would, however, change FilmModel entity output and therefore requires a builder-version increment, migration report, determinism regression, and a new corpus audit. It should be implemented only after passage semantic contracts are stable; passage semantics do not depend on it.

## Hard gates

- No implicit model or dependency download.
- No vector payload in FilmModel JSON.
- No schedule change in disabled, report-only, unavailable, invalid, or zero-weight modes.
- No cosine score may bypass a hard constraint.
- No multilingual quality claim from language metadata alone.
- No turn semantics until the structural audit/correction produces valid deterministic turns.
