# Phase 2 — Required End-of-Phase Report

Date: 2026-07-22  
Phase state: `COMPLETE`  
Acceptance: 31/31 criteria pass; 610 repository tests pass.

## Executive finding

Phase 2 established a complete, local, optional semantic-passage path. Exact
transcript-vector similarity can make a useful donor distinction, but it is not
reliable enough to govern scheduling by itself. In the final proof it contributed a
lightly preferred rendered result only because legacy legality and compatibility
remained authoritative, acoustic evidence was checked before rendering, and a failed
semantic substitution was rolled back at candidate level.

The production default remains semantic weight zero. The evidence supports retaining
semantic assistance as an experimental, bounded reranker and proceeding to a small,
explicit Phase 3 dialogue-function taxonomy—not increasing semantic weight or treating
cosine similarity as understanding.

## 1. Selected model and rationale

Selected provider: `intfloat/multilingual-e5-small`, revision
`d829207ab28e6a5fb3aafb5d4c44111b8146db32`.

The local asset set is fixed by aggregate digest
`327ab8a846605d1bdac7f2cf14cfc259e53c28fa4303766665ac4c991c6e7cfd` and occupies
492,796,439 bytes. The implementation uses direct Transformers inference with the
model tokenizer, 384-dimensional float32 vectors, explicit `query: ` and `passage: `
prefixes, a deliberately reduced 256-token ceiling, deterministic head truncation,
attention-mask mean pooling, and L2 normalization.

This model was selected because it met the contract's local CPU requirement, offered
optional CUDA acceleration, produced compact normalized vectors, and provided a
multilingual-capable architecture without requiring a remote API. Direct Transformers
inference avoids adding the broader SentenceTransformers runtime. The 256-token cap is
a deliberate deviation from the model's larger native limit: film speech passages are
short, and the lower bound makes cache identity, memory, and worst-case work explicit.

Assets are never downloaded during production rendering. Missing or invalid assets
produce `DOWNLOAD_REQUIRED` or `UNAVAILABLE` and preserve the legacy Translation path.

## 2. Semantic retrieval quality

Five real E5 bundles used in the principal scheduling work account for 1,544 English
SpeechPassages: 966 normal embeddings and 578 explicitly marked low-information
embeddings, with zero failures and zero truncations. Their bundle metadata and vector
files occupy approximately 4.20 MiB in addition to the shared model assets.

Retrieval was useful but selective:

- The final bounded Mega Man → short-excerpt schedule raised mean selected cosine from
  `0.782622` to `0.797048` while preserving transcript completeness and measured hard
  compatibility constraints.
- One repaired semantic donor survived the final render at 100% rendered-word
  coverage. A second requested semantic donor failed in-context verification and was
  automatically rolled back.
- In blinded review, the semantic condition was selected for semantic relatedness,
  performance fit, intelligibility/completeness, and overall preference; the reviewer
  described it as “lightly better.”
- Restraint was common. In one 255-placement case, weights through 0.20 changed no
  selections. This is desirable when semantics adds no safe distinction.
- Across the initial four-pair, 1,170-placement cross-film matrix, 859 mappings
  (73.42%) had semantic coverage and three cases changed selections. Every initially
  changed case exposed a compatibility conflict or incomplete coverage, so none was
  promoted directly to rendering.

The retrieval conclusion is therefore narrow: embeddings can identify a useful
alternative absent from the legacy score, but high cosine alone is neither sufficient
nor consistently editorially useful. The Pareto, completeness, acoustic, render,
repair, and human-review gates are essential parts of the result.

## 3. Runtime and memory cost

The production-provider proof measured:

| Path | Cold load plus encode | Warm encode | Memory |
|---|---:|---:|---:|
| CPU, four passages | 7.285 s | 0.0165 s | not captured in the original proof |
| CUDA, first encode | 3.172 s | not separately recorded | 480,530,432 B peak GPU memory |
| CPU, supplemental 32-passage batch | 10.487 s | 0.1072 s mean | 1,216,262,144 B process peak working set |
| CUDA, supplemental 32-passage batch | cold start excluded as unstable | 0.01434 s mean | 489,932,800 B peak allocated; 511,705,088 B reserved |

The supplemental CPU process ended with 3,081,957,376 private bytes. That is a
process-wide Windows measurement including Python, PyTorch, Transformers, tokenizer,
and model runtime; it must not be read as model weights alone. These are single-session
local measurements, not cross-hardware benchmarks.

The asset footprint is approximately 470 MiB. Cached semantic bundles are much
smaller: the five principal real-model bundles total 4,407,749 bytes. Exact search is
linear and remained in the tens-of-milliseconds range for hundreds of entities in the
mechanics corpus; those mechanics timings used deterministic fake vectors and prove
service scaling, not E5 inference speed. No approximate index is warranted at the
current corpus scale.

Practical implication: cold model startup is noticeable; warmed embedding batches are
cheap, particularly on CUDA. Entity-level caching and resume should remain mandatory,
and production renders should not load the model when semantic mode is disabled.

## 4. Multilingual limitations

The selected architecture is multilingual-capable, but Phase 2 did not establish
multilingual Translation quality. All 1,544 passages in the five principal real-model
bundles were labeled English. A provider sanity check scored an English weather
sentence against a Spanish weather sentence at cosine `0.914877`; that proves the
provider produces cross-language proximity for one example, not that it preserves
dialogue function, register, idiom, irony, or film-specific meaning across languages.

Language metadata may itself be missing or inherited from ASR configuration. Phase 2
reports `same_language`, `cross_language`, or `unknown`, but does not transform those
states into a quality guarantee. There was no balanced multilingual corpus, no native
speaker review, no code-switching evaluation, and no language-pair calibration.

Recommendation: keep cross-language similarity report-only until Phase 3 or a later
calibration phase supplies balanced language pairs, native-speaker judgments, noisy-ASR
controls, and per-language-pair error reporting. Do not impose one global cosine
threshold across languages.

## 5. Observed false similarities

False or misleading similarity appeared in several forms:

1. **Topical but functionally unrelated text.** “The weather is lovely today” and “He
   drove to the stadium” scored `0.750066`. This is high enough to look meaningful on
   an uncalibrated scale despite offering no matching dialogue function.
2. **Generic short utterances.** Repeated “Oh,” “Ah,” and other interjections form dense
   similarity clusters. Several had transcript records while the scheduled audio
   contained no corresponding speech. Vector similarity cannot validate acoustic
   existence or boundary integrity.
3. **Duplicate or formulaic lines.** Exact or near-duplicate short transcripts can
   score 1.0 while differing materially in speaker, performance, conversational turn,
   or usefulness. Stable tie-breaking makes retrieval deterministic, not correct.
4. **Availability bias.** Partial semantic coverage caused extensively changed
   Wallace & Gromit → Mega Man schedules with dozens of performance, duration, speaker,
   visual, or completeness conflicts. Full-coverage and Pareto admission gates were
   added to prevent missing evidence from masquerading as improvement.
5. **Transcript/audio disagreement.** A selected donor may be semantically appropriate
   as text but incomplete, absent, or different in the exact WAV region. Selected-donor
   preflight, word-boundary repair, rendered word attribution, quarantine, and rollback
   were necessary to make text-level evidence safe to use.

These failures establish that cosine similarity is not a probability and does not
encode intention, speech act, emotion, character, relationship, scene meaning,
narrative purpose, irony, or comedy.

## 6. Effect on rendered quality

The first semantic render was worse than control: average word coverage fell from
72.22% to 51.34%, and failed mappings rose from one to two. It was correctly withheld
from human review. A later full-length guarded proof improved aggregate measurements
but its two actual semantic interventions failed in context, so it too was withheld.

The final proof added word-level boundary recovery, independent retranscription,
in-context word attribution, and candidate-level rollback. Its outcome was:

- control average rendered-word coverage: 69.95%;
- semantic average rendered-word coverage after repair: 72.22%;
- requested semantic changes: 2;
- semantic donors surviving final verification: 1;
- semantic donors rolled back: 1;
- failed semantic interventions after repair: 0;
- destination-voice residue regression: none detected;
- human judgment: semantic condition lightly preferred in all four questions.

This is a positive rendered result, but it is intentionally bounded. It demonstrates
that semantic assistance can survive the full safety and repair path and sometimes
improve the outcome. One reviewed case does not support enabling it by default or
learning a larger weight.

## 7. Recommendation for Phase 3 taxonomy design

Phase 3 should address the failure that embeddings cannot distinguish topical
relatedness from conversational function. It should begin with a small, auditable,
multi-axis taxonomy rather than a single flat “meaning” label.

Recommended axes:

1. **Surface form:** declarative, interrogative, imperative, exclamatory, fragment,
   non-lexical vocalization, unknown.
2. **Interaction function:** provide information, request information, request action,
   acknowledge, accept/commit, reject/refuse, greet/address, evaluate/react,
   clarify/repair, manage discourse, unknown/ambiguous.
3. **Sequence position:** initiating, responding, follow-up/continuing, closing,
   standalone, unavailable. This axis is legal only where ordered DialogueTurn evidence
   exists.

Design requirements:

- Labels must be multi-label where appropriate and always permit `UNKNOWN`,
  `AMBIGUOUS`, and `NOT_APPLICABLE`.
- SpeechPassage remains the dependable base unit. Do not infer sequence position where
  turn structure is absent; seven of eight audited FilmModels had zero DialogueTurns.
- Keep dialogue function separate from performance, emotion, character identity,
  relationship, visual action, scene meaning, and narrative role.
- Build a human-labeled seed set with written annotation rules, counterexamples, and
  inter-rater agreement before training or adopting a classifier.
- Balance the seed set across long/short utterances, questions and answers,
  interjections, noisy transcripts, animation/live action, and at least two reviewed
  language pairs before making multilingual claims.
- Store label source, annotator agreement, confidence, ambiguity, model/version, and
  provenance. Never silently replace human labels with inferred ones.
- Introduce taxonomy compatibility as a separate report-only contributor first. It
  may rerank only already-legal candidates and must pass the same zero-weight,
  Pareto-safety, acoustic, render, rollback, and blinded-review gates used in Phase 2.
- Evaluate whether function matching adds value beyond lexical and embedding scores.
  Required counterfactuals should include high-cosine/wrong-function and
  lower-cosine/right-function pairs.

The recommended Phase 3 target is therefore **dialogue-function compatibility with
explicit structural uncertainty**, not broader narrative understanding. Its first
vertical proof should show that a function-aware donor beats both the legacy and
embedding-only choices in a rendered, blinded comparison without weakening technical
quality.

## Evidence index

- `evaluation/phase2_e5_provider_proof_20260721.json`
- `evaluation/phase2_resource_benchmark_supplement_20260722.json`
- `evaluation/phase2_semantic_bundle_mechanics_20260721.json`
- `evaluation/phase2_turn_coverage_20260721.json`
- `evaluation/phase2_crossfilm_corpus_screen_20260721/semantic_corpus_screen.json`
- `evaluation/phase2_render_proof_mega_man_to_excerpt_word_repaired_v2_20260721/semantic_render_proof.json`
- `evaluation/phase2_render_proof_mega_man_to_excerpt_word_repaired_v2_20260721/blinded_review/semantic_review_result.json`
- `docs/review_artifacts/phase2_completion_status_20260722.md`
