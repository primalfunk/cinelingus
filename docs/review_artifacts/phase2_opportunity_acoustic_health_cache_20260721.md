# Phase 2 — Opportunity-pool acoustic health cache

Date: 2026-07-21

## Outcome

Phase 2 can now acoustically screen the bounded, globally admissible Pareto opportunity pool before guarded schedule nomination. Results are cached per exact clip audio digest, transcript, trim, Whisper configuration, coverage threshold, verifier version, and audit context policy.

The Wallace & Gromit → Mega Man opportunity set was fully exhausted without another expensive render. Thirteen unique source performances were audited; eight were quarantined by the pool audit and the remaining five were rejected by authoritative selected-trim preflight. The terminal screen reports `NO_CONFLICT_FREE_CHANGED_ASSISTED_CANDIDATE`.

## Implementation

- New command: `audit-semantic-opportunities`.
- Inputs: semantic screen, clip library, source performance artifact, bounded maximum source-performance count, optional durable cache.
- The audit includes globally admissible direct candidates and both donors required by admissible two-cycles.
- Uncached clips are grouped into independent batches of at most three, limiting Whisper context bleed.
- Every batch rejection is retranscribed in an isolated one-clip confirmation reel before it can authorize quarantine.
- Batch acceptance is provisional only: the ordinary exact scheduled-trim preflight remains authoritative before rendering.
- Cache replay does not load Whisper or rebuild reels when all identities are current.
- `screen-semantic-schedules --opportunity-audio-audit` imports rejected source performances and inherited repair lineage.
- Later preflight and render failures continue extending the same quarantine chain.
- Missing clips are explicitly rejected rather than silently omitted.

## Real audit

Artifact: `evaluation/phase2_opportunity_acoustic_audit_wallace_to_mega_20260721/semantic_opportunity_acoustic_audit.json`

- Audited source performances: 13
- Audited clips: 13
- Pool-accepted performances: 5
- Pool-rejected performances: 8
- Initial small transcription batches: 5
- Isolated rejection confirmations: 10
- Subsequent cache replay: `REUSED`, 0 batches, 0 confirmations

The two-stage policy corrected several context-sensitive first-pass outcomes. This confirms that large mixed reels are unsuitable as direct quarantine evidence for short cinematic interjections.

## Authoritative trim recovery

The five pool survivors were nominated through deterministic repair screens and checked at their exact scheduled trim:

| Source performance | Intended transcript | Trim result |
| --- | --- | --- |
| `p000196` | Oh! | Rejected; heard “No!” |
| `p000115` | Geronimo! | Rejected; heard “You’re running low!” in the later isolated trim |
| `p000174` | Went… | Rejected; mismatched fragment |
| `p000087` | Going | Rejected |
| `p000147` | Something | Rejected |

These failures, combined with the eight pool-level confirmed failures, quarantine every globally admissible performance in this bounded screen. No candidate is promoted to render.

## Interpretation

The cache substantially reduces futile sequential nomination while preserving restraint:

- It does not treat ASR as semantic understanding.
- It does not bypass exact selected-trim preflight.
- It does not relax timing, reuse, completeness, performance, or visual constraints.
- It does not convert a pool acceptance into render authorization.
- It provides a durable, evidence-backed terminal state when the legal semantic opportunity set is acoustically unsupported.

## Validation

- Full regression suite: **596 passed**.
- Source media remained read-only.
- No rejected candidate was rendered or released for human review.
