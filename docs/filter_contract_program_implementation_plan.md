# Cinelingus Filter Contract Program: Work Order and Implementation Plan

> **Superseded for public presentation and current inventory counts.** This historical plan predates the 40-entry engineering registry and the separate public apparatus catalog. Use [Public apparatus catalog](architecture/public_apparatus_catalog.md) for names, disciplines, operating modes, status, capability tiers, and Procedure eligibility. Retain this document only for historical contract-program rationale.

Status: proposed next implementation phase  
Source work order: `CINELINGUS FILTER CONTRACT PROGRAM IMPLEMENTATION WORK ORDER`  
Prepared against the executable repository state on 2026-07-13.

## 1. Outcome

This document combines the supplied work order with repository-derived implementation logic. The work order remains the creative authority. The derived plan prevents its conceptual vocabulary from being mistaken for machinery that already exists.

The next phase will:

1. define and validate a formal contract schema;
2. write code-derived contracts for the eight runnable filters;
3. write and review contracts for the twenty catalog-only filters;
4. classify every contract by real engineering requirements;
5. implement only reviewed filters whose requirements are supported by the existing engine;
6. add specimen, Procedure, graph, field, DSP, or renderer infrastructure only when an accepted contract proves it necessary.

## 2. Repository-derived architectural truth

### Available now

- A 28-filter, seven-family registry with stable namespaced identifiers.
- Eight runnable filters: Translation, Self Shuffle, Echo, Drift, Contagion, Possession, Foreshadow, and Bloom.
- Versioned recipes and normalized transformation plans.
- Deterministic strategy builders for the four new Filter Laboratory operators and compatibility adapters for the four legacy operators.
- Shared dialogue-event, performance, speaker, scene, shot, and speaker-graph artifacts.
- Schema-driven GUI controls, implementation-state gating, progress, reports, previews, publication, and retained audit artifacts.
- Proven schedule renderers for replacement dialogue, overlay, soundtrack-preserving mutation, muting/ducking, time displacement, and complete-source timelines curtailed only by required supporting audio.
- Deterministic seeds, explicit rejection reasons, filter metrics, invariant validation, requested/actual backend reporting, and final audio provenance.

### Conceptual but not executable yet

No repository implementation currently defines:

- an authoritative evolving `Specimen` object;
- a multi-step `Procedure` executor;
- reusable relationship, temporal, performance, or semantic graph interfaces beyond filter-specific structures;
- reusable active fields or field simulation;
- state variables such as Identity Entropy, Temporal Strain, Attention Gradient, or Narrative Curvature;
- runtime relationship domains beyond Dialogue, Performance, Identity, and Time;
- semantic embeddings, semantic clusters, or a general emotional-analysis artifact;
- a general DSP/voice-transformation operation model.

`FilterExecutionContext.semantic_features` and `.emotional_features` are placeholders, not populated analysis products. Current emotion-like evidence is limited to transcript text and performance proxies such as estimated energy, speaking rate, pauses, density, and timing.

### Planning consequence

Contracts may describe future Procedure behavior, fields, and state variables now, but the contract must label whether that behavior is:

- `conceptual`: specified but not executable;
- `single_step_validated`: proven when operating on an original analyzed film;
- `multi_step_validated`: proven when operating on a specimen already modified by another filter.

No filter may claim multi-step Procedure support until a real Procedure test passes. This is not permission to build a speculative Procedure engine during the contract-writing phase.

## 3. Derived implementation rules

### 3.1 Contract authority

Create one machine-valid contract per registry definition. Runtime registry metadata may summarize the contract, but it must not become a second creative authority.

Proposed source-of-truth paths:

- `schemas/filter_contract.schema.json`
- `filter_contracts/<family>/<filter>.json`
- `src/cinelingus/filter_lab/contracts.py`
- generated human catalog: `docs/filter_contract_catalog.md`

Each contract will carry:

- schema and filter version;
- review status: `draft`, `reviewed`, or `accepted`;
- every section required by the supplied work order;
- implementation class A-F;
- analysis availability and missing prerequisites;
- Procedure support status;
- named invariants, metrics, failure behavior, and preview requirements.

### 3.2 Honest implementation classification

A filter is Class A only if its accepted behavior can be produced from artifacts and render operations that exist today. A name that sounds simple does not make its implementation Class A.

In particular:

- identity filters must define behavior for `pyannote_partial` maps and exclude unknown/fallback identities from hard identity mappings;
- semantic or emotion filters are Class A only if their law explicitly uses existing deterministic transcript/performance evidence;
- a contract requiring embeddings, sentiment, semantic clusters, learned emotion, or voice conversion must be reclassified and remain unavailable until that prerequisite exists;
- filters that remove dialogue must specify whether the existing mute/duck renderer is sufficient;
- filters that layer voices must prove the current mixer handles overlap, loudness, clipping, and provenance;
- arbitrary randomness is forbidden; controlled uncertainty must be seeded, weighted, measured, and reproducible.

### 3.3 Activation gate

`implemented=True` is the final step, not the first. A catalog path becomes runnable only after all of these pass:

1. accepted contract;
2. normalized parameters and registry metadata;
3. deterministic strategy and dispatcher registration;
4. hard-invariant property tests;
5. sparse/missing-artifact failure tests;
6. representative preview selection;
7. cached real-artifact plan proof;
8. final MP4 render, audio activity, provenance, and surviving-report verification;
9. regression proof for all previously runnable filters.

### 3.4 Evolving-specimen rule

Until a Procedure runtime exists, first-wave strategies consume the current analyzed film and produce a normalized plan plus a specimen-delta declaration in the contract. They must not pretend to have consumed prior filter output.

The first accepted contract whose defining law cannot be expressed without prior transformed state becomes the trigger for a narrowly scoped specimen/Procedure implementation. That implementation must preserve immutable source provenance while making the preceding plan, mappings, state deltas, and generated fields available to the next operator.

## 4. Contract schema deliverable

The schema should require these structured sections:

| Section | Required content |
| --- | --- |
| Identity | Filter ID, name, family, versions, review status, implementation class |
| Creative proposition | One-sentence cinematic law |
| Relationship domains | Reads, modifies, preserves |
| Specimen inputs | Required current-state objects and whether each exists today |
| Required analysis | Artifacts, minimum confidence, accepted fallback policy |
| Candidate generation | Eligible material and exclusions |
| Selection rules | Priorities, weights, tie breaks, uncertainty |
| Transformation | Exact changes by domain |
| Progression | Constant, increasing, decreasing, scene/field-dependent |
| State variables | Reads, writes, strengthens, weakens |
| Fields | Creates, reads, modifies, decay/propagation law |
| Hard invariants | Named boolean or measured properties |
| Validation | Metrics and pass thresholds |
| Failure behavior | Empty/sparse/low-confidence degradation |
| Preview | Representative region selection |
| Renderer | Scheduler/graph/field/DSP/renderer requirements |
| Procedure behavior | First-step and later-step behavior plus support status |

Contract validation tests must prove:

- exactly one contract exists for every registry definition;
- contract ID, family, and version agree with the registry;
- accepted contracts contain no placeholder language;
- implemented filters have accepted contracts;
- named hard invariants have corresponding test identifiers;
- requested artifacts are either available or explicitly recorded as blockers;
- Class A contracts request no unavailable analysis or render primitive.

## 5. Delivery phases

### Phase 0: Vocabulary and schema

1. Add the contract JSON schema and loader.
2. Freeze canonical domain and requirement vocabularies.
3. Add contract/registry completeness tests.
4. Add a generator for the human-readable contract catalog.

Exit condition: an empty or placeholder contract cannot be accepted accidentally.

### Phase 1: Reference contracts for runnable filters

Write contracts from current behavior, in this order:

1. Foreshadow — strongest existing hard temporal invariant.
2. Possession — stable identity mapping and fallback-speaker policy.
3. Contagion — graph, exposure, infection state, and progression.
4. Bloom — field-like progression metrics without claiming a general field engine.
5. Self Shuffle — legacy behavior and temporal/source-line guarantees.
6. Translation — two-film roles, speaker mapping, performance fit, and audio gate.
7. Echo — overlay and ducking behavior.
8. Drift — positive progressive displacement and soundtrack preservation.

Exit condition: every existing behavior relied upon by tests and reports is described by an accepted contract.

### Phase 2: Contracts for catalog-only filters

Write all twenty contracts before activating more filters. Ambiguity is recorded as a blocker rather than filled with implementation guesswork.

Provisional implementation hypotheses for contract review:

| Filter | Provisional law to review | Likely class from current evidence |
| --- | --- | --- |
| Flashback | Replacement dialogue may originate only from sufficiently earlier narrative events. | A |
| Recollection | A later cue may recall previously heard dialogue under explicit identity/time trigger rules. | A if cue is identity/time-based; B if semantic similarity is required |
| Amnesia | Previously available dialogue becomes progressively ineligible and cannot re-enter the active pool. | A/B depending on whether silence/removal or reassignment defines the law |
| Chorus | One utterance manifests through multiple distinct eligible identities at one destination event. | A with mixer validation |
| Doppelgänger | A stable, non-self identity pairing causes one speaker to shadow or exchange another's dialogue. | A |
| Paranoia | Candidate choice increasingly favors dialogue carrying a defined threat/uncertainty signal. | B unless the contract accepts deterministic lexical/performance proxies |
| Wonder | Candidate choice favors a defined novelty/awe signal while preserving intelligibility. | B unless the contract accepts deterministic lexical/performance proxies |
| Spiral | Dialogue revisits earlier relationships in cycles with measurably increasing displacement or instability. | A |
| Dialect | Undefined until “dialect” is declared as selection, lexical substitution, performance change, or voice transformation. | Deferred; possibly B or E |
| Mutation | Undefined transformation law. | Deferred |
| Whisper | Undefined whether this means gain/EQ, performance selection, or synthesized whispering. | Deferred; likely E |
| Split Personality | Undefined identity partition and switching law. | Contract first; likely A/C |
| Dream | Undefined temporal/semantic distortion law. | Contract first; likely B/D |
| Regret | Undefined backward-looking semantic/emotional law. | Contract first; likely B |
| Optimist | Undefined positive semantic/emotional selection law. | Contract first; likely B |
| Exhaustion | Undefined performance-selection versus DSP law. | Deferred; likely B or E |
| Möbius | Undefined loop closure and termination law. | Contract first; likely A/C |
| Ouroboros | Undefined self-consuming Procedure or graph law. | Deferred; likely C/F |
| Shed Skin | Undefined identity replacement versus timbral transformation law. | Deferred; likely C/E |
| Venom | Undefined propagation and transformation law distinct from Contagion. | Contract first; likely C/D/E |

Exit condition: all contracts are either accepted or explicitly blocked with a named unanswered question.

### Phase 3: Contract review

Apply the three supplied review questions and these additional engineering gates:

- Can the law be proven through named invariants?
- Is every requested artifact real?
- Is the actual renderer sufficient?
- Is fallback behavior honest for partial diarization and sparse dialogue?
- Can a seeded run be reproduced exactly?
- Does later-step Procedure behavior distinguish conceptual from validated support?

Exit condition: the first implementation wave contains only accepted, executable contracts.

### Phase 4: First implementation wave

Target order from the work order, subject to the reviewed class:

1. Flashback
2. Recollection
3. Amnesia
4. Chorus
5. Doppelgänger
6. Paranoia
7. Wonder
8. Spiral

Implement one vertical slice at a time. For each filter:

1. finalize its registry parameters and descriptions from the accepted contract;
2. add its deterministic strategy;
3. register progress stages and dispatcher entry;
4. emit contract-named metrics and validation;
5. add its preview policy;
6. prove a cached plan;
7. render and inspect a real final artifact;
8. activate the GUI path only after the gate passes.

If Paranoia or Wonder requires unavailable semantic/emotional analysis, stop that filter at the contract gate and continue to the next accepted Class A filter. Do not smuggle a speculative analyzer into the wave.

### Phase 5: Procedure trigger, not speculative Procedure work

After at least two accepted filters have explicit later-step behavior, select one concrete two-filter Procedure. Implement only the minimum evolving-specimen state needed to execute that pair. Required acceptance:

- operator two consumes operator one's normalized output state rather than rebuilding from the original film;
- original media provenance remains immutable;
- state/field deltas are serialized;
- incompatibilities are enforced;
- identical Procedure recipe and seed reproduce the same plan;
- final render and reports identify contributions from both filters.

## 6. Verification matrix

| Layer | Required proof |
| --- | --- |
| Contract | Schema validation, registry parity, no placeholders in accepted contracts |
| Strategy | Determinism, candidate/rejection accounting, hard invariants |
| Analysis | Required artifacts exist; requested/actual backend and fallback coverage reported |
| Preview | Region demonstrates the defining law, not merely any mapped line |
| Render | Correct soundtrack policy, no clipping/truncation, expected audio activity |
| Publication | Published MP4 exists; recipe, plan, report, and provenance survive cleanup and reference it |
| Regression | All prior runnable filters and compatibility aliases remain green |
| Procedure | Later operator consumes transformed state; contribution lineage is visible |

## 7. Immediate next work package

The next coding task should be limited to Phase 0 and Phase 1:

- add the contract schema/loader/catalog generator;
- author the eight code-derived reference contracts;
- add parity and invariant-linkage tests;
- do not implement a new filter in the same change.

This produces the reference language needed to review the twenty planned contracts without conflating contract discovery with implementation.

---

# Appendix A: Supplied Filter Contract Program Work Order

## Purpose

The current architecture has reached an important milestone.

The filter registry, execution pipeline, specimen model, reporting, recipes, previews, cache management, and filter-family framework are now sufficiently mature that the primary remaining work is no longer infrastructure.

The remaining work is definition.

Every filter must become a precisely specified cinematic operator with well-defined behaviour.

This phase should produce a complete library of behavioural contracts before attempting to rapidly implement every remaining filter.

The objective is to ensure that every future operator:

- behaves consistently;
- composes naturally with future Procedures;
- modifies the evolving specimen rather than the original film;
- contributes to a coherent cinematic physics rather than existing as an isolated feature.

## Primary Goal

Shift development from infrastructure-first to operator-first.

Do not continue adding new architectural layers unless a specific operator genuinely requires them.

Instead:

Define.

Classify.

Implement.

Validate.

Repeat.

## The Filter Contract

Every existing and planned operator must receive a formal behavioural contract.

The contract becomes the authoritative specification.

Implementation follows the contract rather than the filter name alone.

Each contract should contain:

### Filter Name

### Family

### Creative Proposition

One sentence describing the cinematic law.

Example:

> Dialogue may originate only from events that occur later in the narrative.

### Relationship Domains

Identify which domains are read.

Identify which domains are modified.

Possible domains:

- Dialogue
- Performance
- Identity
- Time
- Narrative
- Attention
- Dynamics

### Specimen Inputs

Specify exactly which parts of the current specimen state are required.

Examples:

- relationship graph
- identity mappings
- dialogue pools
- speaker graph
- temporal graph
- performance graph
- active fields
- state variables

### Required Analysis

Specify the required analysis artefacts.

Examples:

- speaker identities
- scene graph
- dialogue embeddings
- performance metrics
- semantic clusters

### Candidate Generation

Define how possible source material is collected.

Do not describe implementation.

Describe behaviour.

### Selection Rules

Describe how the final candidates are chosen.

Specify priorities.

Specify tie-breaking behaviour.

Specify weighted uncertainty.

### Transformation Behaviour

Describe precisely what changes.

- Dialogue
- Performance
- Identity
- Time
- Narrative
- Attention

### Progression

Describe whether behaviour changes across runtime.

Examples:

- constant
- increasing
- decreasing
- scene-dependent
- field-dependent

### State Variables

Specify:

- Reads
- Writes
- Strengthens
- Weakens

Examples:

- Identity Entropy
- Temporal Strain
- Attention Gradient
- Narrative Curvature

### Fields

Specify which reusable fields the operator creates or modifies.

Examples:

- Identity Diffusion Field
- Temporal Field
- Semantic Drift Field
- Narrative Gravity Field

### Controlled Uncertainty

Specify exactly where deterministic weighted choice is permitted.

Never use arbitrary randomness.

### Hard Invariants

Specify behaviour that may never be violated.

Examples:

- Foreshadow may never source dialogue from the past.
- Possession may never map a speaker onto itself unless explicitly enabled.

### Validation

Specify measurable success criteria.

### Failure Behaviour

Describe graceful degradation.

### Preview Strategy

Specify the most representative preview region.

### Renderer Requirements

Classify whether the operator requires:

- existing scheduler
- scheduler extension
- graph extension
- field extension
- DSP
- renderer changes

### Procedure Behaviour

Most importantly:

Describe how the operator behaves when it is NOT the first operator.

The contract must answer:

> What changes if another operator has already transformed the specimen?

## Implementation Classes

Classify every operator.

### Class A

Existing scheduling engine.

No new infrastructure.

### Class B

Scheduling extensions.

### Class C

Relationship graph extensions.

### Class D

Field simulation extensions.

### Class E

Audio/DSP extensions.

### Class F

Renderer extensions.

The purpose is to expose future engineering effort honestly.

## First Implementation Wave

Once contracts are complete, implement only operators that fit the existing machinery.

Recommended order:

1. Flashback
2. Recollection
3. Amnesia
4. Chorus
5. Doppelgänger
6. Paranoia
7. Wonder
8. Spiral

These operators should activate the currently sparse Memory and Emotion families while exercising the existing scheduler and specimen architecture.

Do not prematurely implement operators that require speculative DSP or voice synthesis.

## Deferred Operators

Explicitly postpone operators whose intended behaviour is still ambiguous.

Examples:

- Whisper
- Dialect
- Mutation
- Exhaustion
- Shed Skin
- Ouroboros

These should receive contracts but may remain unimplemented until their requirements are fully understood.

## Contract Review

Before implementation, review each contract against three questions:

1. Can another developer predict the behaviour without seeing the code?
2. Does the operator still make sense inside a multi-step Procedure?
3. Does the operator describe a cinematic law rather than merely an editing trick?

If any answer is "no," revise the contract before implementation.

## Regression Requirement

Existing runnable operators:

- Drift
- Echo
- Translation
- Self Shuffle
- Contagion
- Possession
- Foreshadow
- Bloom

must also receive formal contracts.

The current implementation becomes the reference behaviour.

Do not allow undocumented behaviour to accumulate simply because those filters already function.

## Design Principle

The Filter Registry is now complete enough.

The next creative work is not writing code.

It is discovering the laws of Cinelingus.

Every operator should feel like a natural law acting upon an evolving cinematic specimen.

If a future operator cannot be expressed as a precise behavioural contract, it is not yet ready to implement.

If it can be precisely described, the architecture should make its implementation comparatively routine.

Success for this phase is not measured by the number of new filters.

Success is measured by the creation of a coherent and extensible language of cinematic transformation that will guide the remainder of the project.
