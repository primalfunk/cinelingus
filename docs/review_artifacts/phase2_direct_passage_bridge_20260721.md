# Phase 2 direct passage bridge checkpoint — 2026-07-21

## Implemented

The semantic scheduler now has two narrowly bounded direct-evidence bridges for source and destination analyzers that describe the same media with slightly different segmentation:

- `direct_passage_boundary_bridge`: accepts a unique canonical-start match within 0.5 seconds; from 0.5 through 1.0 seconds it also requires normalized transcript similarity of at least 0.5.
- `direct_passage_text_bridge`: accepts exact normalized multi-token text as before, and now permits an exact one-token match within 3.0 seconds only when its temporal lead over the next candidate is at least 0.5 seconds.

Ambiguous candidates are rejected. Bridge provenance is retained on every mapping and is reported separately from exact direct evidence and performance aggregates.

## Fixed-corpus result

Across the four cross-film screens:

- selected mappings: 1,169;
- semantically represented: 1,122 (95.98%);
- direct placement evidence: 1,067 (91.27%);
- exact direct: 891;
- boundary bridge: 111;
- time-disambiguated text bridge: 65;
- performance aggregate fallback: 55;
- unrepresented: 47;
- invariant failures: 0;
- guarded admissions/render nominees: 0.

Before this bridge, direct placement evidence covered 856 mappings (73.23%) and 241 used performance aggregates. The new bridge therefore adds 211 directly represented placements and removes 186 aggregate fallbacks. Overall represented coverage rises from 93.84% to 95.98%.

## Per-case provenance

| Pair | Mappings | Exact | Boundary | Text | Performance aggregate | Unrepresented |
|---|---:|---:|---:|---:|---:|---:|
| Mega Man → Magic School Bus | 256 | 256 | 0 | 0 | 0 | 0 |
| WKYK → Wallace & Gromit | 371 | 371 | 0 | 0 | 0 | 0 |
| Wallace & Gromit → Mega Man | 282 | 6 | 111 | 65 | 53 | 47 |
| Magic School Bus → WKYK | 260 | 258 | 0 | 0 | 2 | 0 |

The bridges affect only Wallace & Gromit → Mega Man, which is consistent with the known segmentation difference between the source-dialogue and destination-video analysis artifacts for that film.

## Guarded-admission result

No opportunity passes the complete placement-level Pareto gate.

The strongest fully direct two-cycle remains rejected because its second leg lowers mapping-level legacy score by 0.0158. The only remaining globally reusable positive cycle depends on `c000094` ("Gromit?"), for which the FilmModel contains no nearby direct passage. That absence is retained rather than inferred away.

## Claim boundary

This checkpoint establishes deterministic evidence recovery and provenance accounting. It does not establish rendered quality or human preference, and it does not authorize acoustic preflight without a guarded schedule nominee.

Durable aggregate: `evaluation/phase2_crossfilm_corpus_screen_20260721/semantic_corpus_screen.json`
