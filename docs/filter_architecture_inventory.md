# Cinelingus Filter Architecture Inventory

This inventory is the Phase 1 baseline for the Filter Laboratory migration. It records current behavior before adapters change how transformations are selected or described.

## Existing execution paths

| Filter | Current entry points | Inputs | Schedule behavior | Rendering | Primary cache/artifacts |
| --- | --- | --- | --- | --- | --- |
| Self Shuffle | `Pipeline.execute_transformation("self_shuffle")`, `Pipeline.run_self_shuffle()`, `Pipeline._build_single_film_mutation_schedule()` | One film used as source and destination | Speaker-aware deterministic shuffle, whole-line fill, changed-line enforcement, minimum temporal separation in short-reel selection | Dialogue-only replacement soundtrack muxed to the original picture | `self_shuffle_schedule.json`, speaker maps, dialogue events, clip library, performances, transformation report |
| Echo | `Pipeline.run_mutation("echo")`, `Pipeline._build_single_film_mutation_schedule()` | One film | Every configured nth line is repeated at a fixed delay, bounded by film duration and repeat limit | Mutated dialogue mixed over original audio; original is ducked at echo placements by default | mutation schedule/plan/report under `output/mutations/echo`, shared single-film analysis caches |
| Transposition | `Pipeline.execute_transformation("movie_masher")`, `Pipeline.run_all()`, `Pipeline.run_best_short_remix(app_mode="Movie Masher")` | Destination picture/speaking windows plus a separate source-dialogue film | Performance-aware whole-line filling with stable cross-film speaker mapping when validated | Replacement soundtrack rendered against destination windows and muxed to destination picture | destination replacement schedule, source/destination speaker maps and performances, transformation plan/report |
| Drift | `Pipeline.run_mutation("drift")`, `Pipeline._build_single_film_mutation_schedule()` | One film | Each line moves later by an offset interpolated from `starting_offset` to `maximum_offset` over source time | Mutated dialogue mixed over the original soundtrack; detected speech regions are muted by default | mutation schedule/plan/report under `output/mutations/drift`, shared single-film analysis caches |

## Stable behavior that adapters must preserve

- Existing media-role handling and published output paths.
- Self Shuffle changed-line and temporal-separation guarantees.
- Transposition source/destination identity, validated speaker mapping, performance-aware scheduling, and the final audio-activity gate.
- Echo's fixed-delay repetition defaults and Drift's progressive positive time offset.
- Cache reuse based on media hashes and content-dependent signatures.
- Best Short candidate selection and final MP4 publication behavior.
- Honest requested-versus-actual Whisper and diarization backend reporting.

## Existing duplication and migration seams

- GUI transformation labels, input visibility, and dispatch are hard-coded independently.
- Transposition and Self Shuffle use `movie_masher.transformations`; Echo and Drift use `movie_masher.mutations`.
- Full Movie and Best Short build single-film mutation schedules through separate paths.
- Mutation reports and transformation reports use different shapes.
- The existing `cinematic_filters.py` controls candidate scoring style; it is not the new product-level filter registry and must remain distinct.

## Approved internal schemas

The implementation uses `movie_masher.filter_lab` to avoid colliding with the existing segment-filter module.

- `FilterDefinition`: immutable registry metadata, parameter schemas, dimensions, input/artifact requirements, output support, compatibility, implementation status, version, and legacy aliases.
- `FilterRecipe`: serializable configured filter/stack, media roles, seed, output settings, progression/identity settings, and requested/actual backends.
- `FilterExecutionContext`: structured source/destination media and analysis artifacts supplied to a filter strategy.
- `TransformationPlan`: normalized mappings, destination regions, speaker/time relationships, progression values, rejections, warnings, validation, metrics, deterministic seed, and filter version.

Rendering continues to consume the established schedule shape. A filter strategy must first create and validate a normalized plan; an adapter then converts or attaches that plan to the schedule used by the proven renderer.

## Compatibility policy

Legacy identifiers resolve explicitly and emit migration notes:

- `movie_masher` -> `translation.movie_masher`
- `self_shuffle` -> `translation.self_shuffle`
- `echo` -> `translation.echo`
- `drift` -> `translation.drift`

Recipes retain the version they were created with. Loading an older version produces a version-mismatch warning instead of silently claiming equivalence.
