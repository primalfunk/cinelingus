# Adding a Cinelingus Filter

The Filter Laboratory is registry-driven. A new filter should not require edits to the main GUI navigation or renderer.

## 1. Define the creative contract

Add one machine-valid JSON contract beneath filter_contracts/<family>/ and one matching FilterDefinition in movie_masher.filter_lab.registry. The contract is the behavioral authority; registry metadata declares runtime capabilities.

- a stable namespaced ID and family;
- creative and operational descriptions;
- relationship dimensions read, changed, and preserved;
- required media roles and analysis artifacts;
- parameter schemas and defaults;
- output, preview, compatibility, and implementation flags;
- an implementation version and any legacy aliases.

Leave `implemented=False` until execution and validation exist. The Laboratory will show **This filter is not yet implemented.** and prevent it from running.

For a Multiworld filter, also declare `minimum_films`, `maximum_films` (`null` means unbounded), one dominant `cinematic_law`, `anchor_behavior`, affected cinematic elements, quality requirements, deterministic-seed support, and the complete input/output/artifact interface. The GUI generates Film A/B/C rows and its Add Film control from this contract; do not add filter-specific GUI branches.

## 2. Plug into the Multiworld pipeline

Multiworld execution follows one reusable sequence: load films, inspect films, create the shared timeline, construct the world model, apply the cinematic law, generate replacement decisions, review, and render. New behavior belongs only in the cinematic-law applicator. Film A is the anchor unless `anchor_behavior` declares a shared or law-defined timeline.

Dialogue-centric laws use movie_masher.filter_lab.multiworld_strategies and Pipeline.run_multiworld_filter. Every mapping must carry source_film_id, destination_film_id, source_media_hash, and destination_media_hash. Do not advertise Best Short until a reel-level test proves the shortened result retains all contract-required films, phases, and layers.

The normalized plan must expose `inputs`, `outputs`, `affected_artifacts`, and `intermediate_products`. These are Procedure composition boundaries; Procedures themselves are not executed in this release.

## 3. Produce a normalized plan

Implement a deterministic strategy in `movie_masher.filter_lab.strategies`. It receives analyzed clips/windows, duration, normalized parameters, and a seed. It must return a schedule carrying:

- mappings compatible with the established renderer;
- rejected candidates and reasons;
- `filter_metrics`;
- `filter_validation` with explicit invariants;
- a plain-language `filter_summary`;
- representative `preview_regions`.

Decorate the builder with @scheduling_strategy(...). The strategy registers itself; pipeline dispatch reads the definition's execution_mode and needs no filter-specific branch. The integration layer writes filter_recipe.json and filter_plan.json before rendering.

## 4. Validate defining behavior

Tests must prove what makes the filter distinct. At minimum add:

- identical recipe/seed produces identical mappings;
- property tests for the filter's hard invariants;
- honest failure tests for missing or sparse artifacts;
- a registry/UI metadata test;
- a real cached-artifact plan check before spending render time.
- registry/contract parity and schema validation;
- a passing filter_acceptance.json containing MP4, provenance, coverage, silence, audio-stream, and invariant checks.

Do not silently fall back to another filter. If defining behavior cannot be achieved, raise a specific error or report an intentionally subtle result.

## 5. Reports and artifacts

Store filter-specific measurements in `filter_metrics`. Large structured results may also be emitted as named artifacts, as Contagion does with `speaker_graph.json` and `infection_timeline.json`, and Bloom with `bloom_profile.json`.

The shared report layer automatically records family, version, dimensions, recipe, plan, requested/actual backends, validation, and the plain-language summary. Successful rendering alone is not operational success: the output acceptance gate must pass.

## 6. Compatibility

This release supports one primary filter and permits Bloom as the only progression modifier. Declare explicit incompatibilities in the registry. Do not expose arbitrary stacking until a combination has deterministic schedule and render tests.
