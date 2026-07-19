# Cinelingus

Cinelingus is a local cinematic transformation laboratory. It analyzes dialogue, performances, speakers, scenes, shots, and safe cinematic moments, then changes the relationships between voice, identity, performance, image, and time while preserving inspectable provenance.

The application is designed for local film material: media and analysis artifacts remain on the workstation unless you deliberately publish them elsewhere.

**Current development release: 0.3.0**

Release 0.3.0 makes montage planning the shared Cinelanguage segmentation layer for every executable filter. Filters select and order complete cinematic moments, retain shot and boundary evidence, declare their visual, temporal, dialogue, and audio laws, and must pass encoded render acceptance before their output is accepted.

Cinelingus remains intentionally experimental. A completed render demonstrates that its declared contract passed; it does not by itself establish production readiness. The evaluation system reserves `EXPERIMENTAL`, `PREVIEW`, and `PRODUCTION_READY` for evidence-based verdicts.

## Current capabilities

- Scalable desktop Instrument Interface built from a fixed faceplate and live native controls
- Full Source Timeline execution for every implemented filter
- Preview, Balanced, and Precision analysis modes
- Whisper transcription and Pyannote-aware speaker diarization
- Item-centric speaker alignment with direct, ambiguous, continuity-inferred, and heuristic provenance
- Duration-, rhythm-, performance-, speaker-, scene-, and shot-aware dialogue placement
- Universal montage plans built from conservative cinematic moments rather than arbitrary duration cuts
- Complete-shot preservation and evidence-gated subdivision of long takes
- Seeded opening selection without implicit timestamp-zero or first-scene privilege
- Beginning, Development, Climax, and Resolution structural roles
- Audio-qualified cinematic moments and continuous source-soundtrack beds with rejection of sustained dead air over 0.75 seconds
- Activity-aware dialogue ducking that preserves the source bed through replacement-clip padding and pauses
- Persistent, role-aware analysis caching for every film in a world
- A GUI command that clears pipeline-owned cache artifacts without disturbing source media or output reports
- Versioned filter recipes, normalized filter plans, montage plans, and render-acceptance artifacts
- Replacement-audio provenance, coverage, interval-level silence diagnostics, stream, source-participation, and contract validation
- Executable multi-input success guarantees with arity-based applicability, pre-render rejection, post-render certification, and an independent all-filter evidence matrix
- Ordered filter-combination compatibility compilation with fail-closed recipe validation and evidence-gated execution
- User-facing alteration guarantees with tolerant defaults, primary-only stack degradation, measurable authored extent, and validated full-film altered fallback
- Deterministic recipes and seeds wherever an operator uses controlled variation
- Cross-process run locking and fresh filter-identity receipts
- Scoped publication cleanup that preserves evidence belonging to other runs

## Instrument interface

The desktop application is now one coherent cinematic laboratory instrument rather than a dashboard or wizard. The faceplate scales proportionally inside the window, leaving margins when the window aspect ratio differs instead of stretching the plate. Transformation, Material, Quality, Filter, Activate, Observation, progress channels, stage lamps, Curator, and Laboratory Notes are live keyboard-focusable controls aligned over the plate.

The controls share a deliberate material grammar: brass identifies engraved hardware and adjustment, cyan is emitted activity and live information, warm white identifies selected material, dim bronze indicates dormant mechanisms, and grey is reserved for genuinely unavailable controls. Rotary selectors use calibrated detents and recessed readouts; progress uses continuous instrument meters; the activation control has distinct ready and engaged states. Transformation separates its `FIELD` from its experiment dial, while Filter separates its matching profile from `BIAS` so coincident values remain unambiguous.

`main_plate.png` remains the design reference supplied for the instrument language. It is deliberately not loaded as the production interface. The replaceable runtime chassis is `assets/instrument_plate.png`; its empty recesses contain no embedded labels or functionality. Overlay geometry and responsive plate fitting live in `src/cinelingus/instrument_ui.py`, so the plate can be replaced later without rebuilding control behavior. The visual and interaction contract is documented in `docs/architecture/instrument_material_system.md`.

The central Observation display is intentionally concise: it shows the phase, current operation, overall and stage progress, elapsed time, estimated remaining time, and estimated completion. A slow heartbeat lamp indicates continued activity. Curator is permanent and indexes Most Convincing, Beautiful Accident, Unstable, Rare Alignment, Worth Revisiting, and Needs Attention observations after a render. Logs, reports, advanced settings, timing, and diagnostic detail remain in the expandable Laboratory Notes section.

Translation transfers spoken performances between two films. Film A is consumed continuously from timestamp zero and Film B supplies donor dialogue. The result uses the complete supported timeline: if Film B's required audio ends before Film A's video, the output video is curtailed to that supporting-audio boundary.

All executable filters share measurable pre-render and final-output gates. A run must provide sufficient dialogue coverage and timeline distribution, remain within its source-reuse rules, retain valid provenance, complete through the montage-native planner, and produce an accepted audio-bearing MP4. Sparse, effectively unchanged, visually unsafe, silent, or wrong-filter schedules fail rather than masquerading as successful output.

## Montage foundation

The public unit of composition is a **cinematic moment**: one or more consecutive shots expressing a coherent visual idea. Shots remain internal evidence. Selected moments record their film, media hash, scene, shot IDs, visual boundaries, audio boundaries, structural role, capability assertions, and selection basis.

Core analysis uses literal local evidence such as cuts, transitions, speech timing, optical motion, camera stability, stillness, and duration. It does not claim to understand gesture, gaze, intention, or dramatic meaning. A long take may still be divided internally when analysis needs evidence-bearing dialogue or shot regions, but those regions no longer select a shortened input or output. Runtime rendering consumes the anchor video continuously from timestamp zero.

Every successful filter authors four laws over a validated montage plan:

- **Visual Law** - imagery that may be selected
- **Temporal Law** - chronology and ordering
- **Dialogue Law** - treatment of spoken material
- **Audio Law** - treatment of production sound, ambience, music, and effects

For filters that do not require chronology, the earliest eligible moment receives no automatic preference. Opening selection uses structural role, filter relationship, cinematic integrity, and seeded diversity. Repeated source-intro selection is measured as planner bias.

The complete source soundtrack is the continuous base layer. Replacement dialogue ducks that layer only across the authored interval. Source-authored quiet remains valid; acceptance rejects unexplained render gaps and outputs that are wholly or effectively silent. There is no requested-runtime or short-reel mode. A one-film filter renders the complete anchor duration. A filter that requires supporting audio renders up to the shortest required supporting-audio duration, so FFmpeg curtails a longer anchor video instead of padding or abandoning the support track. Repetition is used only when the active filter contract or an explicit filter-plan parameter authorizes it, and the plan records that authorization and any repeated source placements.

Audio qualification is hierarchical. Whole-moment soundtrack measurements guide ranking but do not veto unrelated dialogue. Authored placements are evaluated with local guard bands and existing complete-shot boundaries; unsafe parent moments can therefore yield provenance-bearing placement submoments through a deterministic rescue path. The strict 0.75-second dead-air limit remains an encoded-output acceptance rule. Same-film filters reuse one canonical transcription, diarization, speaker attribution, and performance analysis rather than analyzing the same media twice. See docs/montage_IV_audio_qualification_repair.txt.

## Available operators

The Laboratory groups operators into Translation, Infection, Identity, Memory, Emotion, Time, Experimental, and Multiworld families.

Currently executable:

- Translation: Self Shuffle, Echo, Drift
- Infection: Contagion, Whisper, Mutation, Dialect
- Identity: Possession, Doppelg&auml;nger, Chorus, Split Personality
- Memory: Dream, Recollection, Amnesia
- Emotion: Wonder, Regret, Optimist, Paranoia, Exhaustion
- Time: Flashback, Foreshadow, Spiral, M&ouml;bius
- Experimental: Bloom, Venom, Shed Skin, Ouroboros
- Multiworld: Translation, Possession, Contagion, Echo Chamber, Prophecy

Emotion, Dream, and Venom use disclosed lexical proxies rather than claiming unavailable semantic or emotional embeddings. Whisper and Exhaustion emit bounded gain and EQ controls that the renderer applies per placement.

Every implemented operator declares Full Length as its only output form. The GUI contains no clip-length or workflow selector. Multiworld operators preserve Film A picture and chronology while recording the source film ID and media hash for every dialogue layer; the output ends at the shortest required supporting-audio boundary when that boundary precedes the end of Film A.

The remaining Multiworld placeholders are Doppelg&auml;nger, Mirror World, Bleed, Parallel Universes, Wormhole, Chimera, Triangle, and Civilization. They remain visible with the explicit message **This filter is not yet implemented.** Their contracts require cross-film picture, shot, music, soundscape, or genre composition beyond the present renderer.

## Speaker evidence

Speaker analysis keeps three independent truths:

- `diarization_status` records whether direct vocal-identity analysis succeeded.
- `alignment_status` records how completely those turns align with transcript items.
- `fallback_status` records whether continuity or heuristic inference was required.

A single diarization turn may support several transcript items. Each item records its assignment method, confidence, overlap, supporting segment IDs, ambiguity, and whether inference was used. Successful Pyannote analysis is not relabeled as a backend failure merely because Whisper divided the same turn into several passages.

Identity-dependent planning requires sufficient direct speaker evidence. Inferred or unknown identities remain available to non-identity operations only where the active filter contract permits them.

## Requirements

- Windows
- Python 3.11 or newer
- FFmpeg and FFprobe on `PATH`, or the project-provided shared FFmpeg build
- Sufficient local storage for decoded audio, caches, intermediate media, and rendered video
- A Hugging Face access token when the selected Pyannote model requires one

GPU acceleration is optional. Speaker analysis can run on CPU. Use the diagnostic command before committing to a long diarization run, particularly after changing Torch, CUDA, Pyannote, TorchCodec, or FFmpeg.

## Installation

From PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Place working films in an ignored local media folder such as `inputs/`, or select them directly in the GUI. Source media, model weights, caches, temporary audio, and rendered outputs are intentionally excluded from version control.

## Launch

Start the desktop laboratory:

```powershell
cinelingus-gui
```

Equivalent module form:

```powershell
python -m cinelingus.gui
```

Use the canonical command-line entry point:

```powershell
cinelingus --help
cinelingus --config config/default.json run
```

Run or list saved presets:

```powershell
cinelingus --config config/default.json presets
cinelingus --config config/default.json preset translation
cinelingus --config config/default.json preset self_shuffle --seed 7
```

Configuration defaults live in `config/default.json`. Keep machine-specific paths and credentials in ignored local configuration rather than committing them.

Run a focused diarization diagnostic from the repository checkout:

```powershell
python run_cinelingus.py diagnose-diarization path\to\analysis_audio.wav
```

## Typical workflow

1. Choose Film A, the anchor, and as many additional films as the selected contract requests.
2. Turn the Transformation, Quality, and Filter controls. Cinelingus always consumes the complete source timeline.
3. Review the operator description and its relevant controls.
4. Press **Activate**. Compatible analysis is reused automatically.
5. Inspect the finished MP4 together with its recipe, filter plan, montage plan, provenance, and acceptance reports.

Use **Clear Pipeline Cache** when you intentionally need analysis regenerated. It removes only pipeline-owned cache children and is disabled while an experiment is running. The application window, journal, and technical record scroll when their content exceeds the available display area.

The progress journal distinguishes analysis, matching, arrangement, reconstruction, and validation. It consumes structured operator events for warnings such as partial speaker alignment; an empty field such as `fallback_reason: None` is not treated as a real fallback. The technical record retains backend names, thresholds, counts, and trace details.

Only one active experiment may use a given output directory. Successful GUI filter runs write a receipt beneath `output/run_receipts/`, recording the requested filter, the filter identified by fresh evidence, and the published artifact. A mismatch is a failed run.

## Artifacts and repository boundaries

Runtime material is separated from source code:

- `cache/` - reusable role-aware media analysis and extracted dialogue
- `temp/` - intermediate renders and diagnostic files
- `output/` - final media and acceptance/provenance reports
- `output/run_receipts/` - requested-versus-executed filter identity records
- `filter_contracts/` - machine-valid operator laws
- `schemas/` - artifact, contract, montage, calibration, and evaluation schemas
- `presets/` - canonical and compatibility presets
- `presets/filter_recipes/` - shareable operator recipes
- `config/montage_corpus_resolver.example.json` - portable calibration resolver example
- `docs/` - design authority, architecture, and extension documentation

Important per-run evidence includes:

- `filter_recipe.json`
- `filter_plan.json`
- `montage_plan.json`
- `montage_render_acceptance.json`
- `filter_acceptance.json`
- `audio_provenance.json`
- `transformation_report.json` or the applicable mutation report

Do not commit copyrighted films, extracted clips, model weights, access tokens, machine-local resolver paths, caches, or generated outputs. The supplied `.gitignore` excludes these while keeping documentation, schemas, contracts, presets, tests, and intentional visual assets trackable.

## Development

Run the complete regression suite:

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
python -m pytest -q
```

The CUDA override keeps routine tests CPU-only. Remove it only when deliberately validating GPU behavior. Media acceptance requires more than a passing unit suite: inspect the final MP4, `filter_acceptance.json`, `montage_render_acceptance.json`, audio provenance, silence measurements, and source/shot provenance.

## Versioning

Cinelingus follows Semantic Versioning while remaining pre-1.0:

- `0.1.0` - original local dialogue-replacement pipeline
- `0.2.0` - contract-driven Filter Laboratory, operator catalog, and Multiworld foundation
- `0.3.0` - universal montage-native execution, safe cinematic moments, continuous-audio acceptance, destination-intro non-privilege, evidence-aware speaker alignment, and strengthened GUI/cache/run safeguards

Minor releases may add or materially refine filters, contracts, artifacts, planners, and operator workflows. Patch releases should correct behavior without introducing a new public capability layer. A `1.0.0` release should follow only when recipe and contract formats, supported operator behavior, installation, calibration evidence, and production-readiness criteria are considered stable.

## Design and extension references

- [Contract-driven pipeline](docs/architecture/contract_driven_pipeline.md)
- [Multi-input success guarantee](docs/architecture/multi_input_success_guarantee.md)
- [Ordered filter-combination compatibility](docs/architecture/filter_combination_compatibility.md)
- [Configuration outcome guarantee](docs/architecture/configuration_outcome_guarantee.md)
- [Alteration guarantee](docs/architecture/alteration_guarantee.md)
- [Montage composition foundation (ADR-001)](docs/architecture/adr-001-montage-composition-foundation.md)
- [Montage design decisions](docs/montage_I_design.txt)
- [Montage foundation authorization](docs/montage_II_foundation.txt)
- [Montage implementation addendum](docs/montage_III_addendum.txt)
- [Montage audio qualification repair authorization](docs/montage_IV_audio_qualification_repair.txt)
- [Filter architecture inventory](docs/filter_architecture_inventory.md)
- [Filter contract catalog](docs/filter_contract_catalog.md)
- [Adding a filter](docs/adding_filters.md)

New operators enter through the registry, a machine-valid contract, strategy, recipe, montage laws, acceptance invariants, and regression tests. Repetition, destructive timing changes, chronology, identity requirements, or other defining behavior belongs to the explicit operator contract rather than the shared base scheduler.
