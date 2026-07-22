# Cinelingus

Cinelingus is a local cinematic transformation laboratory. It analyzes dialogue, performances, speakers, scenes, shots, and safe cinematic moments, then changes the relationships between voice, identity, performance, image, and time while preserving inspectable provenance.

The application is designed for local film material: media and analysis artifacts remain on the workstation unless you deliberately publish them elsewhere.

**Current development baseline: 0.3.0 plus the completed Phase 0-3 research layers**

Release 0.3.0 makes montage planning the shared Cinelanguage segmentation layer for every executable apparatus. Apparatuses select and order complete cinematic moments, retain shot and boundary evidence, declare their visual, temporal, dialogue, and audio laws, and must pass encoded render acceptance before their output is accepted. Stable engineering IDs continue to use the internal `filter` vocabulary for compatibility.

Cinelingus remains intentionally experimental. A completed render demonstrates that its declared contract passed; it does not by itself establish production readiness. The evaluation system reserves `EXPERIMENTAL`, `PREVIEW`, and `PRODUCTION_READY` for evidence-based verdicts.

## Research-program status

| Phase | Delivered | State |
| --- | --- | --- |
| Phase 0 — Repair effectiveness | Rendered verification, candidate and neighborhood repair, atomic rollback, interruption recovery, calibrated hard acceptance gates, evidence-selected benchmarks, and multi-pair corpus execution | Complete, with empirical limitations recorded |
| Phase 1 — Cinematic representation | Versioned `FilmModel` entities, stable provenance, validation and caching, translation bridges, turn coverage, and equivalence proof against the legacy path | Complete |
| Phase 2 — Semantic passage representation | Local E5 passage embeddings, separate semantic bundles, bounded retrieval/reranking, acoustic preflight, word-boundary recovery, schedule screening, rendered verification, repair, and blinded proof tooling | Complete; semantic influence remains optional and disabled by default |
| Phase 3 — Dialogue function | A versioned three-axis taxonomy, deterministic local classifier, function bundles, turn/sequence aggregation, four scheduling modes, acoustic preflight, rendered reclassification, repair, and four-way render proof | Engineering complete; human calibration and subjective review are explicitly deferred |

The latest authoritative closeout documents are the [Phase 0 completion report](docs/review_artifacts/phase0_completion_report_20260721.md), [Phase 2 end-of-phase report](docs/review_artifacts/phase2_end_of_phase_report_20260722.md), and [Phase 3 final report](docs/review_artifacts/phase3_final_report_20260722.md). Phase completion means that the contracted machinery and evidence gates exist and pass regression testing. It does not mean that every long-form render is acceptable or that experimental semantic and dialogue-function judgments have been validated by people.

## Phase 4 — start here next session

Phase 4 should begin with **visual performance and active-speaker evidence**, not with a larger semantic or dialogue-function taxonomy. The immediate product problem is now sharply defined: determine whether visible people appear to be speaking during each destination speech window, which visible participant is most likely active, and whether the picture instead presents listening, reaction, action, occlusion, a wide shot, or an off-screen speaker. This directly addresses substitutions that put one voice across a visible two-person exchange, speech over characters who are plainly not talking, and silence over visible speech.

Start the next session in this order:

1. Read the [Phase 3 final report](docs/review_artifacts/phase3_final_report_20260722.md), especially its limitations and Phase 4 recommendation. Preserve the Phase 3 default: dialogue-function scheduling remains disabled unless explicitly selected.
2. Establish the baseline with `python -m pytest -q`. The closeout baseline is **635 passing tests**.
3. Audit and reuse [visual.py](src/cinelingus/visual.py), [visual_performance.py](src/cinelingus/visual_performance.py), [speakers.py](src/cinelingus/speakers.py), and the [cinematic model](src/cinelingus/cinematic_model/) before adding a parallel representation.
4. Define versioned, provenance-bearing evidence for per-face mouth activity, active-speaker attribution, face visibility/occlusion, shot scale, listener/reaction status, action conflict, and cut continuity. Every field must support confidence and `unavailable`; abstention is preferable to invented certainty.
5. Implement the evidence in report-only mode first, measure coverage and false positives across animation and live action, then add bounded scheduling influence. Only high-confidence visual contradictions should become hard gates.
6. Re-run the existing semantic/function four-way proof and a multi-pair corpus screen to demonstrate that Phase 4 improves visible speech alignment without weakening timing, speaker, acoustic, provenance, residue, or rollback guarantees.

Do not begin Phase 4 by claiming emotion, character identity, relationships, gaze intention, or narrative meaning. Human Phase 3 calibration remains a useful future activity, but it is not a blocker for this engineering sequence. The retained comparison package is [review_manifest.json](evaluation/phase3_render_proof_mega_man_to_excerpt_provisional_20260722/deferred_blinded_review/review_manifest.json); its absence of completed review must remain visible in future reports.

## Current capabilities

- Scalable Qt Instrument Interface built from a transparent faceplate, clipped under-plate instrumentation, and a toolkit-independent production controller
- Full Source Timeline execution for every implemented apparatus
- Glimpse (Fast Preview), Study (Balanced), and Divination (High Accuracy) scrutiny modes
- Whisper transcription and Pyannote-aware speaker diarization
- Item-centric speaker alignment with direct, ambiguous, continuity-inferred, and heuristic provenance
- Versioned cinematic models that preserve films, scenes, shots, speech passages, dialogue turns, performances, evidence, confidence, and provenance as separate entities
- Duration-, rhythm-, performance-, speaker-, scene-, and shot-aware dialogue placement
- Candidate-level and coordinated-neighborhood editorial repair with rendered verification, atomic commit or rollback, checkpointed interruption recovery, and explicit hard-failure gates
- Optional local semantic-passage retrieval using `intfloat/multilingual-e5-small`, with independent artifacts, acoustic preflight, rendered-word verification, and safe rollback
- Optional dialogue-function analysis across surface form, interaction function, and sequence position, with deterministic abstaining classification and separate report-only, assisted, and preserving modes
- Universal montage plans built from conservative cinematic moments rather than arbitrary duration cuts
- Complete-shot preservation and evidence-gated subdivision of long takes
- Seeded opening selection without implicit timestamp-zero or first-scene privilege
- Beginning, Development, Climax, and Resolution structural roles
- Audio-qualified cinematic moments and continuous source-soundtrack beds with rejection of sustained dead air over 0.75 seconds
- Performance-first dialogue replacement with hard suppression inside accepted speech regions; legacy ducking remains an explicit experimental mode
- Persistent, role-aware analysis caching for every film in a world
- A GUI command that clears pipeline-owned cache artifacts without disturbing source media or output reports
- Versioned apparatus recipes backed by compatible internal filter recipes and plans, montage plans, and render-acceptance artifacts
- Replacement-audio provenance, coverage, interval-level silence diagnostics, stream, source-participation, and contract validation
- Executable multi-input success guarantees with arity-based applicability, pre-render rejection, post-render certification, and an independent all-filter evidence matrix
- Ordered filter-combination compatibility compilation with fail-closed recipe validation and evidence-gated execution
- User-facing alteration guarantees with tolerant defaults, primary-only stack degradation, measurable authored extent, and validated full-film altered fallback
- Deterministic recipes and seeds wherever an operator uses controlled variation
- Cross-process run locking and fresh internal operator-identity receipts
- Scoped publication cleanup that preserves evidence belonging to other runs

Whisper `medium` is the default transcription model. Glimpse and Study may deliberately select lighter models; Divination and the default configuration use `medium`. Missing semantic model assets or unavailable function evidence never silently replace the legacy legal path.

## Instrument interface

The desktop application is one coherent cinematic instrument rather than an effects dashboard. The faceplate scales proportionally inside the window, leaving margins when the window aspect ratio differs instead of stretching the plate. Reality, Discipline, Apparatus, Materials, Scrutiny, Calibration, Actuation, Observation, Procession, Curator, Ledger, and Service are keyboard-focusable painted controls clipped beneath the transparent plate.

The controls share a deliberate material grammar: brass identifies engraved hardware and adjustment, cyan means energized, selected, moving, or successfully complete, ivory identifies ordinary readable values, dim bronze indicates dormant mechanisms, amber means caution, and red means failure. Rotary selectors use calibrated detents and recessed readouts; progress uses continuous instrument meters; the invocation control has distinct ready, engaged, complete, and fault states. Reality selects One Film or Several Films, Discipline selects the kind of governing law, Apparatus selects the law itself, and Calibration tunes candidate preference without changing that law.

Run `python -m cinelingus.instrument_ui` to open the isolated component-state sheet used for control-library and scaling review.

Run `python run_cinelingus.py` or `cinelingus-gui` for the production Qt interface; it opens maximized unless `--windowed` is supplied. Permanent engraving is baked into `assets/instrument_faceplate_overlay.png`; values, state, light, and motion render beneath transparent shaped apertures. Configuration, film admission, apparatus parameters, service diagnostics, safe cancellation, and archive access are native Qt dialogs outside the plate surface. Generate deterministic review images with, for example, `python -m cinelingus.qt_faceplate --screenshot output/qt-faceplate.png --state running --scale 1.25`. The retired Tk shell is an explicit compatibility command: `cinelingus-gui-legacy` or `python run_cinelingus.py --legacy-tk`.

The former root-level `main_plate.png` supplied the initial instrument language and has been retired from the runtime asset set. The replaceable production chassis is `assets/instrument_plate.png`; its apertures and labels are defined by `assets/instrument_apertures.json`, while responsive Qt painting lives in `src/cinelingus/qt_faceplate.py`. Emblem variants now live under `assets/`. The visual and interaction contract is documented in `docs/architecture/instrument_material_system.md`.

The central Observation display is intentionally concise: it shows the phase, current operation, overall and stage progress, elapsed time, estimated remaining time, and estimated completion. A slow heartbeat lamp indicates continued activity. Curator is permanent and indexes Most Convincing, Beautiful Accident, Unstable, Rare Alignment, Worth Revisiting, and Needs Attention observations after a render. Logs, reports, advanced settings, timing, and diagnostic detail remain in the expandable Laboratory Notes section.

Transposition transfers spoken performances between two films. Film A is consumed continuously from timestamp zero and Film B supplies donor dialogue. The result uses the complete supported timeline: if Film B's required audio ends before Film A's video, the output video is curtailed to that supporting-audio boundary. Its stable internal ID remains `multiworld.translation`.

All executable apparatuses share measurable pre-render and final-output gates. An invocation must provide sufficient dialogue coverage and timeline distribution, remain within its source-reuse rules, retain valid provenance, complete through the montage-native planner, and produce an accepted audio-bearing MP4. Sparse, effectively unchanged, visually unsafe, silent, or wrong-apparatus schedules fail rather than masquerading as successful output.

## Montage foundation

The public unit of composition is a **cinematic moment**: one or more consecutive shots expressing a coherent visual idea. Shots remain internal evidence. Selected moments record their film, media hash, scene, shot IDs, visual boundaries, audio boundaries, structural role, capability assertions, and selection basis.

Core analysis uses literal local evidence such as cuts, transitions, speech timing, optical motion, camera stability, stillness, and duration. It does not claim to understand gesture, gaze, intention, or dramatic meaning. A long take may still be divided internally when analysis needs evidence-bearing dialogue or shot regions, but those regions no longer select a shortened input or output. Runtime rendering consumes the anchor video continuously from timestamp zero.

Every successful apparatus authors four laws over a validated montage plan:

- **Visual Law** - imagery that may be selected
- **Temporal Law** - chronology and ordering
- **Dialogue Law** - treatment of spoken material
- **Audio Law** - treatment of production sound, ambience, music, and effects

For filters that do not require chronology, the earliest eligible moment receives no automatic preference. Opening selection uses structural role, filter relationship, cinematic integrity, and seeded diversity. Repeated source-intro selection is measured as planner bias.

The destination soundtrack remains intact outside accepted replacement regions. Translation hard-suppresses the detected destination speech span, applies padded adaptive crossfades, and overlays the accepted donor turns; unmatched or failed performances remain untouched. The former -28 dB treatment is available only as explicit `duck` mode. Source-authored quiet remains valid; acceptance rejects unexplained render gaps and outputs that are wholly or effectively silent. There is no requested-runtime or short-reel mode. A one-film filter renders the complete anchor duration. A filter that requires supporting audio renders up to the shortest required supporting-audio duration, so FFmpeg curtails a longer anchor video instead of padding or abandoning the support track. Repetition is used only when the active filter contract or an explicit filter-plan parameter authorizes it, and the plan records that authorization and any repeated source placements.

Audio qualification is hierarchical. Whole-moment soundtrack measurements guide ranking but do not veto unrelated dialogue. Authored placements are evaluated with local guard bands and existing complete-shot boundaries; unsafe parent moments can therefore yield provenance-bearing placement submoments through a deterministic rescue path. The strict 0.75-second dead-air limit remains an encoded-output acceptance rule. Same-film filters reuse one canonical transcription, diarization, speaker attribution, and performance analysis rather than analyzing the same media twice. See docs/montage_IV_audio_qualification_repair.txt.

## Available apparatuses

The public catalog separates operating mode from discipline. **One Film** and **Several Films** describe how many realities participate. The six disciplines describe what kind of cinematic law governs them; Multiworld is not a seventh discipline.

Currently invokable with One Film:

- Chronomancy Engine: Drift, Premonition, Echoes, M&ouml;bius, Spiral, Ouroboros
- Contagion Laboratory: Contagion, Mutation, Whisper, Dialect
- Memory Palace: Amnesia, Dream, Recollection
- Mask Workshop: Possession, Doppelg&auml;nger, Chorus, Split Personality
- Alchemical Engine: Echo, Self Shuffle, Bloom, Shed Skin, Venom, Exhaustion
- Lexicon: Wonder, Regret, Optimist, Paranoia

Currently invokable with Several Films:

- Chronomancy Engine: Prophecy
- Contagion Laboratory: Contagion
- Mask Workshop: Possession
- Alchemical Engine: Transposition, Echo Chamber

Lexicon apparatuses disclose their lexical selection proxies rather than claiming emotional inference. Dream discloses token overlap and temporal distance rather than semantic understanding. Dialect changes cadence selection and does not claim accent conversion. Whisper and Exhaustion emit bounded gain and EQ controls that the renderer applies per placement.

Every implemented apparatus declares Full Length as its only output form. Multiworld apparatuses preserve Film A picture and chronology while recording source-film identity and media hashes for every dialogue layer; output ends at the shortest required supporting-audio boundary when it precedes the end of Film A.

Doppelg&auml;nger, Mirror World, Bleed, Parallel Universes, Wormhole, Chimera, and Civilization are dormant Multiworld apparatuses. Triangle is executable as a three-film A-B-C visual cycle with a closed B-C-A dialogue exchange. It preserves each carrier film's non-speech soundtrack, hard-suppresses every detected carrier-speech interval, and never restores unmatched carrier dialogue when unique donor material is exhausted. Dormant apparatuses are absent from ordinary invocation selection, expose no execution action, and retain honest capability requirements in the public catalog. See [Public apparatus catalog](docs/architecture/public_apparatus_catalog.md).

## Speaker evidence

Speaker analysis keeps three independent truths:

- `diarization_status` records whether direct vocal-identity analysis succeeded.
- `alignment_status` records how completely those turns align with transcript items.
- `fallback_status` records whether continuity or heuristic inference was required.

A single diarization turn may support several transcript items. Each item records its assignment method, confidence, overlap, supporting segment IDs, ambiguity, and whether inference was used. Successful Pyannote analysis is not relabeled as a backend failure merely because Whisper divided the same turn into several passages.

Identity-dependent planning requires sufficient direct speaker evidence. Inferred or unknown identities remain available to non-identity operations only where the active filter contract permits them.

## Performance, semantics, and dialogue function

`SpeechPassage` is the dependable semantic and functional unit. `DialogueTurn` and ordered performance sequences are richer structures used only where the underlying FilmModel contains sufficient turn evidence; they are not synthesized from missing data. At Phase 3 closeout, passage-level function analysis covered 3,946 passages, but only four passages had normalized DialogueTurn evidence. Seven of eight audited FilmModels therefore had no usable ordered turns. This limitation is one reason visual active-speaker evidence is the next priority.

Semantic and dialogue-function evidence are independent, additive signals. Neither can override the established timing, duration, speaker, acoustic, visual-safety, reuse, or provenance gates. Both have explicit disabled and report-only states so their diagnostics can be generated without changing a schedule. Function scheduling additionally provides assisted and preserving modes; all influencing modes remain experimental opt-ins.

The selected Phase 2 semantic provider is the pinned local `intfloat/multilingual-e5-small` model. It creates 384-dimensional normalized vectors and does not download assets during a production render. Phase 2's final proof found semantic assistance only lightly preferable, confirming that cosine similarity is useful as a bounded distinction rather than a claim of meaning or an autonomous editorial authority.

Phase 3 adds 39 labels across three non-interchangeable axes: seven surface-form states, 26 interaction-function states, and six sequence-position states. Its rule classifier is local, deterministic, inspectable, dependency-free, ambiguity-aware, and able to abstain. It classified 3,946 passages in 1.622 seconds with 194 abstentions, but those outputs have not been compared with human ground truth. Generic declaratives, clipped utterances, request-versus-command distinctions, and animation dialogue remain known weak areas.

The final Phase 3 proof replaced one acoustically unsafe proposed donor with an already-legal, preflight-accepted donor and verified the actual rendered words afterward. Average rendered word coverage improved from **68.65% to 87.30%**, failed rendered mappings fell from one to zero, and no destination-voice residue was detected. The proof establishes the repair path, not universal classifier quality: only one changed placement survived, two earlier non-target function mismatches remain, and the prepared blinded review was not completed.

The invariant order is:

1. establish legacy legality and hard compatibility;
2. use optional semantic and function evidence only to rank or diagnose legal alternatives;
3. reject acoustically unsafe candidates before expensive rendering;
4. transcribe and re-evaluate the words that actually rendered;
5. commit only measured improvements, otherwise restore the prior candidate atomically.

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
python run_cinelingus.py
```

Equivalent installed entry point:

```powershell
cinelingus-gui
```

Both commands start the production Qt interface. Tkinter is retained only behind the explicit `--legacy-tk` / `cinelingus-gui-legacy` compatibility path.

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
- `src/cinelingus/cinematic_model/` - versioned cinematic entities, confidence, provenance, validation, lookup, caching, and schedule bridges
- `src/cinelingus/semantic/` - independent semantic bundles, retrieval experiments, preflight, repair, rendered proof, and review tooling
- `src/cinelingus/dialogue_function/` - taxonomy, classifier, bundles, scheduling, calibration, acoustic preflight, repair, and rendered verification
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
- `editorial_decisions.json`
- `editorial_report.json`
- `transformation_report.json` or the applicable mutation report
- `semantic_bundle.json` and `dialogue_function_bundle.json` when those optional analysis layers are built
- semantic/function schedule screens, acoustic preflight reports, rendered-verification reports, repair decisions, and proof manifests for experimental comparisons

Do not commit copyrighted films, extracted clips, model weights, access tokens, machine-local resolver paths, caches, or generated outputs. The supplied `.gitignore` excludes these while keeping documentation, schemas, contracts, presets, tests, and intentional visual assets trackable.

## Development

Run the complete regression suite:

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
python -m pytest -q
```

The CUDA override keeps routine tests CPU-only. Remove it only when deliberately validating GPU behavior. Media acceptance requires more than a passing unit suite: inspect the final MP4, `filter_acceptance.json`, `montage_render_acceptance.json`, audio provenance, silence measurements, and source/shot provenance.

Useful research-layer entry commands include:

```powershell
cinelingus validate-function-taxonomy
cinelingus build-semantic-bundle --help
cinelingus screen-semantic-schedules --help
cinelingus build-function-bundle --help
cinelingus screen-function-schedules --help
cinelingus render-function-proof --help
```

These commands create separate evidence artifacts; invoking a builder or screen does not enable semantic or function influence in normal production scheduling. Prefer a report-only corpus screen before any assisted experiment, and always preserve the disabled control for equivalence comparison.

## Versioning

Cinelingus follows Semantic Versioning while remaining pre-1.0:

- `0.1.0` - original local dialogue-replacement pipeline
- `0.2.0` - contract-driven Filter Laboratory, operator catalog, and Multiworld foundation
- `0.3.0` - universal montage-native execution, safe cinematic moments, continuous-audio acceptance, destination-intro non-privilege, evidence-aware speaker alignment, and strengthened GUI/cache/run safeguards

The Phase 0-3 work currently sits on top of the 0.3.0 public release baseline. Phase numbers describe the research and implementation program; they are not package versions. A future release should be cut only after its intended public defaults, artifact compatibility, and migration notes are settled.

Minor releases may add or materially refine filters, contracts, artifacts, planners, and operator workflows. Patch releases should correct behavior without introducing a new public capability layer. A `1.0.0` release should follow only when recipe and contract formats, supported operator behavior, installation, calibration evidence, and production-readiness criteria are considered stable.

## Design and extension references

- [Contract-driven pipeline](docs/architecture/contract_driven_pipeline.md)
- [Multi-input success guarantee](docs/architecture/multi_input_success_guarantee.md)
- [Ordered filter-combination compatibility](docs/architecture/filter_combination_compatibility.md)
- [Configuration outcome guarantee](docs/architecture/configuration_outcome_guarantee.md)
- [Reflective rendering and editorial refinement](docs/architecture/reflective_rendering.md)
- [Performance-driven scheduling](docs/architecture/performance_driven_scheduling.md)
- [Phase 1 completion report](docs/architecture/phase1_completion_report.md)
- [Phase 1 developer commands](docs/architecture/phase1_developer_commands.md)
- [Phase 2 end-of-phase report](docs/review_artifacts/phase2_end_of_phase_report_20260722.md)
- [Phase 3 implementation checkpoint](docs/review_artifacts/phase3_implementation_checkpoint_20260722.md)
- [Phase 3 final report and Phase 4 recommendation](docs/review_artifacts/phase3_final_report_20260722.md)
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
