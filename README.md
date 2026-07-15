# Cinelingus

Cinelingus is a local cinematic transformation laboratory. It analyzes dialogue, performances, speakers, scenes, and shots, then changes the relationships between voice, identity, performance, and time while preserving inspectable provenance.

The application is designed for local film material: media and analysis artifacts remain on the workstation unless you deliberately publish them elsewhere.

**Current development release: 0.2.0**

This release marks the expansion from the original two-film dialogue replacement pipeline into a contract-driven Filter Laboratory with single-film and Multiworld cinematic operators.

## Current capabilities

- Guided desktop Filter Laboratory with plain-language operator controls
- Preview, Best Short Remix, and Full Movie output forms
- Whisper transcription and Pyannote-aware speaker diarization
- Duration-, rhythm-, performance-, and speaker-aware dialogue placement
- Persistent, role-aware analysis caching for every film in a world
- Versioned `filter_recipe.json` and normalized `filter_plan.json` artifacts
- Replacement-audio provenance, coverage, silence, stream, and contract validation
- Diagnostic reports for diarization, performance matching, scheduling, and renders
- Deterministic recipes and seeds where a filter uses controlled variation
- Cross-process run locking so two experiments cannot publish into the same output tree
- Fresh filter-identity receipts that prevent a completed run from being reported under the wrong operator
- Scoped publication cleanup that preserves audit logs and artifacts belonging to other runs

Movie Masher established the Multiworld foundation by applying the Dialogue Translation law to two films. Film A is the anchor and supplies the timeline; Film B supplies donor dialogue. The same staged runtime now also supports variable film counts and provenance-bearing dialogue laws.

Full-length Movie Masher and the newer dialogue laws now share measurable pre-render and final-output gates. A run must provide sufficient dialogue coverage and timeline distribution, remain within its source-reuse rules, retain valid provenance, and produce an accepted audio-bearing MP4. Sparse or effectively unchanged schedules fail before an expensive full render rather than masquerading as successful output.

## Available operators

The Laboratory groups operators into Translation, Infection, Identity, Memory, Emotion, Time, Experimental, and Multiworld families.

Currently executable:

- Translation: Self Shuffle, Echo, Drift
- Infection: Contagion, Whisper, Mutation, Dialect
- Identity: Possession, Doppelgänger, Chorus, Split Personality
- Memory: Dream, Recollection, Amnesia
- Emotion: Wonder, Regret, Optimist, Paranoia, Exhaustion
- Time: Flashback, Foreshadow, Spiral, Möbius
- Experimental: Bloom, Venom, Shed Skin, Ouroboros
- Multiworld: Movie Masher, Possession, Contagion, Echo Chamber, Prophecy

Emotion, Dream, and Venom deliberately use disclosed lexical proxies rather than claiming unavailable semantic or emotional embeddings. Whisper and Exhaustion emit bounded gain/EQ controls that the renderer applies per placement.

Multiworld Possession, Contagion, Echo Chamber, and Prophecy currently support Full Movie output. They preserve Film A picture and chronology while emitting the source film ID and media hash for every dialogue layer. Best Short remains disabled until reel selection can prove that every required film, infection phase, echo layer, or prophecy relationship survives shortening.

The remaining Multiworld placeholders are Doppelgänger, Mirror World, Bleed, Parallel Universes, Wormhole, Chimera, Triangle, and Civilization. They remain visible with the explicit message **This filter is not yet implemented.** Their contracts require cross-film picture, shot, music, soundscape, or genre composition beyond the current dialogue renderer. Bloom can be added as a progression modifier; this release otherwise runs one primary operator at a time.

## Requirements

- Windows
- Python 3.11 or newer
- FFmpeg and FFprobe available on `PATH`, or the project-provided shared FFmpeg build
- Sufficient local storage for decoded audio, analysis caches, and rendered video
- A Hugging Face access token when the selected Pyannote model requires one

GPU acceleration is optional. Speaker analysis can run on CPU, and the diagnostic command should be used before committing to a long diarization run.

## Installation

From PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Place working films in `source/` or select them from the GUI. Source media, caches, temporary audio, and rendered outputs are intentionally excluded from version control.

## Launch

Start the desktop laboratory:

```powershell
movie-masher-gui
```

Equivalent module form:

```powershell
python -m movie_masher.gui
```

The command-line interface is also available:

```powershell
movie-masher --help
movie-masher --config config/default.json run
```

Run a saved preset:

```powershell
movie-masher --config config/default.json preset movie_masher
```

Configuration defaults live in `config/default.json`. Keep machine-specific paths or credentials in an ignored local configuration file rather than committing them.

## Typical workflow

1. Choose Film A, the anchor, and exactly as many additional films as the selected contract requests.
2. Select an operator and output form.
3. Review the operator description and its relevant controls.
4. Run the experiment. Existing compatible analysis is reused automatically.
5. Inspect the finished media together with its reports in `output/`.

The progress journal distinguishes analysis, matching, arrangement, reconstruction, and final validation. A completed render is not considered successful solely because an MP4 exists: accepted operators must also pass their declared output contract.

Only one active experiment may use a given output directory. Successful GUI filter runs write a receipt beneath `output/run_receipts/` recording the requested filter, the filter identified by fresh recipe or acceptance evidence, and the published artifact. A mismatch is treated as a failed run. Technical timeout configuration is retained in the diagnostic record without being misreported as an actual timeout event.

## Artifacts and repository boundaries

Runtime material is separated from source code:

- `cache/` - reusable media analysis and extracted dialogue
- `temp/` - intermediate render and diagnostic files
- `output/` - final media and acceptance/provenance reports
- `output/run_receipts/` - per-run requested-versus-executed filter identity records
- `filter_contracts/` - machine-valid operator laws
- `schemas/` - artifact and contract schemas
- `presets/filter_recipes/` - shareable operator recipes
- `docs/` - architecture and extension documentation

Do not commit copyrighted source films, extracted clips, model weights, access tokens, or generated outputs. The supplied `.gitignore` excludes these while keeping documentation, schemas, contracts, presets, tests, and intentional visual assets trackable.

## Development

Run the complete regression suite:

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
python -m pytest -q
```

The CUDA override keeps routine tests CPU-only. Remove it only when deliberately validating GPU behavior.

## Versioning

Cinelingus follows Semantic Versioning. The repository began at `0.1.0`; this architecture and operator expansion advances it to `0.2.0`. Minor releases may add or refine filters, contracts, artifacts, and operator workflows while the project remains pre-1.0. A `1.0.0` release should follow once the public recipe/contract formats, supported operator behavior, and installation workflow are considered stable.

Useful references:

- [Filter architecture inventory](docs/filter_architecture_inventory.md)
- [Filter contract catalog](docs/filter_contract_catalog.md)
- [Adding a filter](docs/adding_filters.md)

New operators should be introduced through the registry, machine-valid contract, strategy, recipe, acceptance invariants, and regression tests. Repetition, destructive timing changes, or other defining behavior belongs to the explicit operator contract rather than the shared base scheduler.
