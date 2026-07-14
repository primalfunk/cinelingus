# Cinelingus

Cinelingus is a local cinematic transformation laboratory. It analyzes dialogue, performances, speakers, scenes, and shots, then changes the relationships between voice, identity, performance, and time while preserving inspectable provenance.

The application is designed for local film material: media and analysis artifacts remain on the workstation unless you deliberately publish them elsewhere.

## Current capabilities

- Guided desktop Filter Laboratory with plain-language operator controls
- Preview, Best Short Remix, and Full Movie output forms
- Whisper transcription and Pyannote-aware speaker diarization
- Duration-, rhythm-, performance-, and speaker-aware dialogue placement
- Persistent, role-aware analysis caching for source and destination media
- Versioned `filter_recipe.json` and normalized `filter_plan.json` artifacts
- Replacement-audio provenance, coverage, silence, stream, and contract validation
- Diagnostic reports for diarization, performance matching, scheduling, and renders
- Deterministic recipes and seeds where a filter uses controlled variation

Transposition, the default two-film translation operation, assigns each source dialogue clip at most once. When no suitable unused phrase remains, Cinelingus leaves the unmatched destination interval unfilled instead of recycling distracting audio. Operators that intentionally repeat material, such as Echo, own that behavior explicitly.

## Available operators

The Laboratory groups operators into Translation, Infection, Identity, Memory, Emotion, Time, and Experimental families.

Currently executable:

- Translation: Transposition, Self Shuffle, Echo, Drift
- Infection: Contagion
- Identity: Possession, Doppelganger, Chorus
- Time: Flashback, Foreshadow, Spiral
- Experimental: Bloom

Planned operators remain visible as **In Development** and cannot be selected accidentally. Bloom can be added as a progression modifier; this release otherwise runs one primary operator at a time.

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

1. Choose the destination film and, when required, the source dialogue film.
2. Select an operator and output form.
3. Review the operator description and its relevant controls.
4. Run the experiment. Existing compatible analysis is reused automatically.
5. Inspect the finished media together with its reports in `output/`.

The progress journal distinguishes analysis, matching, arrangement, reconstruction, and final validation. A completed render is not considered successful solely because an MP4 exists: accepted operators must also pass their declared output contract.

## Artifacts and repository boundaries

Runtime material is separated from source code:

- `cache/` - reusable media analysis and extracted dialogue
- `temp/` - intermediate render and diagnostic files
- `output/` - final media and acceptance/provenance reports
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

Useful references:

- [Filter architecture inventory](docs/filter_architecture_inventory.md)
- [Filter contract catalog](docs/filter_contract_catalog.md)
- [Adding a filter](docs/adding_filters.md)

New operators should be introduced through the registry, machine-valid contract, strategy, recipe, acceptance invariants, and regression tests. Repetition, destructive timing changes, or other defining behavior belongs to the explicit operator contract rather than the shared base scheduler.
