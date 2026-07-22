# Phase 1 FilmModel developer commands

These commands are opt-in and do not replace or redirect un_cinelingus.py,
legacy filters, scheduling, rendering, repair, or verification.

## Build

Build from an existing cache-role directory:

`powershell
python -m cinelingus.cli build-film-model 
  --artifact-dir cache/<media-hash>/destination_video 
  --output cache/<media-hash>/destination_video/cinematic_model
`

Or resolve the same directory from canonical cache identity:

`powershell
python -m cinelingus.cli build-film-model 
  --cache-root cache 
  --media-hash <media-hash> 
  --role destination_video
`

Use --include-editorial-run <run-output-dir> to add run-scoped editorial
observations. The full editorial report takes precedence over a redundant
editorial_decisions.json view and only the selected artifact affects the cache
signature.

Additional modes:

- --force rebuilds a compatible cached model.
- --validation-only builds and validates without writing.
- --report-only prints the report without writing.
- --strict rejects VALID_WITH_WARNINGS.
- --deterministic-diagnostic prints selected artifacts and signatures.

The default bundle contains:

- ilm_model.json
- model_report.txt
- alidation_report.json
- migration_report.json
- uild_report.json
- rtifact_manifest.json

## Validate, report, and compare

`powershell
python -m cinelingus.cli validate-film-model <film_model.json>
python -m cinelingus.cli report-film-model <film_model.json> --output <report.txt>
python -m cinelingus.cli compare-film-model <left.json> <right.json>
`

Validation is read-only unless an explicit report output is supplied. Comparison
uses the canonical deterministic representation and returns a non-zero status
when the models differ.

## Trace and reconstruct a Translation schedule

`powershell
python -m cinelingus.cli trace-schedule 
  --schedule <replacement_schedule.json> 
  --source-model <source-film-model.json> 
  --destination-model <destination-film-model.json> 
  --verification <rendered_dialogue_verification.json> 
  --output <schedule_bridge.json>

python -m cinelingus.cli reconstruct-schedule <schedule_bridge.json> 
  --output <reconstructed_schedule.json>

python -m cinelingus.cli compare-schedule 
  <replacement_schedule.json> <reconstructed_schedule.json> 
  --bridge <schedule_bridge.json> 
  --output <schedule_equivalence.json>
`

The bridge stores an exact schedule payload plus model references. Reconstruction
verifies the payload signature before returning it. A behavioral or invalid
difference makes comparison return a non-zero status.
