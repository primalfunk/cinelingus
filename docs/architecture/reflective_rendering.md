# Reflective rendering and editorial refinement

Performance-fill renders pass through a bounded editorial feedback loop after ordinary residue correction and rendered-dialogue verification. Verification is evidence for an editing decision rather than a terminal report.

Each enabled mapping has a stable `editorial_placement_id`. The decision engine joins that identity to rendered transcription, missing-word and sentence-boundary evidence, scheduler compatibility, performance, speaker, visual, timing, reuse, residue, confidence, and the initial problem-region report. The quality model publishes every available contributor and its normalized active weight; unavailable evidence is named and omitted instead of silently scored.

Placements below the configured threshold receive standardized failures with severity, confidence, evidence, and a repair strategy. The repair engine searches scheduler-native donor candidates while its run-local memory excludes donor/placement combinations that already failed. It does not add embeddings, semantic inference, or learned behavior.

Accepted repair batches are rendered only across merged affected regions when `editorial_incremental_render` is enabled. Those regions are re-transcribed and merged into the whole-timeline verification report by stable placement identity. A pass is accepted only if average quality does not decrease and its repair and rejection counts do not increase. A regressing pass is re-rendered from the prior schedule. Placements still unresolved after the bounded passes are disabled and their regions are rendered without the rejected donor audio.

The default controls are:

- `editorial_refinement_enabled`: `true`
- `editorial_max_passes`: `2`
- `editorial_acceptance_threshold`: `0.72`
- `editorial_min_word_coverage`: `0.72`
- `editorial_max_repairs_per_pass`: `24`
- `editorial_incremental_render`: `true`
- `editorial_suppress_unresolved`: `false`

When the pass limit is reached, the default preserves the best-known non-regressing render and marks unresolved placements for review. Explicit suppression is available, but is not the safe default because removing many merely weak placements can be more destructive than retaining them.

The final `editorial_decisions.json` is the placement-level audit. `editorial_report.json` records immediate accepts, successful repairs, terminal rejections, pass-by-pass quality, recurring failures, best repairs, worst unrecoverable moments, and run-local avoidance memory. Both artifacts are schema validated and linked from the run report.

## Phase 0 strategy evidence

`python run_translation.py phase0-strategy-audit` executes one deterministic pre-render contract benchmark for every declared repair strategy. This proves that each route can produce its intended mapping change, or conservative retention, without claiming a successful rendered repair.

Repeat `--calibration-report PATH` for completed calibration reports to produce `phase0_rendered_strategy_coverage.json`. A strategy satisfies the rendered-evidence gate only after a corpus run renders a candidate through that primary strategy. Conservative uncertainty retention instead requires an observed runtime retention, because rendering a speculative candidate would violate its contract.

`python run_translation.py phase0-observed-plan --config CONFIG --prior-report EDITORIAL_REPORT --schedule-artifact SCHEDULE --strategy-coverage COVERAGE` converts failures observed in a full-length run into bounded, provenance-bearing excerpt cases. `--max-strategy-variants N` selects multiple low-quality placements for each still-missing strategy; this is important because a full-run failure may disappear when the placement is reanalysed inside a short excerpt. The generated plan never alters source media and remains an explicit calibration-only route.

The work-order completion gate and the rendered-evidence ledger are intentionally distinct. Phase 0 requires every strategy to have an executable benchmark case. The rendered ledger records the stronger empirical question—whether a real corpus run reached, rendered, and retained each route—and may remain incomplete when the corpus has not exhibited a failure category.

## Interruption recovery

Before an editorial candidate is rendered, the pipeline persists its complete proposed schedule, affected regions, placement decisions, completed-pass history, and run-local avoidance memory in the render checkpoint. If execution stops, the next compatible run restores that exact candidate instead of selecting donors again. Rendering is replayed from the prepared-candidate boundary; incremental region replacement remains atomic, so an interruption cannot promote a partially patched WAV. A resumed result declares `resumed_from_candidate_checkpoint` and publishes the recovery contract in `repair_capabilities.interruption_recovery`.

Checkpoint JSON replacement retries short-lived Windows `PermissionError` contention while retaining the same complete temporary payload. This allows a UI, monitor, or audit process to read the checkpoint without converting a harmless reader race into a failed transformation.

## Run-level acceptance

Quality-corpus acceptance requires a sibling `editorial_report.json` by default. Editorial status and aggregate quality remain useful diagnostics, but they cannot override placement-level delivery gates. Any `mid_word_cut`, `residual_dialogue`, rendered coverage below 25%, or `FAILED_DELIVERY` final state fails its independent corpus check even when the run's average quality is high. Manifests may tune the numeric limits, but missing editorial evidence is itself a failure unless explicitly disabled for a non-editorial corpus.
