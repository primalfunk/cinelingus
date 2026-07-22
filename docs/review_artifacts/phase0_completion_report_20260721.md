# Phase 0 completion report

Status: **Complete with documented empirical limitations.**

Phase 0's reflective foundation is implemented, measured, and regression tested. All written exit gates pass. This closes the foundation phase; it does not certify the current long-form renders as good outputs. The revised acceptance contract correctly rejects both available long-form references.

## Exact requested state

| Area | State | Evidence | Remaining limitation |
| --- | --- | --- | --- |
| Neighborhood repair | Implemented and proven | Neighborhood construction, shared assignment identity, atomic all-member commit, selective isolation of unrelated repairs, and atomic rollback are tested. Fourteen real coordinated candidates were rendered and safely rolled back after regression. | No naturally occurring multi-placement neighborhood has survived corpus verification yet. |
| Performance-group repair | Implemented and proven | The performance strategy produced a measured `+0.0886` repair. A two-placement performance neighborhood commits as 2/2 survivors under deterministic rendered-verification evidence. | Positive multi-member survival has not yet occurred in the real corpus. |
| Candidate-level interruption recovery | Implemented and corpus drilled | A real calibration process was killed after two pass-1 candidates were checkpointed. Resume restored the exact batch and avoidance memory, completed the run, and reproduced `0.7651 -> 0.8537`. | None for the Phase 0 recovery gate. Further crash points remain useful hardening work. |
| Quality-threshold calibration | Calibrated | `0.68`, `0.72`, and `0.75` produced identical decisions on a beneficial repair and a boundary unresolved case. The selected `0.72` threshold is retained at the center of the stable measured interval. Hard failures are now independent gates. | The counterfactual sweep is intentionally small and is not a substitute for later human-preference calibration. |
| Evidence-selected repair benchmarks | Exit gate passed | All 12 declared repair strategies have executable contract benchmarks. Eight have runtime attempts; seven have the appropriate empirical evidence; six rendered candidates; three have surviving repairs. | Five strategies still lack their stronger empirical evidence: word boundary, duration fit, visual intent, local suppression, and reuse pressure. |
| Multi-pair corpus execution | Passed | 45 unique completed case signatures; 40 map back to explicit source plans; 14 films; 22 ordered pairs; 14 informative pairs; all principal animation/live-action directions exercised. At least five materially distinct pairs improved. | The corpus remains opportunistic rather than statistically representative. |

## Repair effectiveness

The extended retained set completed 18/18 attempted cases, with 10 informative cases. It rendered seven candidates, retained three improvements, and accepted no regressive repair. Average quality improved from the retained aggregate by `+0.0157`. Additional targeted evidence produced:

- performance repair: `+0.0886`;
- speaker-role repair: `+0.0028`;
- exchange continuity: `+0.0393`;
- fragment/duration-edge cases: `+0.0483` and `+0.0697`;
- standard animation/visual-role case: `+0.0891`;
- quiet residue/masking case: `+0.0211`.

## Successful and failed neighborhood repairs

The positive deterministic benchmark commits two related placements together only after both improve. Real corpus neighborhoods were less favorable: two performance candidates, six transition candidates, and six masking candidates regressed after rendered verification and were restored. This is correct restraint evidence. Widening one performance donor excerpt removed the initial failure entirely (`0.886` initial quality), so it did not create an artificial repair success.

## Unresolved failures and acceptance

The revised run contract prevents averages from concealing delivery failures. The existing `animation_dense` run contains 52 hard low-coverage placements and is rejected. `live_rapid` lacks an editorial report and is also rejected. Therefore Phase 0 closes with zero accepted long-form reference renders; improving those renders remains product-quality work, not missing foundation infrastructure.

## Regressions discovered and corrected

- Interrupted excerpt extraction could leave a truncated file that looked complete; extraction now validates a partial file and replaces atomically.
- A Windows reader could briefly block atomic checkpoint replacement; JSON persistence now retries transient `PermissionError` contention.
- Run-level quality checks ignored editorial hard gates; editorial evidence and hard-failure counts are now mandatory by default.
- Strategy-isolation failures could be obscured by secondary categories; calibration-only target authority now isolates the intended route without changing production routing.

## Recommended Phase 1 scope

Proceed with the minimal cinematic model core described in the program direction. Preserve Phase 0 artifacts and IDs as migration inputs. Do not broaden Phase 1 to semantic embeddings, character identity, active-speaker recognition, or dialogue-function inference. Continue collecting the five missing empirical strategy routes and a naturally surviving coordinated neighborhood as non-blocking regression coverage.
