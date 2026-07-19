# Ordered filter-combination compatibility

Cinelingus does not infer that two individually working filters can safely be stacked. Compatibility is ordered: `A -> B` and `B -> A` are separate decisions because the successor may read relationship state changed by the predecessor.

The compatibility compiler in `src/cinelingus/filter_lab/combination.py` derives each decision from the registry, both machine-valid filter contracts, the successor's Procedure declaration, and optional passing certification evidence. The generated authority is `filter_combination_compatibility_matrix.json`.

## States

- `CERTIFIED`: both contracts support the order, the successor consumes transformed specimen state, multi-step behavior is validated, and passing combination evidence exists. Only this state is executable.
- `COMPATIBLE_UNPROVEN`: the ordered shape is structurally allowed but transformed-state execution or real certification is absent.
- `REQUIRES_REANALYSIS`: the successor reads a relationship domain changed by the predecessor but does not consume transformed state.
- `INCOMPATIBLE`: the order violates an explicit registry or runtime rule.
- `UNAVAILABLE`: one or both filters are not implemented.

Registry stack validation and recipe creation now invoke this compiler. Any multi-filter recipe whose ordered steps are not all `CERTIFIED` is rejected before analysis or rendering with the pair, state, and reason. A passing evidence file cannot override a single-step contract; the successor must also declare `multi_step_validated` and `receives_transformed_specimen: true`.

## Current boundary

Every current filter contract declares `single_step_validated`. Consequently, no combination is yet certified executable. Primary-filter-to-Bloom pairs are visible as `COMPATIBLE_UNPROVEN`, Bloom-to-primary order is `INCOMPATIBLE`, and overlapping primary-filter pairs generally report `REQUIRES_REANALYSIS`.

This is intentional fail-closed behavior. The first executable pair must arrive with a real Procedure runtime, contract changes, deterministic schedule tests, a full render, and persisted certification evidence.

At the user-facing outcome boundary, a non-certified stack is resolved to its first implemented primary filter or to a validated passthrough. This does not change the matrix state or imply combination certification.

## Audit

Compile and schema-validate every ordered pair with:

```powershell
python tools/audit_filter_combinations.py
```

The matrix signature excludes its creation timestamp, so unchanged registry and contract inputs produce an identical signature. The audit covers implemented and unavailable filters, making future activation visible as a deliberate matrix change.
