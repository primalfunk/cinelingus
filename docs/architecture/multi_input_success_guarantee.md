# Multi-input success guarantee

Cinelingus treats “success for every appropriate filter” as an executable contract, not as an optimistic promise that any file on any machine will render. For two or more inputs, a filter may render only after the system has proved that the selected world is valid for that filter and that its qualified schedule can satisfy the filter's law. A completed experiment is successful only when the encoded MP4 is certified from persisted evidence.

## Applicability

| Input count | Filters included in the guarantee |
| --- | --- |
| 2 | Translation, Possession, Contagion, Echo Chamber, Prophecy |
| 3 or more | Contagion, Echo Chamber |

One-film operators are deliberately outside this matrix. An unimplemented filter and a filter whose declared arity does not accept the selected film count are also outside it; neither is silently attempted.

## What “valid” means

Every selected input must be a complete, readable media file with a usable video stream, a usable audio stream, and positive canonical duration. The filter must be implemented and accept the input count. Analysis must also provide any capability required by the filter, such as direct speaker-identity evidence for Possession. Failure of any of these conditions is a pre-render rejection with a named reason, not a failed or misleading MP4.

## Guarantee stages

1. Media preflight proves that every path is readable and probeable.
2. The contract kernel records all films, stream facts, hashes, arity, full-timeline extent, audio law, repetition policy, and acceptance tolerances.
3. Schedule qualification removes unauthorized repetition and other unusable placements. Grouped laws are preserved atomically: if anti-repetition qualification damages an Echo Chamber group, the entire group is left unmodified.
4. `multi_input_guarantee.json` proves that every required film contributes qualified material, no unknown film contributes, identity capability is sufficient, and the filter is applicable.
5. Rendering consumes Film A continuously from timestamp zero to the canonical extent. The extent is the shorter of Film A's video and every required supporting-audio boundary.
6. Filter acceptance proves the declared cinematic invariants. Render acceptance probes the published MP4 for exactly one video stream, exactly one replacement-audio stream, duration agreement, provenance, and meaningful audio.
7. `filter_certification.json` derives `CERTIFIED`, `DEGRADED`, or `BLOCKED` solely from those artifacts. `DEGRADED` is a successful, disclosed safe omission; it is never a hidden invariant failure.

The matrix runner executes every applicable filter:

```powershell
python tools/certify_multi_input_filters.py OUTPUT_DIR FILM_A FILM_B [FILM_C ...]
```

The independent evidence audit rechecks an existing matrix without re-rendering:

```powershell
python tools/audit_multi_input_certifications.py OUTPUT_DIR
```

The audit passes only when all applicable filters refer to the same input hashes, all guarantee and certification checks pass, the final MP4s exist at the recorded paths, encoded duration matches the canonical contract, and post-qualification grouped laws still hold.

## Boundary of certainty

This architecture gives a software-level guarantee: accepted valid inputs either produce a certified result for every applicable filter or are rejected before rendering with exact evidence about the unmet capability. It cannot make power loss, disk failure, hardware faults, exhausted storage, killed processes, or defects in external codecs physically impossible. Those environmental failures remain explicit run failures and cannot be reported as certified success.
