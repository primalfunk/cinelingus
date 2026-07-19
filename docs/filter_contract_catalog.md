# Cinelingus Filter Contract Catalog

Generated from the machine-valid contracts in filter_contracts/.

| Filter | Family | Films | Cinematic law | Status | Execution | Contract proposition |
|---|---|---:|---|---:|---|---|
| Exhaustion | emotion | 1 | The film speaks through exhaustion | accepted | scheduling_strategy | The film speaks through exhaustion. |
| Optimist | emotion | 1 | The film speaks through optimist | accepted | scheduling_strategy | The film speaks through optimist. |
| Paranoia | emotion | 1 | The film speaks through paranoia | accepted | scheduling_strategy | The film speaks through paranoia. |
| Regret | emotion | 1 | The film speaks through regret | accepted | scheduling_strategy | The film speaks through regret. |
| Wonder | emotion | 1 | The film speaks through wonder | accepted | scheduling_strategy | The film speaks through wonder. |
| Bloom | experimental | 1 | Replacement frequency and measured transformation strength grow along the configured nonlinear curve | accepted | scheduling_strategy | Replacement frequency and measured transformation strength grow along the configured nonlinear curve. |
| Ouroboros | experimental | 1 | The film consumes its own conclusion | accepted | scheduling_strategy | The film consumes its own conclusion. |
| Shed Skin | experimental | 1 | The film leaves voices behind | accepted | scheduling_strategy | The film leaves voices behind. |
| Venom | experimental | 1 | Meaning turns against its scene | accepted | scheduling_strategy | Meaning turns against its scene. |
| Chorus | identity | 1 | One anchor speaker supplies every replacement across a bounded set of non-anchor speakers | accepted | scheduling_strategy | One anchor speaker supplies every replacement across a bounded set of non-anchor speakers. |
| Doppelgänger | identity | 1 | Exactly two selected speakers exchange dialogue identities bidirectionally without changing the pair | accepted | scheduling_strategy | Exactly two selected speakers exchange dialogue identities bidirectionally without changing the pair. |
| Possession | identity | 1 | One selected possessing speaker supplies every replacement for one distinct possessed speaker throughout the run | accepted | scheduling_strategy | One selected possessing speaker supplies every replacement for one distinct possessed speaker throughout the run. |
| Split Personality | identity | 1 | A voice becomes multiple occupants | accepted | scheduling_strategy | A voice becomes multiple occupants. |
| Contagion | infection | 1 | A carrier identity spreads only after measured speaker contact and may transform a speaker only at or after infection time | accepted | scheduling_strategy | A carrier identity spreads only after measured speaker contact and may transform a speaker only at or after infection time. |
| Dialect | infection | 1 | The cast acquires a common tongue | accepted | scheduling_strategy | The cast acquires a common tongue. |
| Mutation | infection | 1 | Contamination alters what it carries | accepted | scheduling_strategy | Contamination alters what it carries. |
| Whisper | infection | 1 | A voice enters at the edge of hearing | accepted | scheduling_strategy | A voice enters at the edge of hearing. |
| Amnesia | memory | 1 | The film forgets how it spoke | accepted | scheduling_strategy | The film forgets how it spoke. |
| Dream | memory | 1 | The film dreams its own speech | accepted | scheduling_strategy | The film dreams its own speech. |
| Recollection | memory | 1 | The film remembers aloud | accepted | scheduling_strategy | The film remembers aloud. |
| Bleed | multiworld | 2 | Reality Collision | blocked | unavailable | Images, sound, and narrative details leak between worlds. |
| Chimera | multiworld | 3 | Genre Mutation | blocked | unavailable | Three films combine into one hybrid cinematic organism. |
| Civilization | multiworld | 5+ | Reality Collision | blocked | unavailable | Five or more films form a persistent shared cinematic society. |
| Contagion | multiworld | 2+ | Narrative Infection | accepted | transformation | Donor films infect the anchor timeline in ordered, non-reverting phases. |
| Doppelgänger | multiworld | 2 | Identity Exchange | blocked | unavailable | Two films discover recurring doubles in one another. |
| Echo Chamber | multiworld | 2+ | Translation | accepted | transformation | Every selected anchor window becomes a staggered, attenuated echo group containing every film. |
| Mirror World | multiworld | 2 | Reality Collision | blocked | unavailable | An anchor film is reflected through a second cinematic reality. |
| Parallel Universes | multiworld | 2+ | Temporal Exchange | blocked | unavailable | Equivalent moments from multiple films coexist on a shared timeline. |
| Possession | multiworld | 2 | Identity Exchange | accepted | transformation | One film's stable donor identity takes residence inside one recurring anchor-film speaker. |
| Prophecy | multiworld | 2 | Temporal Exchange | accepted | transformation | Later normalized dialogue from one film predicts earlier positions in the anchor film. |
| Translation | multiworld | 2 | Translation | accepted | transformation | Dialogue from a donor film fills speaking windows in the anchor film while anchor picture and chronology remain fixed. |
| Triangle | multiworld | 3 | Narrative Infection | blocked | unavailable | Three films exchange pressure through a closed narrative relationship. |
| Wormhole | multiworld | 2+ | Temporal Exchange | blocked | unavailable | Selected moments cross between films through deterministic temporal portals. |
| Flashback | time | 1 | Every replacement source is earlier than its destination by more than the configured minimum distance | accepted | scheduling_strategy | Every replacement source is earlier than its destination by more than the configured minimum distance. |
| Foreshadow | time | 1 | Dialogue normally comes from sufficiently later moments; any configured wraparound is explicit and separately counted | accepted | scheduling_strategy | Dialogue normally comes from sufficiently later moments; any configured wraparound is explicit and separately counted. |
| Möbius | time | 1 | The film discovers its other side | accepted | scheduling_strategy | The film discovers its other side. |
| Spiral | time | 1 | Absolute source-to-destination displacement never decreases across the selected sequence and follows the configured direction policy | accepted | scheduling_strategy | Absolute source-to-destination displacement never decreases across the selected sequence and follows the configured direction policy. |
| Drift | translation | 1 | Each original line is delayed by an offset that grows linearly with its source position | accepted | transformation | Each original line is delayed by an offset that grows linearly with its source position. |
| Echo | translation | 1 | Eligible original lines repeat at the configured positive delay without replacing their source occurrence | accepted | transformation | Eligible original lines repeat at the configured positive delay without replacing their source occurrence. |
| Self Shuffle | translation | 1 | A film's own whole lines move to different speaking slots; no enabled mapping may substantially overlap its original slot | accepted | transformation | A film's own whole lines move to different speaking slots; no enabled mapping may substantially overlap its original slot. |

## Procedure status

Procedure Behaviour entries are architectural commitments. No contract in this catalog claims multi-step runtime support.

