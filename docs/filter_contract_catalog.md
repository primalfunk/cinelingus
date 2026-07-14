# Cinelingus Filter Contract Catalog

Generated from the machine-valid contracts in filter_contracts/.

| Filter | Family | Status | Class | Execution | Contract law |
|---|---|---:|---:|---|---|
| Exhaustion | emotion | deferred | C | unavailable | The film speaks through exhaustion. Dialogue is redirected through exhaustion. |
| Optimist | emotion | blocked | B | unavailable | The film speaks through optimist. Dialogue is redirected through optimist. |
| Paranoia | emotion | blocked | B | unavailable | The film speaks through paranoia. Dialogue is redirected through paranoia. |
| Regret | emotion | blocked | B | unavailable | The film speaks through regret. Dialogue is redirected through regret. |
| Wonder | emotion | blocked | B | unavailable | The film speaks through wonder. Dialogue is redirected through wonder. |
| Bloom | experimental | accepted | A | scheduling_strategy | Replacement frequency and measured transformation strength grow along the configured nonlinear curve. |
| Ouroboros | experimental | deferred | E | unavailable | The film consumes its own conclusion. The ending feeds dialogue back into the beginning. |
| Shed Skin | experimental | deferred | D | unavailable | The film leaves voices behind. Dialogue identities are discarded in stages. |
| Venom | experimental | blocked | B | unavailable | Meaning turns against its scene. Dialogue becomes progressively hostile to its original context. |
| Chorus | identity | accepted | A | scheduling_strategy | One anchor speaker supplies every replacement across a bounded set of non-anchor speakers. |
| Doppelgänger | identity | accepted | A | scheduling_strategy | Exactly two selected speakers exchange dialogue identities bidirectionally without changing the pair. |
| Possession | identity | accepted | A | scheduling_strategy | One selected possessing speaker supplies every replacement for one distinct possessed speaker throughout the run. |
| Split Personality | identity | blocked | A | unavailable | A voice becomes multiple occupants. One speaker divides across several dialogue identities. |
| Contagion | infection | accepted | A | scheduling_strategy | A carrier identity spreads only after measured speaker contact and may transform a speaker only at or after infection time. |
| Dialect | infection | deferred | C | unavailable | The cast acquires a common tongue. A shared vocal pattern spreads between speakers. |
| Mutation | infection | deferred | F | unavailable | Contamination alters what it carries. Infected dialogue changes form as it spreads. |
| Whisper | infection | deferred | C | unavailable | A voice enters at the edge of hearing. Dialogue traits begin to spread quietly. |
| Amnesia | memory | blocked | A | unavailable | The film forgets how it spoke. Dialogue identities and repetitions gradually disappear. |
| Dream | memory | blocked | B | unavailable | The film dreams its own speech. Dialogue returns through associative memory. |
| Recollection | memory | blocked | B | unavailable | The film remembers aloud. Past dialogue resurfaces in later scenes. |
| Flashback | time | accepted | A | scheduling_strategy | Every replacement source is earlier than its destination by more than the configured minimum distance. |
| Foreshadow | time | accepted | A | scheduling_strategy | Dialogue normally comes from sufficiently later moments; any configured wraparound is explicit and separately counted. |
| Möbius | time | blocked | A | unavailable | The film discovers its other side. Beginning and ending dialogue fold into one another. |
| Spiral | time | accepted | A | scheduling_strategy | Absolute source-to-destination displacement never decreases across the selected sequence and follows the configured direction policy. |
| Drift | translation | accepted | A | transformation | Each original line is delayed by an offset that grows linearly with its source position. |
| Echo | translation | accepted | A | transformation | Eligible original lines repeat at the configured positive delay without replacing their source occurrence. |
| Transposition | translation | accepted | A | transformation | Dialogue from the source film fills destination speaking windows while destination picture and chronology remain fixed. |
| Self Shuffle | translation | accepted | A | transformation | A film's own whole lines move to different speaking slots; no enabled mapping may substantially overlap its original slot. |

## Procedure status

Procedure Behaviour entries are architectural commitments. No contract in this catalog claims multi-step runtime support.

