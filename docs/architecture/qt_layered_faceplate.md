# Layered Qt faceplate

The production Qt interface replaces rectangular native-widget overlays with one uniformly scaled scene at a canonical 1536 × 1024 design resolution.

The production layer order is:

1. smoked-glass wells and data-bearing instrumentation;
2. the RGBA faceplate overlay with permanent engraving and transparent aperture interiors;
3. focus and diagnostic overlays only when required.

`assets/instrument_apertures.json` is the geometry authority. It records each aperture silhouette, its safe content rectangle, permanent panel labels, permanent field labels, and title typography. `tools/build_faceplate_overlay.py` deterministically composites the overlay from the original chassis and bundled font files. It does not regenerate, rescale, or reinterpret the plate artwork.

Typography is restricted to three roles:

- **Cinzel Bold** — `CINELINGUS ENGINE`, uppercase, 0.10 em tracking;
- **IBM Plex Sans Condensed Medium** — permanent panel, field, station, verdict, and service labels;
- **Share Tech Mono** — changing values, progress, observations, times, percentages, and machine state.

`cinelingus-gui` launches Qt maximized by default. The production controller owns apparatus selection, film admission, parameter normalization, media preflight, background pipeline execution, progress, cancellation, the technical record, and archive access without importing Tk. `cinelingus-gui --windowed` uses the canonical design size. The retired shell is available only through `cinelingus-gui-legacy`, `--legacy-tk`, or `CINELINGUS_LEGACY_TK=1`.

Deterministic review captures use `--screenshot`, `--state`, and `--scale`; this avoids unreliable foreground-window screen grabs. Operator controls are painter-rendered and clipped to aperture paths. Native Qt widgets appear only in separate configuration and service dialogs, never over the faceplate.
