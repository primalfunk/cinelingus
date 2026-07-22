# Instrument Material System

The Cinelingus faceplate is a replaceable chassis. All text, state, and interaction remain live widgets positioned by `instrument_ui.py`; the plate contains no functional labels.

## Visual grammar

- Brass marks engraved headings, scales, hardware edges, and adjustable controls.
- Cyan is emitted light: active lamps, live readings, focus, and meter fill.
- Warm white is operator material such as selected filenames and current values.
- Dim bronze marks dormant but available mechanisms.
- Desaturated grey is reserved for genuinely unavailable controls.

Every visible control must appear to belong to the same manufactured object. Panels use a shared recessed edge and light direction. Buttons use one raised hardware treatment. Readouts sit in dark wells rather than default operating-system fields.

## Interaction grammar

- Rotary selectors expose calibrated detents and remain operable by mouse wheel, mouse buttons, arrows, Return, and Space.
- The activation control has distinct ready and engaged materials. Engaged remains illuminated while execution is locked.
- Safe Interrupt is visually subordinate but remains keyboard focusable while enabled.
- Progress meters are continuous recessed channels. Stage lamps remain discrete and advance through dormant, active, and complete states.
- Material paths are read-only displays with full-path tooltips. `LOAD`, `ADD MATERIAL`, and `EJECT` perform all mutations.
- Curator controls remain dormant until a completed result can be examined.

## Hierarchy

Transformation is explicitly split into `FIELD` and the rotary experiment selector. Filter is explicitly split into the rotary matching profile and `BIAS`. This prevents coincident labels such as Balanced from appearing to describe the same setting.

The central Observation display remains the only primary status surface. Logs, diagnostics, reports, and advanced settings belong to Laboratory Notes.

## Replacement and scaling

`InstrumentPlateCanvas` fits the plate proportionally and positions overlays from normalized geometry. Each named recess has a separate inset content box so live controls leave its engraved bezel, corners, and separator rules visible at every scale. A replacement plate may change ornament and surface texture, but it must preserve the named recesses or update both the recess and content geometry. Control behavior and evidence semantics must not depend on pixels embedded in the artwork.
