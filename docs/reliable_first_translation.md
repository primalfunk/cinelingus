# Reliable First Translation

This milestone establishes one dependable full-source path before additional filters are certified.

## Input behavior

- A fresh GUI launch does not treat committed example paths as selected material.
- Film selection starts in `~/Downloads/Music`.
- If that folder is unavailable, selection falls back through `~/Downloads`, `~/Music`, and the user home folder.
- After a film is selected, the next chooser starts beside that film for the remainder of the session.

## Preflight contract

Activation is blocked before analysis unless:

- every selected path is a real file that FFmpeg can probe;
- every film has a positive-duration video stream;
- every supporting film has a positive-duration audio stream;
- a one-film operation has its own usable audio stream; and
- the output directory can be created and written.

The predicted output duration is the shorter of the complete anchor video stream and all required supporting audio streams. The operator sees this prediction before the worker starts.

## Render acceptance

The final render must retain exactly one video stream and exactly one intentional replacement-audio stream. Container and audio duration must match the full-timeline plan within the strict configured tolerance. A copied video stream may end on the last encoded packet preceding the audio boundary; acceptance allows at most 250 milliseconds for that codec-packet tail and records the measured value.

The deterministic regression creates real media, gives the anchor a longer runtime than the replacement audio, muxes the result, probes the MP4, and requires the output to end at the supporting-audio boundary with one audio stream.
