# Implementation Guide

## Architecture

The browser sends one leaf image plus optional coarse context to the Flask API.
The API validates the file, calculates a quality score, runs the versioned
MobileNetV3 checkpoint when present, and records the output in SQLite. The
history API and CSV export provide traceability. Training and robustness
evaluation are deliberately offline so experimental results cannot be altered
by serving traffic.

## Research questions

1. How much does classification accuracy fall under increasing blur, exposure,
   noise, resolution loss, and JPEG compression?
2. Is model confidence calibrated to those performance reductions?
3. Which crop-condition classes are most sensitive to each degradation?
4. Do coarse contextual strata reveal systematic reliability differences?

## Experimental controls

- Fixed seed: 42
- Leaf-group-aware 70/15/15 split
- Same checkpoint and preprocessing for every test condition
- Three severity levels per degradation
- Clean test set retained as the reference
- Report accuracy, macro precision/recall/F1, per-class recall, confidence, and
  accuracy drop; use bootstrap confidence intervals for the final thesis

## Contextual metadata

Metadata is logged for stratified evaluation and traceability. It does not
alter the baseline prediction. A later experimental branch may use crop hint
to mask impossible classes, but must report both unassisted and assisted
performance and must test incorrect hints.

## Acceptance criteria

- Dataset validator finds exactly 38 class folders
- No leaf group appears in more than one split
- Training history and versioned checkpoint are saved
- Clean and 18 degraded test conditions are evaluated
- API rejects invalid/oversized inputs
- Demo mode never presents a fabricated diagnosis
- Every prediction records model version, quality score, timestamp, and context

## Suggested thesis analysis

After training, add bootstrap 95% confidence intervals, paired McNemar tests
between clean and degraded predictions, expected calibration error, and
class-wise error analysis. Treat metadata subgroup results as exploratory when
sample sizes are small.

