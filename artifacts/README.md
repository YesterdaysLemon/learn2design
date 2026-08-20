# Submission artifacts

This directory documents and indexes files prepared for competition submission or external review.

Generated ZIP bundles and machine-produced reports belong under `artifacts/generated/` and are intentionally ignored by Git. Each release-worthy artifact should have a small manifest containing its checksum, source revision, creation command, and validation result.

Completed UIFO studies can be revalidated and packaged with
`python tools/package_uifo_study.py STUDY_DIR --output STUDY.zip`. The tool
refuses active, incomplete, foreign, or internally inconsistent studies and
writes `.sha256` and `.manifest.json` sidecars for off-machine verification.
