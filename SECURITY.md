# Security policy

## Supported version

The latest commit on `main` is the supported research version.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature rather than opening a public issue. Include the affected endpoint, reproduction steps, expected impact, and any suggested mitigation. Do not include real personal, medical, or farm-identifying data.

## Scope and limitations

PlantGuard Lab is a research artefact, not a production diagnostic service. It stores prediction metadata in a local SQLite database and is intended to run on a trusted local machine. Production deployment would require authentication, request throttling, malware scanning, encrypted storage, retention controls, observability, and a formal privacy and security review.
