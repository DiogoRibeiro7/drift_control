# Documentation

This folder contains the publish-ready reference docs for `drift-control`.

## Structure

- `api/reference.md`: public Python API reference
- `api/detector-selection-guide.md`: quick "use X when..." guide across detectors
- `cli/schema.md`: JSON output schema, versioning, and examples
- `integrations/pipeline-integrations.md`: MLflow, DVC, and feature-store integration patterns
- `integrations/evidently-nannyml-compat.md`: migration mapping from Evidently/NannyML workflows
- `operations/production-playbook.md`: baseline sizing, thresholding, alerts, and response strategy
- `operations/release-checklist.md`: pre-PyPI release checklist
- `operations/drift-incident-template.md`: incident report template for drift events
- `operations/drift-postmortem-checklist.md`: postmortem closure checklist

## Suggested publication flow

1. Keep docs in-repo under `docs/`.
2. Publish with a docs site generator (MkDocs, Sphinx, or GitHub Pages).
3. Tag releases only when checklist items are complete.
