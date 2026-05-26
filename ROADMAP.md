# Project Roadmap

## Completed

- [x] Package structure and Poetry build configuration
- [x] Core univariate drift detectors (PSI, KS)
- [x] Multivariate covariate shift detector
- [x] Streaming concept drift support (DDM, EDDM)
- [x] Streaming monitors and sklearn adapter
- [x] CLI with machine-readable JSON output
- [x] Visualization helpers and baseline management
- [x] Multivariate kernel drift detector (MMD + permutation calibration)
- [x] Additional concept drift detectors (ADWIN, Page-Hinkley)

## Next Steps

- [ ] Standardize detector output schema across all methods
- [ ] Add typed public protocol for custom detectors and monitors
- [ ] Add benchmark harness for drift scenarios (mean shift, variance shift, label shift)
- [ ] Add typed config objects for CLI and library calls
- [ ] Publish docs site with API reference + operational guides
- [ ] Expand MLOps integrations and deployment examples
