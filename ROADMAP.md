# Project Roadmap

This roadmap outlines the main steps to make **drift-control** a professional package for monitoring and handling data and concept drift in machine-learning models.

## Completed

- [x] Package configured with **Poetry** using only `pyproject.toml`
- [x] Added MIT licensing information and `LICENSE` file
- [x] Included maintainer information linking to [ORCID](https://orcid.org/0009-0001-2022-7072)
- [x] Added basic detectors and alert utilities with accompanying tests
- [x] Exposed package version dynamically via `importlib.metadata`
- [x] Implemented PSI-based drift detection
- [x] Added KS-based drift detection

## Next Steps

- [ ] Expand documentation with detailed examples and API references
- [ ] Publish the package to PyPI
- [x] Implement additional drift detection algorithms (e.g., KS-test, clustering-based)
- [ ] Support multivariate and concept drift detection techniques
- [ ] Integrate with scikit-learn pipelines and major ML frameworks
- [ ] Provide real-time monitoring utilities for streaming data
- [ ] Offer dashboards and visualization tools for drift analysis
- [ ] Add dataset versioning and baseline management helpers
- [ ] Set up continuous integration and automated testing
- [ ] Provide a command line interface and REST API for easy adoption

