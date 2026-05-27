# Pre-PyPI Release Checklist

## Package readiness

- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG/release notes prepared
- [ ] License and metadata verified
- [ ] `README.md` updated with latest methods and CLI options

## Quality gates

- [ ] `pytest -q` passes
- [ ] `ruff .` passes
- [ ] `mypy` passes for enforced modules
- [ ] Optional-dependency smoke tests pass (`tests/test_optional_imports.py`)

## API/Schema stability

- [ ] `DriftResult` shape unchanged or documented
- [ ] CLI JSON `schema_version` validated
- [ ] CLI compatibility tests pass

## Benchmark and observability

- [ ] Benchmark CLI outputs JSON and CSV reports correctly
- [ ] Telemetry hooks do not fail when OpenTelemetry is unavailable
- [ ] Telemetry metrics (`latency`, `drift_rate`, `error_count`) exercised in tests

## Publish steps

1. Build distributions:
   `python -m pip install build && python -m build`
2. Validate artifacts:
   `python -m pip install twine && twine check dist/*`
3. Upload to TestPyPI (recommended first):
   `twine upload --repository testpypi dist/*`
4. Upload to PyPI:
   `twine upload dist/*`
5. Create git tag and GitHub release.
