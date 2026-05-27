# CLI JSON Schema

`drift-control --output-json` emits versioned machine-readable payloads.

## Schema version

- Top-level key: `schema_version`
- Current value: `"1.0"`

## Base payload

```json
{
  "schema_version": "1.0",
  "method": "psi",
  "threshold": 0.2,
  "columns": {
    "feature_name": {
      "score": 0.34,
      "drift": true,
      "p_value": 0.01
    }
  }
}
```

## Method-specific behavior

- `psi`, `js`: `score` compared with `threshold` (`drift = score > threshold`)
- `ks`, `cvm`, `wasserstein`, `mmd`, `c2st`: p-value based (`drift = p_value < threshold`)
- `mmd`, `c2st`: `columns` contains a single `dataset` key
- `ensemble`: includes top-level `ensemble` block:

```json
{
  "ensemble": {
    "methods": ["psi", "ks", "cvm", "js"],
    "vote_mode": "majority",
    "min_votes": null
  }
}
```

## Backward compatibility contract

- Existing keys remain stable within a major schema version.
- New keys may be added non-breakingly.
- Breaking shape changes require a schema version bump.

## Benchmark CLI schema

`drift-control-benchmark --output-format json` returns:

```json
{
  "schema_version": "1.0",
  "benchmark_type": "synthetic_drift",
  "methods": ["psi", "ks"],
  "sample_size": 300,
  "n_trials": 25,
  "random_seed": 42,
  "results": []
}
```
