# Optional exact anchor-query execution

Candidate-bank anchor extraction can reuse the original robot-local surface
projections and use the existing compiled float64 triangle query. Set the
following option in a `surface_grasp.propose` bank configuration:

```json
"anchor_queries": {
  "backend": "numba_cached",
  "max_cache_entries": 256,
  "max_mesh_entries": 64
}
```

The default remains the uncached Python implementation. `python_cached` enables
only exact projection caching, which is useful as a separate parity comparison.
`numba_cached` requires Numba and uses `contact_query_numba.OriginalTriangleQuery`;
missing dependencies or unsupported reduction ordering fail explicitly. This
does not use Warp or replace optimization/collision screening backends.

Each bank owns its bounded session. Robot-local query results are keyed by the
complete float64 query bytes and original mesh vertex/face dtype, shape and
bytes. The cache returns isolated result arrays. Compiled traversals retain
immutable mesh instances; a new transformed mesh receives its own traversal.
Posed hand-to-object points are queried again for **every candidate** and their
answers are never cached. Original face IDs, barycentrics, lowest-face tie
breaking, degenerate triangle handling, gap thresholds, spacing, member order
and rejection rules stay on their existing paths.

Reports expose the selected backend, bounded cache statistics, traversal builds
and robot/object query seconds. Query time includes first-use compilation and
is already included in candidate anchor-extraction time; do not add it again.
Patch reports contain per-call counters and query time. Anchor hypotheses and
candidate quality remain unreviewed. CPU fixture parity does not establish
parity or acceleration on real case assets; keep this option explicit until those
actual candidate-bank comparisons pass. A workflow wrapper must forward this
field into the bank configuration before it can use the option.
