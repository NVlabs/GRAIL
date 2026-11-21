# Newton contact detection

Optional adapter for an external contact-detection backend that is not
distributed with GRAIL (the importable `ho_contact` package at the revision
pinned in `src/collision.py`). It executes the actual `ContactScene` and
`extract_human_object_contacts` implementation in an external Python process.
Core implementation lives in `src/collision.py` and `src/collision_backend.py`.
The Blender integration exports an explicit mesh snapshot and displays the
result as diagnostic evidence; it does not change the pose or run a motion
repair solver.

## External runtime

Keep this older Newton version separate from the SOMA retargeter environment.
Clone dependencies locally; their source and assets are not bundled here:

```bash
# /absolute/deps/ho_contact_detection: a separately obtained checkout of the
# external backend at the pinned revision, with its Newton submodule initialized.
python3 -m venv /absolute/venvs/ho-contact
/absolute/venvs/ho-contact/bin/python -m pip install numpy==2.2.6 scipy==1.15.3 warp-lang==1.9.1
/absolute/venvs/ho-contact/bin/python -m pip install --no-deps -e /absolute/deps/ho_contact_detection/third_party/newton
```

The worker requires Python 3.11+ and pins the Newton submodule to
`e5a7cfa02c190eec0f6e94c3844b115b5870792a`; modified implementation or a mismatched
checkout fails. The upstream declared Warp minimum is insufficient for some
Newton imports (Warp 1.7.2 fails): the versions above passed a real CPU smoke
query with Python 3.12.3.
The adapter imports the collision library directly from the explicit checkout,
so RP2 loader, PyTorch, xatlas, and the RP2 CLI are unnecessary for this route.
Newton is Apache-2.0. This adapter does not redistribute the backend's code.

```bash
python tools/collision/run.py \
  --snapshot /absolute/snapshot.json \
  --python /absolute/venvs/ho-contact/bin/python \
  --repo /absolute/deps/ho_contact_detection \
  --device cpu --timeout 120 \
  --output-dir /absolute/new-query
```

`cpu` is the default. `cuda:N` opts into a particular CUDA device. Queries run in
the foreground with a timeout. Each new directory keeps `snapshot.json`,
`report.json`, `stdout.log` and `stderr.log`; failures preserve available logs.
Existing output directories are never reused.

The installed CLI also accepts a runtime JSON file containing `python`, `repo`,
`device` and `timeout_s`:

```bash
python -m agentic_retargeter collision --input /absolute/snapshot.json \
  --runtime /absolute/runtime.json --output /absolute/new-query
```

## Snapshot and report contract

```json
{
  "schema_version": 1,
  "source_frame": 25.5,
  "units": "m",
  "meshes": [
    {
      "name": "left_hand",
      "vertices": [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
      "triangles": [[0, 1, 2]],
      "matrix_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    },
    {
      "name": "object",
      "vertices": [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
      "triangles": [[0, 1, 2]],
      "matrix_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.005], [0, 0, 0, 1]]
    }
  ],
  "pairs": [["left_hand", "object"]]
}
```

This small example illustrates the schema, not reliable contact geometry.
Preserve the original evaluated surface and bind exact frame/subframe. Local
vertices and matrix translation are metres; any Blender scene-unit conversion
must be applied before export. Affine transforms, including scale, are accepted.
Explicit pairs can represent robot–object, robot-link self-contact, or
environment contact. All requested pairs are queried independently; contacts
inside a merged mesh are not detected. A single snapshot says nothing about
unsampled frames.

`canonical_snapshot` validates input; `snapshot_digest` binds geometry, transforms,
frame, units and pairs. `run_query(snapshot, runtime, output_dir)` returns a report;
`validate_report(report, snapshot)` rejects stale data, wrong pair coverage,
unsupported dependency commits and inconsistent point arrays. Reports contain
dependency source hashes and runtime versions, plus per-pair `points_a_world`,
`points_b_world`, `point_distances_m`, `contact_count` and
`max_point_distance_m`. All reported coordinates and distances are metres.

## Interpretation and limits

- This release provides detection and projection, not a pose-repair solver.
- Newton uses mesh-vertex-to-surface queries with its default **10 mm contact
  margin**. Candidates can occur with a positive gap.
- Upstream names the norm of paired world points `penetration_depths`, but that
  quantity is **unsigned point distance**. The adapter labels it accordingly;
  it does not invent normals, signed penetration, force or contact probability.
- No candidate does not establish clearance. Edge/face-only intersection,
  coarse tessellation and deep penetration beyond the query radius can be
  missed. Open meshes, winding and reflected geometry affect sign-dependent
  candidate detection.
- The upstream helper adds a ground plane and then filters ground contacts.
  Explicit environment meshes must be named as query pairs.
- Contact buffer saturation fails the query. Candidate count depends on mesh
  tessellation and is not contact area.
- Whole-mesh intersection checks, source review and temporal checks remain
  separate. Diagnostic output does not select or accept a motion.
