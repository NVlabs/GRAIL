# Hunyuan3D object generation

## Object descriptions

Use this route for textured rigid objects. Keep the user's category, count and
constraints, and produce distinct descriptions whose generated slugs do not
collide. Explicit materials, handles, feet and support surfaces improve the
specification. Use the existing lists under `configs/gen_3d/` as format examples.

For indoor manipulation, graspable objects with visible orientation cues are
usually easier to render and track. Approximate historical scene ranges were
5–60 cm for tabletop items and 50–150 cm for floor items; measure the actual
generated dimensions and scene fit rather than treating these as acceptance
thresholds. Tiny details may disappear in the source image, and a rotationally
symmetric object may leave orientation ambiguous even with successful tracking.

Hunyuan produces a rigid mesh, not functional articulation. For requests involving
hinges, cloth, flexible objects or moving parts, explain that representation limit
and resolve any material ambiguity before changing the requested object. A
painted seam does not create a mechanical joint. Plain block/pedestal prompts can
produce an unwanted baked-in base; describe the intended contact surface and
inspect it rather than silently rescaling or deleting geometry downstream.

Record the descriptions in a batch-specific YAML list:

```yaml
- Compact metal toolbox with a rigid carry handle and flat rubber feet
- Rounded ceramic storage jar with a wide opening and a blue patterned surface
```

## Runtime and execution

Check the selected Hunyuan runtime and initialized `imports/Hunyuan3D-2.1`
dependency. The image-generation/verification adapter needs its configured API
credentials; test their availability without printing values. Consult
[the asset guide](../../../../docs/source/gen_3d_assets.md) and
`grail/adapters/openai_api.py` for actual provider/model settings rather than
assuming a historical service or latency estimate still applies.

From the repository root, with a new batch YAML and output directory:

```bash
conda run --no-capture-output -n hunyuan python -u -m grail.pipelines.gen_3d_assets \
    -i configs/gen_3d/<batch>.yaml \
    -o data/gen_<batch> \
    --max-image-retries 3
```

The image-verification retry count is an explicit generator option, not a policy
to repeat a whole campaign. Inspect actual errors and existing work before any
further attempt. The default skip checks for an existing textured mesh; it does
not compare the prompt, model or quality record. Bind those separately when
resuming. Use a new output namespace for changed inputs instead of silently
replacing a same-slug asset.

Image generation is
API-bound while shape/texturing is GPU-bound; choose concurrency from actual
capacity and provider behavior. Extra GPUs do not increase an API quota. Reuse
cached images only when the selected implementation supports that path and the
input identity matches.

## Completion

Retain the YAML, logs and partial outputs. Verify each requested asset's mesh,
materials/textures, units, up-axis, extents and intended support surface. Report
missing or unusable assets separately from produced files. Do not remove partial
directories as automatic cleanup. Carry the selected asset and its original
scale/coordinate convention into 2D scene configuration.
