# Procedural terrain

Use `grail.pipelines.gen_terrain` for curbs, slopes and stairs. It needs the
procedural geometry/image dependencies, not a Hunyuan model or video API. Resolve
type, count, dimensions, seed and intended ascent/descent use from the request.
Read the current CLI definitions in `grail/pipelines/gen_terrain.py` before
choosing optional dimension flags; preserve requested bounds explicitly.

For a new stair dataset:

```bash
python -m grail.pipelines.gen_terrain \
    --type stairs --num <count> --seed <seed> \
    --output_dir data/syn_<batch>
```

`--type all` produces each supported terrain type, with `--num` assets per type.
If separate datasets are useful, invoke each type into its own new directory.
The generator uses deterministic indexed names; inspect the destination first
and choose a fresh directory when names would collide. Do not use a shared
`/tmp/terrain_gen` directory or overwrite an existing dataset incidentally.

Each generated asset contains `model.obj`, `model.mtl` and `texture.jpg`.
Measure actual geometry, axis convention, support surfaces, step count and
rise/tread/width or slope. Rendering a terrain mesh does not prove that a human
or robot can traverse it.

For the requested video stage, select the corresponding current config under
`configs/gen_2dhoi/` through [grail-2dhoi](../../grail-2dhoi/SKILL.md). Bind the
original terrain and scene placement for later locomotion support fitting.
