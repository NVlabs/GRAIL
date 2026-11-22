# 3D Asset Generation

<span id="generate-a-3d-asset"></span>

The released stairs and kettlebell tutorials start with a generated video and its scene files. This guide builds a stair mesh on the CPU. The release includes no complete worked example that follows a third object from its mesh through video generation to a reviewed robot motion.

## Procedural terrain

<span id="make-one-staircase"></span>

In the `grail` environment, from the checkout root, generate a uniform four-step staircase. The count options select four steps in this **one** asset. The seed fixes the random draws.

```bash
python -m grail.pipelines.gen_terrain --type stairs --num 1 --seed 42 --uniform \
  --stair-min-steps 4 --stair-max-steps 4 --output_dir data/guide_terrain
```

Open `data/guide_terrain/stairs_000/model.obj`. Keep `model.mtl` and `texture.jpg` beside it. The verified output was 6.6 KiB of OBJ geometry, a 129-byte material and a 1.8 KiB texture. Generation took less than a second on an Intel Core i9-14900K in `nvgrail/grail:1.2.0`, without a GPU. The [video generation guide](gen_2dhoi.md) uses this exact directory.

If the renderer reports a missing mesh, check the dataset directory and category separately. Here the dataset is `guide_terrain` and the category is `stairs_000`. If the shape opens without its texture, check the two relative material references before moving the asset.

## Manipulation objects

<span id="choose-a-manipulation-object"></span>

For pickup, put your metric, triangulated rigid mesh at `data/<dataset>/<category>/model.obj` with the referenced MTL and textures. Keep its scale and vertex order after binding contacts. Give your dataset and category their own entry in a copy of the {blob}`object list <configs/objects/robocasa.yaml>`, then select that list from your [pickup preset](configuration.md#configuration-by-motion-type). Choose a camera that shows the handle opening and lift direction.

<span id="hunyuan3d-text-textured-mesh"></span>
<span id="bundled-object-lists"></span>

For text-to-mesh, the {blob}`Hunyuan3D pipeline <grail/pipelines/gen_3d_assets.py>` takes a list of objects such as {blob}`configs/gen_3d/example_objects.yaml`. Select the list with `--input` and a new asset directory with `--output`. It generates a reference image, reconstructs and textures the mesh, and estimates scale. Use the `hunyuan` environment, a GPU and `OPENAI_API_KEY` for the image service. The tutorial download uses `--skip-hunyuan3d`. For this route, omit that flag when downloading checkpoints to fetch `RealESRGAN_x4plus.pth`. The Hunyuan3D-2.1 shape and paint weights download on first use. Inspect the available inputs in the release image:

```bash
conda run -n hunyuan python -m grail.pipelines.gen_3d_assets --help
```

The CLI loaded in about half a minute on the reference workstation. This check exercised argument discovery only. Text-to-mesh generation time and peak GPU memory were not measured for this guide. Inspect any generated mesh and verify its scale before fitting a human to it.

<span id="sharded-fan-out"></span>

For a larger batch of independent assets, the Hunyuan entry point accepts a chunk count and index. Use the same object list on each worker and assign disjoint indices. The timed example above covers CLI setup only.

## Asset collections

<span id="generate-a-collection"></span>

From the checkout root, pick a procedural type (`curb`, `slope`, `stairs` or `all`) and keep the seed to reproduce its geometry. For example:

```bash
python -m grail.pipelines.gen_terrain --type slope --num 3 --seed 1234 --output_dir /workspace/runs/slopes
```

For text-conditioned generation, use the separate Hunyuan environment and [generation prerequisites](manual_installation.md). The pipeline enhances the object description, generates a reference image, checks image–prompt agreement, removes the background, generates shape and texture, and estimates metric height. Adjust `--max-image-retries` to bound reference-image verification. Inspect the resulting OBJ, material and texture before preparing an interaction. Supply an object list and a fresh output location:

```bash
conda run -n hunyuan python -m grail.pipelines.gen_3d_assets \
  -i configs/gen_3d/example_objects.yaml -o /workspace/runs/generated-objects
```

Use `configs/gen_3d/chairs.yaml` for a larger chair collection. On each of N workers pass the same input and `--num_job_chunks N --job_chunk_idx I`, with a distinct zero-based I. These options were checked in the image. Generating a new textured collection requires model downloads, API access and GPU time.

### Object selection guidelines

<span id="select-objects-that-can-be-tracked"></span>

Choose a rigid object with a visible handle, texture or distinct faces. A smooth sphere has no observable orientation. Very small objects can disappear in a 720p frame, and a large tabletop object can exceed the character's reach. As starting ranges, the earlier asset guide used a largest dimension of 5–60 cm on a table and 50–150 cm on the floor. Check the actual scene. Avoid hinged or flexible items for a single-rigid-mesh pipeline. Describe the real contact surface, such as four recessed feet, to reduce spurious plinths in generated geometry.
