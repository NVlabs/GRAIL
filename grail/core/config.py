import os

import yaml

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "configs")
DEFAULT_RECON_CONFIG = os.path.join(_CONFIG_DIR, "recon_4dhoi", "manip_soma.yaml")
_DEFAULT_SCENES = os.path.join(_CONFIG_DIR, "scenes", "default.yaml")
_DEFAULT_OBJECTS = os.path.join(_CONFIG_DIR, "objects", "comasset.yaml")
_DEFAULT_PIPELINE = os.path.join(_CONFIG_DIR, "gen_2dhoi", "manipulation.yaml")


def load_config(cfg_path):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file {cfg_path} not found")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_scene_config(path=None):
    with open(path or _DEFAULT_SCENES) as f:
        return yaml.safe_load(f)


def load_object_config(path=None):
    """Load object config YAML and return the objects dict.

    The YAML is expected to have a top-level ``objects:`` mapping.
    Only the ``objects`` dict is returned so callers can do
    ``config[category]`` directly.  Use ``load_object_config_full()``
    if you also need ``dataset``.
    """
    with open(path or _DEFAULT_OBJECTS) as f:
        raw = yaml.safe_load(f)
    if "objects" in raw:
        return raw["objects"]
    return raw


def load_object_config_full(path=None):
    """Return the full object config: ``{dataset, objects}``."""
    with open(path or _DEFAULT_OBJECTS) as f:
        return yaml.safe_load(f)


def load_pipeline_config(path=None):
    """Load a pipeline YAML and resolve the referenced object config.

    Returns a dict with keys:
      - All top-level pipeline settings (character, results_dir, skip_step*, …)
      - ``scenes``   – dict of scene definitions (embedded in pipeline YAML)
      - ``objects``   – dict of per-object settings (loaded from the referenced
                        object_config YAML, with ``dataset`` popped to top level)
      - ``dataset``   – string pulled from the object config YAML
      - ``categories`` – list of object keys (everything except ``dataset`` and
                         ``default``)
      - ``object_config_path`` – resolved path so sub-processes can receive it
    """
    with open(path or _DEFAULT_PIPELINE) as f:
        cfg = yaml.safe_load(f)

    obj_cfg_path = cfg.get("object_config", _DEFAULT_OBJECTS)
    full = load_object_config_full(obj_cfg_path)

    objects = full.get("objects", full)
    dataset = full.get("dataset")
    categories = [k for k in objects if k != "default"]

    cfg["objects"] = objects
    cfg["dataset"] = dataset
    cfg["categories"] = categories
    cfg["object_config_path"] = obj_cfg_path

    scene_cfg_path = cfg.pop("scene_config", None)
    if scene_cfg_path and "scenes" not in cfg:
        cfg["scenes"] = load_scene_config(scene_cfg_path)

    return cfg


SCENE_CONFIG = load_scene_config()
OBJECT_CONFIG = load_object_config()


# -- 2D Generation config -----------------------------------------------------

# Keys renamed when flattening the 2D-generation YAML sections so that
# the resulting dict matches the attribute names expected by grail.pipelines.gen_2dhoi.
_GEN_KEY_RENAMES = {
    ("scale", "max_iterations"): "max_scale_iterations",
    ("video", "model_api"): "video_model_api",
}

# Sections whose contents are flattened directly (no prefix).
_GEN_FLAT_SECTIONS = ["paths", "simulation", "scale", "rendering", "video", "chatgpt"]

# Top-level scalar keys that should be included as-is.
_GEN_TOP_LEVEL_KEYS = [
    "character",
    "character_usd",
    "character_scale",
    "character_dir",
    "texture_dir",
    "results_dir",
    "character_init_pose_file",
    "object_config",
    "verbose",
    "skip_step1",
    "skip_step2",
    "skip_step3",
    "skip_step4",
    "skip_done",
]


def load_gen_config(cfg_path):
    """Load a 2D-generation pipeline YAML and return a flat dict suitable
    for ``argparse.set_defaults``.

    Handles the key renames defined in ``_GEN_KEY_RENAMES`` so the flat
    dict keys match the argparse attribute names used by the step
    functions (e.g. ``scale.max_iterations`` -> ``max_scale_iterations``).
    """
    cfg = load_config(cfg_path)
    flat = {}

    for key in _GEN_TOP_LEVEL_KEYS:
        if key in cfg:
            flat[key] = cfg[key]

    for section in _GEN_FLAT_SECTIONS:
        for k, v in cfg.get(section, {}).items():
            renamed = _GEN_KEY_RENAMES.get((section, k))
            flat[renamed or k] = v

    return flat


# -- 4D Reconstruction config -------------------------------------------------

# Sections that are flattened into the argparse namespace for the
# reconstruction pipeline.  "optimization" is intentionally excluded —
# it is passed to HOIOptimizer as a structured dict.
_RECON_FLAT_SECTIONS = [
    "paths",
    "human_model",
    "obj_pose_tracking",
    # NOTE: "filtering" and "post_processing" are intentionally NOT flattened.
    # They are passed as nested dicts to filter_hoi_result() and post_process_hoi_result().
    "pipeline",
]


def load_recon_config(cfg_path=None):
    """Load the unified reconstruction config and return both the full
    structured dict and a flat dict suitable for ``argparse.set_defaults``.

    Returns:
        cfg (dict):  Full YAML dict with section keys preserved.
        flat (dict): Merged key-value pairs from the sections listed in
                     ``_RECON_FLAT_SECTIONS``, ready to be used as argparse
                     defaults (CLI flags override these).
    """
    cfg = load_config(cfg_path or DEFAULT_RECON_CONFIG)
    flat = {}
    for section in _RECON_FLAT_SECTIONS:
        flat.update(cfg.get(section, {}))
    return cfg, flat
