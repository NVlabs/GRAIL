#!/usr/bin/env python3
"""Training child: run the vendored SONIC trainer with the manipulation runtime adapters.

Launched by ``python -m grail.tracking.manipulation.train`` inside the ``sonic``
environment, with this directory and the vendored ``imports/SONIC-manipulation``
tree on ``PYTHONPATH``. The vendored source stays unchanged; only process-level
adapters are installed before the upstream entry point runs:

* Kit/TBB/PhysX thread pools sized to the run's CPU budget,
* float32 FK inputs for archived float64 motion samples,
* explicit PhysX GPU buffer capacities and the absolute PPO update counter
  (``env2048_adapter``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys

from kit_runtime import defer_fk_float32_until_import, install_thread_limit


def main() -> None:
    repo = Path(os.environ["CONTACTFIRST_REPO"]).resolve(strict=True)
    entry = repo / "gear_sonic/train_agent_trl.py"
    if not entry.is_file():
        raise ValueError("CONTACTFIRST_REPO does not contain the vendored SONIC trainer")
    sys.path[:0] = [str(repo)]
    from shader_cache import prepare_from_environment

    install_thread_limit(int(os.environ["CONTACTFIRST_CPU_THREADS"]), prepare_from_environment())
    defer_fk_float32_until_import()
    from env2048_adapter import install

    config = json.loads(Path(os.environ["CONTACTFIRST_RUN_CONFIG_FILE"]).read_text())
    install(config)
    # Preserve upstream __file__, Hydra's relative config path and all CLI overrides.
    sys.argv = [str(entry), *sys.argv[1:]]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
