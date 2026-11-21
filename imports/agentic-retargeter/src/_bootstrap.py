"""Bind the public package name when running tools from a source checkout.

Installed wheels use the normal agentic_retargeter package directory. The
checkout intentionally keeps implementation directly under src/.
"""
import importlib.util
from pathlib import Path
import sys


def bootstrap():
    folder = Path(__file__).resolve().parent
    existing = sys.modules.get("agentic_retargeter")
    if existing is not None:
        if Path(existing.__file__).resolve() != folder / "__init__.py":
            raise RuntimeError("A different agentic_retargeter runtime is already loaded")
        return existing
    spec = importlib.util.spec_from_file_location(
        "agentic_retargeter", folder / "__init__.py", submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module
