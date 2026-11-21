"""Read native numeric checkpoints without executing arbitrary pickle reducers.

Native fitting uses dictionaries, lists and NumPy arrays in standard pickle or
joblib's array format. Only their reconstruction primitives are supported here;
Python objects, object arrays and legacy external-array references are rejected.
This is intentionally narrower than general ``joblib.load``.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np


def load_native_pickle(path):
    """Load an existing numeric pickle/joblib checkpoint, including compression.

    The source is opened once and never reopened during deserialization. A
    caller-owned hash check is still required to establish provenance; this
    reader only prevents executable payloads if a writable source is replaced.
    """
    from joblib import numpy_pickle

    class NumericArrayWrapper(numpy_pickle.NumpyArrayWrapper):
        def read_array(self, unpickler, *args, **kwargs):
            # Joblib delegates object-array contents to unrestricted pickle.load.
            # Reject them before reaching that separate nested unpickler.
            if np.dtype(self.dtype).hasobject:
                raise ValueError("Object arrays are not supported in native checkpoints")
            if self.subclass is not np.ndarray:
                raise ValueError("Native checkpoints require ordinary NumPy arrays")
            return super().read_array(unpickler, *args, **kwargs)

    class NumericUnpickler(numpy_pickle.NumpyUnpickler):
        def find_class(self, module, name):
            if (module, name) == ("_codecs", "encode"):
                return _latin1_bytes
            if (module, name) == ("joblib.numpy_pickle", "NumpyArrayWrapper"):
                return NumericArrayWrapper
            allowed = (
                (module == "numpy" and name in {"ndarray", "dtype"})
                or (
                    module in {"numpy.core.multiarray", "numpy._core.multiarray"}
                    and name in {"_reconstruct", "scalar"}
                )
                or (
                    module in {"numpy.core.numeric", "numpy._core.numeric"}
                    and name == "_frombuffer"
                )
            )
            if not allowed:
                raise ValueError(f"Unsupported native checkpoint global: {module}.{name}")
            return getattr(importlib.import_module(module), name)

    # Joblib 1.5 renamed its compression context and added a byte-order option.
    # Use the installed reader's signature, never fall back to unrestricted load.
    reader = getattr(numpy_pickle, "_validate_fileobject_and_memmap", None)
    if reader is None:
        reader = numpy_pickle._read_fileobject
    kwargs = {"mmap_mode": None}
    if "ensure_native_byte_order" in inspect.signature(NumericUnpickler).parameters:
        kwargs["ensure_native_byte_order"] = True
    path = Path(path)
    with path.open("rb") as stream:
        with reader(stream, str(path), mmap_mode=None) as decoded:
            handle = decoded[0] if isinstance(decoded, tuple) else decoded
            if isinstance(handle, str):
                raise ValueError("Legacy external-array checkpoints need explicit migration")
            value = NumericUnpickler(str(path), handle, **kwargs).load()
    _validate_value(value, set())
    return value


def _validate_value(value, seen):
    """Reject object arrays even when encoded by ordinary NumPy pickling."""
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        if type(value) is not np.ndarray or value.dtype.hasobject:
            raise ValueError("Object arrays are not supported in native checkpoints")
    elif isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise ValueError("Object scalars are not supported in native checkpoints")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_value(key, seen)
            _validate_value(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_value(item, seen)
    elif value is not None and type(value) not in (str, bytes, bool, int, float, complex):
        raise ValueError(f"Unsupported native checkpoint value: {type(value).__name__}")


def _latin1_bytes(value, encoding):
    """Protocol 2 represents array bytes through a literal Latin-1 string."""
    if type(value) is not str or encoding != "latin1":
        raise ValueError("Only literal Latin-1 byte reconstruction is supported")
    return value.encode("latin1")
