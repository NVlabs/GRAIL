"""Exact-query reuse for repeated original hand surface projections.

This changes neither nearest-triangle selection nor its tie breaking. Keys bind
the full float64 query coordinates and original mesh vertices/faces. Callers
must opt in with a bounded cache, which can be installed on contact_patches only.
"""
from collections import OrderedDict
import hashlib
import numpy as np


class ExactSurfaceQueryCache:
    def __init__(self, query, max_entries=256):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        self.query, self.max_entries = query, max_entries
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, points, mesh):
        p = np.asarray(points, dtype=np.float64)
        digest = hashlib.sha256()
        for a in (p, mesh.vertices, mesh.faces):
            digest.update(str((a.dtype.str, a.shape)).encode())
            digest.update(a.tobytes())
        key = digest.digest()
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return {k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in self.entries[key].items()}
        self.misses += 1
        answer = self.query(p, mesh)
        self.entries[key] = {k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in answer.items()}
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return answer

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, entries=len(self.entries),
                    max_entries=self.max_entries, exact_geometry_and_query_bound=True)
