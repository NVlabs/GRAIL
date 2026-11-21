"""Bound TopoRetarget/DexOAK-inspired graph terms for native Dex3 proposals.

This integration adds no acceptance rules and alters no native geometry. It
does not implement the papers' complete solvers or DexOAK's learned network.
The standalone experimental caller chooses whether and how to weight these
objectives; the default production proposal route remains independent.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .grasp_interaction_graph import build_bone_reference, build_interaction_graph


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _weight(options, key, default):
    value = options.get(key, default)
    if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return float(value)


def _point_mapping(side, wrist_name):
    human_prefix = side.capitalize()+"Hand"
    pairs = [(human_prefix, wrist_name)]
    pairs.extend((human_prefix+"Thumb"+suffix, side+"_hand_thumb_"+link)
                 for suffix, link in [("1", "0_link"), ("2", "1_link"),
                                      ("3", "2_link"), ("End", "finger_tip")])
    for finger in ["Index", "Middle"]:
        pairs.extend((human_prefix+finger+suffix, side+"_hand_"+finger.lower()+"_"+link)
                     for suffix, link in [("2", "0_link"), ("3", "1_link"), ("End", "finger_tip")])
    return pairs


@dataclass
class PaperObjective:
    """Once-bound references; evaluate receives the proposal's actual free wrist."""

    graph: object
    bone: object
    body_ids: object
    graph_ids: object
    wrist: int
    graph_weight: float
    direction_weight: float
    metadata: dict

    def evaluate(self, rr, pp, rots, poss):
        """Return [batch] weighted loss; graph coordinates are object-relative.

        ``rots/poss`` are original full-robot FK at current finger angles.
        ``rr/pp`` are the candidate free-wrist object-relative rotation/position.
        The original wrist-relative robot node offsets are reposed by rr/pp.
        """
        import torch

        offset = poss[:, self.body_ids]-poss[:, self.wrist, None]
        local = torch.einsum("bji,bnj->bni", rots[:, self.wrist], offset)
        target = torch.einsum("bij,bnj->bni", rr, local)+pp[:, None]
        result = self.graph_weight*self.graph.loss(target[:, self.graph_ids])
        if self.bone is not None and self.direction_weight:
            result = result+self.direction_weight*self.bone.loss(target, target_frame=rr)
        return result


def build_paper_objective(ctx, hand_name, hand, frames, fk, options):
    """Bind raw human/source-object data to a batch of native Dex3 proposals.

    Required options: ``method`` (topo/dexoak), ``source_arrays`` NPZ path.
    Optional: graph_weight, direction_weight (bone directions, NOT surface
    normals), graph_decay, object_sample_count, fallback_neighbors, node_set.
    ``node_set`` defaults to all_mapped for Topo and wrist_tips for DexOAK.
    Frame indices may repeat for multistart but must address the exact source.
    """
    import torch

    allowed = {"method", "source_arrays", "graph_weight", "direction_weight", "graph_decay",
               "object_sample_count", "fallback_neighbors", "node_set"}
    if not isinstance(options, dict) or set(options)-allowed:
        raise ValueError("unknown paper objective option")
    method = options.get("method")
    if method not in ("topo", "dexoak"):
        raise ValueError("paper objective method must be topo or dexoak")
    if hand_name not in ("left", "right"):
        raise ValueError("paper objective hand_name must explicitly be left or right")
    path = Path(options["source_arrays"]).resolve(strict=True)
    indices = np.asarray(frames)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(ctx.q))):
        raise ValueError("frames must be valid source indices, with repeats allowed")
    with np.load(path, allow_pickle=False) as arrays:
        names = [str(n) for n in arrays["human_joint_names"]]
        human = np.array(arrays["human_joint_positions"], dtype=np.float64)
        rotations = np.array(arrays["human_joint_rotations"], dtype=np.float64)
        for source_key, actual in [("object_R", ctx.R), ("object_trans", ctx.t),
                                   ("object_scale", ctx.native["object_scale"])]:
            if not np.array_equal(arrays[source_key], actual):
                raise ValueError(f"source-object binding mismatch: {source_key}")
    if (human.shape != (len(ctx.q), len(names), 3)
            or rotations.shape != (len(ctx.q), len(names), 3, 3)
            or not np.isfinite(human).all() or not np.isfinite(rotations).all()
            or len(set(names)) != len(names)):
        raise ValueError("human arrays must be finite and aligned to the native source frames")
    wrist = int(hand["wrist"])
    wrist_name = ctx.model.body(wrist).name
    mapping = _point_mapping(hand_name, wrist_name)
    human_ids, body_ids = [], []
    for human_name, robot_name in mapping:
        if human_name not in names:
            raise ValueError(f"missing mapped human point: {human_name}")
        human_ids.append(names.index(human_name))
        try:
            body = int(ctx.model.body(robot_name).id)
        except KeyError:
            raise ValueError(f"missing mapped original robot body: {robot_name}") from None
        ancestor = body
        while ancestor > 0 and ancestor != wrist:
            ancestor = int(ctx.model.body_parentid[ancestor])
        if ancestor != wrist:
            raise ValueError(f"mapped robot point is not a descendant of selected wrist: {robot_name}")
        body_ids.append(body)
    if len(set(body_ids)) != len(body_ids):
        raise ValueError("robot mapping must use distinct native body origins")
    full_human = np.einsum("bji,bnj->bni", ctx.R[indices],
                           human[indices][:, human_ids]-ctx.t[indices, None])
    node_set = options.get("node_set", "all_mapped" if method == "topo" else "wrist_tips")
    if node_set not in ("all_mapped", "wrist_tips"):
        raise ValueError("node_set must be all_mapped or wrist_tips")
    graph_ids = list(range(len(mapping))) if node_set == "all_mapped" else [0, 4, 7, 10]
    count = options.get("object_sample_count", 50 if method == "topo" else 64)
    if type(count) is not int or count < 1 or count > len(ctx.object.faces):
        raise ValueError("object_sample_count must be a positive integer within the original face count")
    face_ids = np.linspace(0, len(ctx.object.faces)-1, count, dtype=np.int64)
    object_points = np.asarray(ctx.object.triangles[face_ids].mean(axis=1), dtype=np.float64)
    decay = _weight(options, "graph_decay", 30 if method == "topo" else 0)
    graph_weight = _weight(options, "graph_weight", 500 if method == "topo" else 1)
    direction_weight = _weight(options, "direction_weight", 0)
    graph = build_interaction_graph(full_human[:, graph_ids], object_points,
                                    decay=decay, fallback_neighbors=options.get("fallback_neighbors", 20))
    bone = None
    chains = [[0, 1, 2, 3, 4], [0, 5, 6, 7], [0, 8, 9, 10]]
    if direction_weight:
        source_frame = np.einsum("bji,bjk->bik", ctx.R[indices],
                                  rotations[indices, human_ids[0]])
        bone = build_bone_reference(full_human, chains, source_frame=source_frame).to_torch(
            fk.device, dtype=fk.dtype)
    metadata = dict(
        method=method+"_inspired_dex3_objective", official_reproduction=False,
        source_binding=dict(path=str(path), sha256=_hash(path), exact_object_R_t_scale=True,
                            human_scaling=1.0, source_frame_indices=indices.tolist(),
                            native_binding=dict(ctx.binding)),
        mapping=[dict(human=h, robot_body=r, human_index=i, robot_body_id=j,
                      graph_node=k in graph_ids)
                 for k, ((h, r), i, j) in enumerate(zip(mapping, human_ids, body_ids))],
        mapping_scope="Pose prior uses thumb/index/middle anatomy; existing contact-role union is unchanged.",
        omitted_human_fingers=["ring", "pinky"], node_set=node_set,
        graph=graph.metadata, graph_weight=graph_weight, direction_weight=direction_weight,
        direction_term="adjacent unit-bone differences in each wrist frame; no surface-normal alignment",
        direction_chains=chains if direction_weight else [],
        object_sampling=dict(method="deterministic uniform original-face centroid stride",
                             original_face_ids=face_ids.tolist(), original_geometry_preserved=True,
                             points_sha256=hashlib.sha256(object_points.astype("<f8").tobytes()).hexdigest()),
        coordinates="object-local metric points; source positions unscaled",
        source_urls=["https://arxiv.org/html/2606.16272v2", "https://arxiv.org/html/2610.26762v1"],
        code_sha256={name: _hash(Path(__file__).with_name(name))
                     for name in ("grasp_paper_objectives.py", "grasp_interaction_graph.py")},
        scope="Additional experimental optimization loss; no candidate is accepted here.")
    return PaperObjective(graph.to_torch(fk.device, dtype=fk.dtype), bone,
                          torch.tensor(body_ids, device=fk.device),
                          torch.tensor(graph_ids, device=fk.device), wrist,
                          graph_weight, direction_weight, metadata)
