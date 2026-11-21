"""Sampled free-hand clearance against the bound, fixed native world scene.

The object-relative hand changes coordinates at every source time; the table
stays in native world metres. Original packed triangles are queried directly.
Signed distances guide optimization and do not certify whole-mesh clearance.
"""
from .static_scene import static_scene_fingerprint, static_scene_payload


class HandSceneClearance:
    def __init__(self, native, device):
        manifest, arrays = static_scene_payload(native)
        self.queries = []
        self.metadata = dict(
            status=manifest['status'], world_frame=manifest['world_frame'],
            geometry_sha256=static_scene_fingerprint(manifest, arrays),
            coordinate_rule='Object-local metric hand points -> matching-time native world R @ p + t; static scene stays fixed.',
            scope='Original packed scene triangles and sampled whole-hand surface points; all configured phases.',
            signed_distance_is_heuristic=True, full_mesh_collision_certified=False,
            motion_accepted=False, meshes=[],
        )
        if manifest['meshes']:
            from .repair_tensor import WarpSignedDistance
            for mesh in manifest['meshes']:
                vertices, faces = arrays[mesh['vertices_key']], arrays[mesh['faces_key']]
                query = WarpSignedDistance(vertices, faces, device=device)
                self.queries.append(query)
                self.metadata['meshes'].append(dict(
                    name=mesh['name'], role=mesh['role'], vertices=len(vertices),
                    triangles=len(faces), source=mesh['source'], query=query.metadata()))
        self.enabled = bool(self.queries)
        self.metadata['enabled'] = self.enabled

    def penalty(self, points_object, object_R, object_t, margin):
        """Return per-pose loss and minimum signed distance, preserving gradients.

        Neither phases nor contact masks gate this loss. It also accepts actual
        interpolated object poses for between-frame collision collocation.
        No geometry/scale transform is inferred from the optimization candidate.
        """
        import torch
        if points_object.ndim != 3 or points_object.shape[-1] != 3:
            raise ValueError('Scene clearance requires batched BxNx3 hand points')
        count = points_object.shape[0]
        if object_R.shape != (count, 3, 3) or object_t.shape != (count, 3):
            raise ValueError('Scene clearance requires matching-time object poses')
        loss = points_object.sum(dim=(1, 2))*0
        minimum = torch.full((count,), float('inf'), dtype=points_object.dtype,
                             device=points_object.device)
        if not self.enabled:
            return loss, minimum
        world = torch.einsum('bij,bnj->bni', object_R, points_object) + object_t[:, None]
        for query in self.queries:
            signed = query.distance(world)
            violation = torch.relu(margin-signed)
            loss = loss + 400*violation.square().mean(1) + 20*violation.max(1).values.square()
            minimum = torch.minimum(minimum, signed.min(1).values)
        return loss, minimum
