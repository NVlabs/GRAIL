"""Detached current-object exterior sampling for optional outline eligibility.

This is a finite pixel-grid predicate, not a continuous silhouette certificate.
Original optimizer correspondences remain their original edge/face/barycentric
points; the sampled exterior witnesses below are separate diagnostics.
"""
import time

import numpy as np
import torch
from scipy.ndimage import binary_propagation
from scipy.spatial import cKDTree


def exterior_midpoints(hit):
    """Full-object hit/miss brackets whose miss is 4-connected to the canvas."""
    if hit.ndim != 2 or hit.dtype != np.bool_:
        raise ValueError('Expected a two-dimensional boolean object hit mask')
    miss = ~hit
    seed = np.zeros_like(hit)
    seed[0] = miss[0]; seed[-1] = miss[-1]
    seed[:, 0] = miss[:, 0]; seed[:, -1] = miss[:, -1]
    outside = binary_propagation(seed, mask=miss)
    ids = np.arange(hit.size).reshape(hit.shape)
    inside_ids, outside_ids = [], []
    for a, b in [(ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])]:
        ha, hb = hit.ravel()[a], hit.ravel()[b]
        keep = ha != hb
        inner = np.where(ha, a, b)[keep]
        outer = np.where(ha, b, a)[keep]
        exterior = outside.ravel()[outer]
        inside_ids.append(inner[exterior]); outside_ids.append(outer[exterior])
    inner, outer = np.concatenate(inside_ids), np.concatenate(outside_ids)
    width = hit.shape[1]
    xy = lambda i: np.stack([i % width + .5, i // width + .5], -1) / 2
    inside_xy, outside_xy = xy(inner), xy(outer)
    return (inside_xy + outside_xy) * .5, inner, outer, inside_xy, outside_xy, outside


class ExteriorFrame:
    def __init__(self, xy, inner, outer, inside_xy, outside_xy, faces, bary, axial_depth, audit):
        self.xy, self.inner, self.outer = xy, inner, outer
        self.inside_xy, self.outside_xy = inside_xy, outside_xy
        self.faces, self.bary, self.audit = faces, bary, audit
        self.axial_depth = axial_depth
        if not len(xy):
            raise ValueError(f'No sampled canvas-connected object exterior at frame {audit["frame"]}')
        self.tree = cKDTree(xy)
        audit['tree_builds'] = 1

    def distances(self, uv):
        points = uv.detach().cpu().numpy().astype(np.float64)
        if points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ValueError('Nonfinite or malformed exterior query UV')
        distance, _ = self.tree.query(points.reshape(-1, 2), k=1)
        self.audit['tree_query_calls'] += 1
        self.audit['tree_query_points'] += int(distance.size)
        return distance.reshape(points.shape[:-1])

    def selected_witness(self, uv):
        point = uv.detach().cpu().numpy().astype(np.float64)
        distance, _ = self.tree.query(point, k=1)
        self.audit['selected_witness_queries'] += 1
        # Stable original sample ordering for equal/near ties; the minimum
        # distance, not this ownership convention, defines eligibility.
        ties = self.tree.query_ball_point(point, r=float(distance) + 1e-10)
        self.audit['selected_tie_radius_queries'] += 1
        ties.sort(key=lambda i: (int(self.inner[i]), int(self.outer[i]), int(self.faces[i])))
        records = [dict(midpoint_xy=self.xy[i].tolist(), inside_sample_xy=self.inside_xy[i].tolist(),
                        outside_sample_xy=self.outside_xy[i].tolist(),
                        inside_original_object_face_id=int(self.faces[i]),
                        inside_original_barycentric=self.bary[i].tolist(),
                        inside_camera_axial_depth_m=float(self.axial_depth[i]),
                        inside_sample_id=int(self.inner[i]), outside_sample_id=int(self.outer[i])) for i in ties]
        return dict(distance_px=float(distance), nearest_sample_ties=records,
                    sample_ownership_is_not_optimizer_correspondence=True)


class CurrentObjectExterior:
    """Cache calibration only; build a fresh live raster/tree on every call."""
    def __init__(self):
        self._camera_key = None
        self._rasterizer = None
        self.last_audit = None

    @torch.no_grad()
    def build(self, local_vertices, faces, rotation, translation, camera_R,
              camera_t, focal, image_center, frame):
        self.last_audit = None
        from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
        from pytorch3d.structures import Meshes

        size = (image_center.detach().cpu().numpy() * 2).astype(np.float64)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 1) or not np.equal(size, np.round(size)).all():
            raise ValueError('Exterior sampling requires positive integer image dimensions')
        width, height = map(int, size)
        if not torch.isfinite(focal).all() or float(focal) <= 0:
            raise ValueError('Exterior sampling requires a positive finite focal length')
        world = local_vertices.detach() @ rotation.detach().T + translation.detach()
        camera = (world - camera_t) @ camera_R
        if not torch.isfinite(camera).all() or torch.any(camera[:, 2] >= -1e-6):
            raise ValueError('Full object crosses the calibrated camera plane or is nonfinite')
        key = (str(world.device), str(world.dtype), width, height, int(len(faces)),
               tuple(camera_R.detach().cpu().flatten().tolist()),
               tuple(camera_t.detach().cpu().tolist()), float(focal))
        if key != self._camera_key:
            change = torch.diag(camera_R.new_tensor([-1., 1., -1.]))
            row_R = camera_R @ change
            row_T = -camera_t @ row_R
            p3d = PerspectiveCameras(R=row_R[None], T=row_T[None],
                focal_length=focal.reshape(1, 1).expand(1, 2) * 2,
                principal_point=((float(width), float(height)),),
                image_size=((2 * height, 2 * width),), in_ndc=False, device=world.device)
            settings = RasterizationSettings(image_size=(2 * height, 2 * width),
                blur_radius=0., faces_per_pixel=1, perspective_correct=True,
                cull_backfaces=False, clip_barycentric_coords=False,
                bin_size=None if world.is_cuda else 0, max_faces_per_bin=max(10000, len(faces)))
            self._rasterizer = MeshRasterizer(cameras=p3d, raster_settings=settings)
            self._camera_key = key
        audit = dict(frame=int(frame), sampling_scale=2, connectivity=4,
            max_distance_px=.5, raster_attempts=1, raster_returns=0, hit_transfer_completed=False,
            sample_capacity=4 * height * width, tree_builds=0, tree_query_calls=0,
            tree_query_points=0, selected_witness_queries=0, selected_tie_radius_queries=0,
            policy='current_full_object_canvas_connected_sampled_exterior_v1')
        self.last_audit = audit
        start = time.perf_counter()
        fragments = self._rasterizer(Meshes(verts=[world], faces=[faces]))
        audit['raster_returns'] = 1
        hit = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy()
        audit['hit_transfer_completed'] = True
        audit['raster_and_hit_transfer_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        xy, inner, outer, inside_xy, outside_xy, outside = exterior_midpoints(hit)
        index = torch.as_tensor(inner.copy(), device=world.device, dtype=torch.long)
        owner = fragments.pix_to_face.reshape(-1)[index].cpu().numpy()
        bary = fragments.bary_coords.reshape(-1, 3)[index].cpu().numpy()
        axial_depth = fragments.zbuf.reshape(-1)[index].cpu().numpy()
        audit.update(exterior_midpoints=int(len(xy)), hit_samples=int(hit.sum()),
            outside_miss_samples=int(outside.sum()), enclosed_miss_samples=int((~hit & ~outside).sum()),
            object_touches_canvas=bool(hit[0].any() or hit[-1].any() or hit[:, 0].any() or hit[:, -1].any()))
        result = ExteriorFrame(xy, inner, outer, inside_xy, outside_xy, owner, bary, axial_depth, audit)
        audit['exterior_tree_and_witness_transfer_seconds'] = time.perf_counter() - start
        return result
