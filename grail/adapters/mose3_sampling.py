"""First-frame-only query selection for dense MoSE3 tracking adapters.

The caller must first restrict candidates to the visible CAD surface and object
mask (excluding the human mask). This module cannot infer those conditions from
pixel coordinates. It then applies the existing visibility-logit > 0 gate and
subsamples the eligible pixels without using later tracks or reference poses.

Sampling cached trajectories changes the CAD-PnP input, not MoSE3's dense model
inference cost. Pixel-space diversity does not guarantee accurate correspondences
or non-degenerate 3D geometry; those remain separate pose-estimation checks.
"""

import numpy as np

MIN_QUERY_POINTS = 12


def _eligible_indices(pixels_xy, visibility_logits0, eligible_mask):
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_xy must have shape (N, 2)")
    visibility = np.asarray(visibility_logits0, dtype=np.float64)
    if visibility.shape != (len(pixels),):
        raise ValueError("visibility_logits0 must have shape (N,); pass only the first frame")
    eligible = np.isfinite(pixels).all(axis=1) & np.isfinite(visibility) & (visibility > 0)
    if eligible_mask is not None:
        mask = np.asarray(eligible_mask)
        if mask.shape != (len(pixels),) or mask.dtype.kind != "b":
            raise ValueError("eligible_mask must be a Boolean array with shape (N,)")
        eligible &= mask
    indices = np.flatnonzero(eligible)
    # Keep the first eligible occurrence, preserving original-index tie order.
    _, first = np.unique(pixels[indices], axis=0, return_index=True)
    distinct = indices[np.sort(first)]
    if len(distinct) < MIN_QUERY_POINTS:
        raise ValueError(
            f"Need at least {MIN_QUERY_POINTS} distinct first-frame-visible finite pixels; "
            f"got {len(distinct)} from {len(indices)} eligible candidates"
        )
    return pixels, distinct, len(indices)


def select_query_indices(
    pixels_xy, visibility_logits0, max_points, method="farthest", eligible_mask=None
):
    """Return distinct original-pool indices chosen using first-frame data only.

    ``farthest`` uses Euclidean distance in unscaled image pixels, preserving
    aspect ratio. The seed is the eligible pixel nearest their centroid. Ties
    choose the lowest original index. Its order is independent of ``max_points``,
    so a smaller selection is a prefix of a larger one on the same pool.

    ``raster_uniform`` sorts by y, then x and takes evenly spaced raster indices.
    This is a point-count control, not a spatial-diversity guarantee.

    Visibility values are logits, not probabilities. The strict ``> 0`` gate
    matches the existing bridge. If fewer than ``max_points`` distinct eligible
    pixels exist, all are returned; fewer than 12 raises ``ValueError``. No input
    is modified. Finite/visible CAD constraints belong in ``eligible_mask``.
    """
    if isinstance(max_points, (bool, np.bool_)) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be an integer")
    if max_points < MIN_QUERY_POINTS:
        raise ValueError(f"max_points must be at least {MIN_QUERY_POINTS}")
    if method not in ("farthest", "raster_uniform"):
        raise ValueError("method must be 'farthest' or 'raster_uniform'")
    pixels, eligible, _ = _eligible_indices(pixels_xy, visibility_logits0, eligible_mask)
    count = min(int(max_points), len(eligible))
    xy = pixels[eligible]
    if method == "raster_uniform":
        order = np.lexsort((eligible, xy[:, 0], xy[:, 1]))
        positions = np.rint(np.linspace(0, len(order) - 1, count)).astype(np.int64)
        return eligible[order[positions]]

    centroid = xy.mean(axis=0)
    next_index = int(np.argmin(np.sum((xy - centroid) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    remaining_distance2 = np.full(len(xy), np.inf)
    for step in range(count):
        selected[step] = next_index
        distance2 = np.sum((xy - xy[next_index]) ** 2, axis=1)
        np.minimum(remaining_distance2, distance2, out=remaining_distance2)
        remaining_distance2[selected[: step + 1]] = -np.inf
        next_index = int(np.argmax(remaining_distance2))
    return eligible[selected]


def query_sampling_diagnostics(
    pixels_xy, visibility_logits0, selected_indices, eligible_mask=None, object_points=None
):
    """Describe coverage of the eligible candidate pool; do not rate pose accuracy.

    Fill radius is the maximum eligible-pixel distance to a selected pixel, in
    image pixels. Optional object points must be the corresponding first-frame
    CAD anchors (N, 3); rank describes the selected anchors, not unseen surfaces.
    """
    pixels, eligible, before_dedup = _eligible_indices(pixels_xy, visibility_logits0, eligible_mask)
    selected = np.asarray(selected_indices)
    if selected.ndim != 1 or selected.dtype.kind not in "iu":
        raise ValueError("selected_indices must be a one-dimensional integer array")
    if len(selected) < MIN_QUERY_POINTS or len(np.unique(selected)) != len(selected):
        raise ValueError("selected_indices must contain at least 12 unique indices")
    if not np.isin(selected, eligible).all():
        raise ValueError("selected_indices includes an ineligible or duplicate-pixel index")
    xy = pixels[selected]
    min_distance2 = np.full(len(eligible), np.inf)
    for point in xy:
        np.minimum(
            min_distance2, np.sum((pixels[eligible] - point) ** 2, axis=1), out=min_distance2
        )
    result = {
        "candidate_count": len(pixels),
        "eligible_before_dedup_count": before_dedup,
        "eligible_count": len(eligible),
        "duplicate_pixel_count": before_dedup - len(eligible),
        "selected_count": len(selected),
        "fill_radius_pixels": float(np.sqrt(min_distance2.max())),
        "mean_nearest_selected_pixels": float(np.sqrt(min_distance2).mean()),
        "selected_pixel_rank": int(np.linalg.matrix_rank(xy - xy.mean(axis=0))),
        "selection_frame": 0,
        "visibility_logit_threshold": 0.0,
        "visibility_comparison": ">",
    }
    if object_points is not None:
        points = np.asarray(object_points, dtype=np.float64)
        if points.shape != (len(pixels), 3):
            raise ValueError("object_points must have shape (N, 3)")
        points = points[selected]
        if not np.isfinite(points).all():
            raise ValueError("Selected object_points must be finite; filter using eligible_mask")
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        result["selected_object_rank"] = int(np.linalg.matrix_rank(points - points.mean(axis=0)))
        result["selected_object_singular_values"] = singular.tolist()
    return result
