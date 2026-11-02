"""Metric CAD-PnP proposals with separate geometric availability and quality.

A low inlier fraction does not erase an otherwise computable pose. Consumers
must explicitly choose whether low-support proposals are acceptable for their
use case. These flags do not certify physical validity or 6D pose accuracy.
"""

import cv2
import numpy as np


def estimate_pose_proposals(
    object_points,
    pixels_native,
    K,
    initial_pose,
    visibility_logits,
    native_hw,
    min_inliers=12,
    min_inlier_fraction=0.30,
    reprojection_error_px=10.0,
    seed=0,
    max_median_residual_px=10.0,
):
    """Preserve finite geometric proposals and report the original support gate.

    Uses the original EPNP-RANSAC settings (300 iterations, 0.999 confidence) and
    LM on its inliers. No reference poses after frame0, time smoothing, repeated
    threshold search or invalid filling. ``quality_pass`` is a deterministic
    inlier-support check, not calibrated confidence. Unavailable poses stay NaN.
    ``reprojection_error_px`` sets RANSAC support; ``max_median_residual_px``
    independently bounds the post-LM median inlier residual.
    """
    obj, uv, K, initial = (
        np.asarray(x, dtype=np.float64) for x in [object_points, pixels_native, K, initial_pose]
    )
    logits = np.asarray(visibility_logits)
    if obj.ndim != 2 or obj.shape[1:] != (3,) or uv.ndim != 3 or uv.shape[1:] != (len(obj), 2):
        raise ValueError("Expected object points [N,3] and pixels [T,N,2]")
    if logits.shape != uv.shape[:2] or not len(uv):
        raise ValueError("Visibility shape/timeline mismatch")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    if (
        min_inliers < 4
        or not 0 <= min_inlier_fraction <= 1
        or not np.isfinite([reprojection_error_px, max_median_residual_px]).all()
        or min(reprojection_error_px, max_median_residual_px) <= 0
    ):
        raise ValueError("Invalid PnP thresholds")
    h, w = native_hw
    if min(h, w) <= 0:
        raise ValueError("Invalid image dimensions")

    def proper(T):
        return (
            T.shape == (4, 4)
            and np.isfinite(T).all()
            and np.allclose(T[3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            and np.allclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)
        )

    if not proper(initial):
        raise ValueError("Initial pose is not a finite proper SE3")
    count = len(uv)
    poses = np.full((count, 4, 4), np.nan)
    available, quality = np.zeros(count, bool), np.zeros(count, bool)
    poses[0], available[0] = initial, True
    # Supplied initialization is an explicit exception, not inferred quality.
    quality[0] = True
    rows = [
        {
            "frame": 0,
            "pose_available": True,
            "quality_pass": True,
            "status": "provided_initial_pose",
            "inlier_ratio": None,
            "quality_basis": "supplied initialization, excluded from tracking aggregates",
        }
    ]
    visible = logits > 0
    for t in range(1, count):
        good = np.isfinite(obj).all(1) & np.isfinite(uv[t]).all(1) & visible[t] & visible[0]
        good &= (uv[t, :, 0] >= 0) & (uv[t, :, 0] < w) & (uv[t, :, 1] >= 0) & (uv[t, :, 1] < h)
        row = {
            "frame": t,
            "pose_available": False,
            "quality_pass": False,
            "status": "unavailable",
            "correspondences": int(good.sum()),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }
        try:
            x, y = np.ascontiguousarray(obj[good]), np.ascontiguousarray(uv[t, good])
            if len(x) < min_inliers:
                raise ValueError("Insufficient visible correspondences")
            sv = np.linalg.svd(x - x.mean(0), compute_uv=False)
            if sv[0] < 1e-9 or sv[1] / sv[0] < 1e-4:
                raise ValueError("Collinear/collapsed CAD anchors")
            cv2.setRNGSeed(int(seed + t))
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                x,
                y,
                K,
                None,
                iterationsCount=300,
                reprojectionError=reprojection_error_px,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            nin = 0 if inliers is None else len(inliers)
            row.update(ransac_solved=bool(ok), inliers=nin, inlier_ratio=nin / len(x))
            if not ok or inliers is None or nin < min_inliers:
                raise ValueError("PnP failed or has fewer than the minimum absolute inliers")
            ix = inliers[:, 0]
            rvec, tvec = cv2.solvePnPRefineLM(x[ix], y[ix], K, None, rvec, tvec)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
            camera = x[ix] @ T[:3, :3].T + T[:3, 3]
            projected = camera @ K.T
            residual = np.linalg.norm(projected[:, :2] / projected[:, 2:] - y[ix], axis=1)
            if (
                not proper(T)
                or not (camera[:, 2] > 1e-6).all()
                or not np.isfinite(residual).all()
                or np.median(residual) > max_median_residual_px
            ):
                raise ValueError(
                    "Nonphysical numeric transform, nonpositive inlier depth or failed reprojection check"
                )
            poses[t], available[t] = T, True
            quality[t] = nin >= min_inlier_fraction * len(x)
            camera_all = obj @ T[:3, :3].T + T[:3, 3]
            row.update(
                pose_available=True,
                quality_pass=bool(quality[t]),
                status="estimated" if quality[t] else "estimated_low_inlier_support",
                reprojection_median_px=float(np.median(residual)),
                anchor_positive_depth_fraction=float(np.mean(camera_all[:, 2] > 1e-6)),
                planar_anchor_ratio=float(sv[2] / sv[0]),
            )
        except (ValueError, cv2.error) as exc:
            row["reason"] = str(exc)
        rows.append(row)
    return {
        "T_camera_object": poses,
        "pose_available": available,
        "quality_pass": quality,
        "per_frame": rows,
        "method": "mose3_cad_pnp_proposals_with_separate_support_flag",
    }
