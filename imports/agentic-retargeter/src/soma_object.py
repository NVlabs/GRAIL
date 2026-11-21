"""Reference-guided object blocks for contact-aware SOMA sequence solving.

Reference poses are immutable. Soft translation/geodesic-rotation costs retain
the original motion. Optional explicit caps bound correction magnitudes/rates;
omitting them introduces no implicit corridor. Precontact frames share one
optimizable pose, so their stationarity is exact without locking their placement.
Temporal penalties act on actual translations and rotation matrices. Matrix
derivatives are smooth rotation surrogates, not physical angular acceleration
or angular jerk. This module has no Newton/Warp dependency.
"""
from numbers import Real

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

try:
    from .soma_temporal import conditional_target
    from .soma_object_policy import source_object_is_static
except ImportError:  # External SOMA subprocess loads sibling modules directly.
    from soma_temporal import conditional_target
    from soma_object_policy import source_object_is_static


_ORDERS = ("velocity", "acceleration", "jerk")
_INTEGER_FIELDS = {"first_edit_frame", "contact_frame", "stationary_reference_frame", "max_iterations"}
_POSITIVE_FIELDS = {
    "contact_tolerance_m",
    "reference_translation_weight", "reference_rotation_weight",
}
_OPTIONAL_LIMIT_FIELDS = {"max_translation_m", "max_rotation_rad",
                          "max_translation_offset_rate_m_s", "max_rotation_offset_rate_rad_s"}
_TEMPORAL_FIELDS = {f"{kind}_{order}_weight" for kind in ("translation", "rotation") for order in _ORDERS}
_FIELDS = _INTEGER_FIELDS | _POSITIVE_FIELDS | _TEMPORAL_FIELDS
_DEFAULTS = {"stationary_reference_frame": 0, "contact_tolerance_m": .015}


def validate_object_config(value):
    """Resolve direct optimization from declared contact; gaps are diagnostic.

    Legacy first_edit_frame is accepted, but normalized to contact_frame. It no
    longer postpones object updates. Contact onset remains explicit source data.
    """
    required = _FIELDS - set(_DEFAULTS) - {"first_edit_frame"}
    if (not isinstance(value, dict) or required-set(value)
            or set(value)-_FIELDS-_OPTIONAL_LIMIT_FIELDS):
        raise ValueError("object config requires " + ", ".join(sorted(required))
                         + "; optional explicit limits: " + ", ".join(sorted(_OPTIONAL_LIMIT_FIELDS)))
    value = {**_DEFAULTS, "first_edit_frame": value["contact_frame"], **value}
    result = {}
    for field in _INTEGER_FIELDS:
        number = value[field]
        minimum = 1 if field == "max_iterations" else 0
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            raise ValueError(f"object {field} must be an integer >= {minimum}")
        result[field] = number
    result["first_edit_frame"] = result["contact_frame"]
    if result["stationary_reference_frame"] >= max(1, result["contact_frame"]):
        raise ValueError("object stationary reference must precede contact (or be frame 0 for contact at 0)")
    positive = _POSITIVE_FIELDS | (set(value) & _OPTIONAL_LIMIT_FIELDS)
    for field in positive | _TEMPORAL_FIELDS:
        number = value[field]
        if (isinstance(number, bool) or not isinstance(number, Real)
                or not np.isfinite(number) or number < 0
                or (field in positive and number == 0)):
            raise ValueError(f"object {field} must be finite and "
                             + ("positive" if field in positive else "nonnegative"))
        result[field] = float(number)
    if "max_rotation_rad" in result and result["max_rotation_rad"] >= np.pi:
        raise ValueError("object max_rotation_rad must be strictly below pi")
    return result


def _poses(rotations, positions, label):
    rotations = np.asarray(rotations, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
            or rotations.shape != (len(positions), 3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
            # Match the contact compiler's bound upstream rotation precision.
            # Validation must not normalize or overwrite the original arrays.
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-5)):
        raise ValueError(f"{label} must contain finite proper SE(3) poses for every frame")
    return rotations, positions


def _timeline(fps, frames, cfg):
    if (isinstance(fps, bool) or not isinstance(fps, Real)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("object fps must be finite and positive")
    if cfg["contact_frame"] >= frames:
        raise ValueError("object frame settings exceed the source timeline")
    return float(fps)


def resolve_object_policy(reference_R, reference_t, cfg):
    """Choose object variables from the entire immutable source, without an agent.

    Exact pose constancy is intentional: no undocumented tolerance can classify
    small translations or rotation-only motion as static. Static poses are
    copied unchanged; moving sources have one shared precontact pose and
    independent frame variables from declared contact onward.
    """
    cfg = validate_object_config(cfg)
    rotations, positions = _poses(reference_R, reference_t, "reference object")
    _timeline(1., len(positions), cfg)
    static = source_object_is_static(rotations, positions)
    return dict(mode="static_passthrough" if static else "joint_with_shared_precontact",
        optimization_enabled=not static, source_is_static=static,
        precontact_pose_policy="original_static_trajectory" if static else "shared_optimized_pose",
        detection="exact_full_source_translation_and_rotation",
        contact_frame=cfg["contact_frame"], first_edit_frame=cfg["first_edit_frame"],
        stationary_reference_frame=cfg["stationary_reference_frame"],
        contact_quality_gates_object_updates=False)


def _offsets(reference_R, reference_t, rotations, positions):
    return (positions-reference_t,
            Rotation.from_matrix(rotations @ reference_R.transpose(0, 2, 1)).as_rotvec())


def _maximum_norm(values):
    return float(np.linalg.norm(values, axis=-1).max()) if len(values) else 0.


def audit_object_trajectory(reference_R, reference_t, rotations, positions, fps, cfg):
    """Audit precontact constancy and explicitly configured correction caps/rates.

    Invalid shapes/SE(3)/timing raise. Constraint violations return passed=False.
    Tolerances below account only for numerical roundoff, not an edit allowance.
    Passing does not certify small deviations when reference retention is soft.
    """
    cfg = validate_object_config(cfg)
    reference_R, reference_t = _poses(reference_R, reference_t, "reference object")
    rotations, positions = _poses(rotations, positions, "optimized object")
    if positions.shape != reference_t.shape:
        raise ValueError("object/reference timelines differ")
    fps = _timeline(fps, len(positions), cfg)
    dt, dr = _offsets(reference_R, reference_t, rotations, positions)
    stop, anchor = cfg["first_edit_frame"], cfg["stationary_reference_frame"]
    translation_max, rotation_max = _maximum_norm(dt), _maximum_norm(dr)
    translation_rate = _maximum_norm(np.diff(dt, axis=0))*fps
    rotation_rate = _maximum_norm(np.diff(dr, axis=0))*fps
    gates = {
        "stationary_translation": bool(np.all(positions[:stop] == positions[0])),
        "stationary_rotation": bool(np.all(rotations[:stop] == rotations[0])),
    }
    translation_variation = _maximum_norm(positions[:stop]-positions[0])
    rotation_variation = (0. if gates["stationary_rotation"] else _maximum_norm(
        Rotation.from_matrix(rotations[:stop] @ rotations[0].T).as_rotvec()))
    policy = resolve_object_policy(reference_R, reference_t, cfg)
    if not policy["optimization_enabled"]:
        gates["static_original_unchanged"] = bool(np.array_equal(positions, reference_t)
                                                   and np.array_equal(rotations, reference_R))
    for field, gate, measured, tolerance in (
            ("max_translation_m", "translation_cap", translation_max, 1e-9),
            ("max_rotation_rad", "rotation_cap", rotation_max, 1e-9),
            ("max_translation_offset_rate_m_s", "translation_offset_rate", translation_rate, 1e-8),
            ("max_rotation_offset_rate_rad_s", "rotation_offset_rate", rotation_rate, 1e-8)):
        if field in cfg:
            gates[gate] = measured <= cfg[field] + tolerance
    derivatives = {}
    for name, values in (("translation", positions), ("rotation_matrix", rotations.reshape(len(rotations), 9))):
        derivatives[name] = {}
        for order, label in enumerate(_ORDERS, 1):
            difference = np.diff(values, n=order, axis=0)*fps**order
            norms = np.linalg.norm(difference, axis=1)
            derivatives[name][label] = dict(
                rms=float(np.sqrt(np.mean(norms**2))) if len(norms) else None,
                maximum=float(norms.max()) if len(norms) else None)
    angular_speed = Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).magnitude()*fps if len(rotations)>1 else np.empty(0)
    return dict(passed=bool(all(gates.values())), gates=gates, frame_count=len(positions), fps=fps,
        motion_policy=policy,
        reference_policy="soft_reference_with_optional_caps" if set(cfg) & _OPTIONAL_LIMIT_FIELDS else "soft_reference",
        passed_scope="Finite SE(3), stationary precontact poses and explicitly configured caps only; small reference deviation is not established by soft costs",
        max_total_translation_m=translation_max, max_total_rotation_rad=rotation_max,
        max_translation_offset_rate_m_s=translation_rate, max_rotation_offset_rate_rad_s=rotation_rate,
        max_precontact_translation_variation_m=translation_variation,
        max_precontact_rotation_variation_rad=rotation_variation,
        actual_derivatives=derivatives,
        actual_angular_speed_rad_s=dict(rms=float(np.sqrt(np.mean(angular_speed**2))) if len(angular_speed) else None,
                                       maximum=float(angular_speed.max()) if len(angular_speed) else None),
        first_edit_frame=stop, contact_frame=cfg["contact_frame"], stationary_reference_frame=anchor,
        rotation_temporal_representation="finite differences of actual rotation matrices, not angular acceleration/jerk",
        offset_rate_representation="Euclidean differences of immutable-reference translation and rotation-vector corrections")


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _left_jacobian(vector):
    """SO(3) left Jacobian: d Exp(r) = [J_l(r) dr]x Exp(r)."""
    vector = np.asarray(vector, dtype=float)
    squared = float(vector @ vector)
    cross = _skew(vector)
    if squared < 1e-8:
        a = .5 - squared/24. + squared*squared/720.
        b = 1./6. - squared/120. + squared*squared/5040.
    else:
        theta = np.sqrt(squared)
        a, b = (1.-np.cos(theta))/squared, (theta-np.sin(theta))/(squared*theta)
    return np.eye(3) + a*cross + b*(cross @ cross)


class ObjectTrajectory:
    """Working object path with immutable references and optional hard corridors."""
    def __init__(self, schedule, cfg, *, allow_empty_material=False):
        self.config = validate_object_config(cfg)
        rotations, positions = _poses(schedule["object_reference_rotations"],
                                      schedule["object_reference_positions"], "reference object")
        self.fps = _timeline(float(np.asarray(schedule["fps"])), len(positions), self.config)
        self.reference_rotations, self.reference_positions = rotations.copy(), positions.copy()
        self.reference_rotations.setflags(write=False)
        self.reference_positions.setflags(write=False)
        self.motion_policy = resolve_object_policy(rotations, positions, self.config)
        self.optimization_enabled = self.motion_policy["optimization_enabled"]
        self.schedule = schedule
        self.local_points = np.asarray(schedule["object_local_points"], dtype=float).copy()
        self.local_normals = np.asarray(schedule["object_local_normals"], dtype=float).copy()
        self.gains = np.asarray(schedule["gains"], dtype=float).copy()
        self.weights = np.asarray(schedule["weights"], dtype=float).copy()
        contacts = len(self.local_points)
        if ((not contacts and self.optimization_enabled and not allow_empty_material) or self.local_points.shape != (contacts, 3)
                or self.local_normals.shape != (contacts, 3)
                or self.gains.shape != (len(positions), contacts) or self.weights.shape != (contacts, 2)
                or any(not np.isfinite(x).all() for x in (self.local_points, self.local_normals, self.gains, self.weights))
                or (self.gains < 0).any() or (self.weights < 0).any()
                or not np.allclose(np.linalg.norm(self.local_normals, axis=1), 1., rtol=0, atol=1e-6)):
            raise ValueError("object material arrays require finite matching frames/contact dimensions and unit normals")
        self.positions, self.rotations = positions.copy(), rotations.copy()
        if not self.optimization_enabled:
            self.translation_offsets = np.zeros_like(positions)
            self.rotation_offsets = np.zeros_like(positions)
            self.apply_schedule()
            return
        stop, anchor = self.config["first_edit_frame"], self.config["stationary_reference_frame"]
        self.positions[:stop] = positions[anchor]
        self.rotations[:stop] = rotations[anchor]
        self.translation_offsets, self.rotation_offsets = _offsets(rotations, positions, self.rotations, self.positions)
        # A noisy reference can require nonzero frozen-prefix corrections. Carry
        # the last correction forward, decaying at its maximum rate, instead of
        # creating an infeasible discontinuity at the first editable frame.
        for array, field in ((self.translation_offsets, "max_translation_offset_rate_m_s"),
                             (self.rotation_offsets, "max_rotation_offset_rate_rad_s")):
            if field not in self.config:
                array[stop:] = 0.
                continue
            maximum = self.config[field]
            for frame in range(stop, len(array)):
                if frame == 0:
                    array[frame] = 0.
                    continue
                previous = array[frame-1]
                length = np.linalg.norm(previous)
                array[frame] = previous * max(0., 1.-maximum/self.fps/max(length, 1e-30))
        self.positions[stop:] = self.reference_positions[stop:] + self.translation_offsets[stop:]
        if stop < len(positions):
            self.rotations[stop:] = (Rotation.from_rotvec(self.rotation_offsets[stop:]).as_matrix()
                                     @ self.reference_rotations[stop:])
        initial = self.report()
        if not initial["passed"]:
            failed = [name for name, passed in initial["gates"].items() if not passed]
            raise ValueError("stationary object initialization conflicts with original-reference constraints: "
                             + ", ".join(failed))
        self.apply_schedule()

    def _frame(self, frame):
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(self.positions):
            raise ValueError("object frame must index the source timeline")
        return int(frame)

    def apply_schedule(self, frame=None):
        """Synchronize material positions, normals and collision poses atomically."""
        if frame is None:
            self.schedule["targets"] = np.einsum("tij,cj->tci", self.rotations, self.local_points) + self.positions[:, None]
            self.schedule["target_normals"] = np.einsum("tij,cj->tci", self.rotations, self.local_normals)
            self.schedule["collision_object_poses"] = np.c_[self.positions, Rotation.from_matrix(self.rotations).as_quat()]
        else:
            frame = self._frame(frame)
            self.schedule["targets"][frame] = self.local_points @ self.rotations[frame].T + self.positions[frame]
            self.schedule["target_normals"][frame] = self.local_normals @ self.rotations[frame].T
            self.schedule["collision_object_poses"][frame] = np.r_[self.positions[frame], Rotation.from_matrix(self.rotations[frame]).as_quat()]

    def _temporal_target(self, values, frame, kind):
        weights = {order+"_weight": self.config[f"{kind}_{order}_weight"] for order in _ORDERS}
        if not any(weights.values()):
            return values[frame].copy(), 0.
        return conditional_target(values, frame, self.fps, dict(sweeps=1, max_seconds=1., **weights))

    def _energy_function(self, frame, robot_points, robot_normals, collision_callback):
        robot_points = np.asarray(robot_points, dtype=float)
        robot_normals = np.asarray(robot_normals, dtype=float)
        if (robot_points.shape != self.local_points.shape or robot_normals.shape != self.local_normals.shape
                or not np.isfinite(robot_points).all() or not np.isfinite(robot_normals).all()):
            raise ValueError("object block robot anchors/normals must be finite matching [contacts, 3] arrays")
        position_weight = self.gains[frame]*self.weights[:, 0]
        normal_weight = self.gains[frame]*self.weights[:, 1]
        target_t, gain_t = self._temporal_target(self.positions, frame, "translation")
        target_R, gain_R = self._temporal_target(self.rotations.reshape(len(self.rotations), 9), frame, "rotation")
        target_R = target_R.reshape(3, 3)
        reference_t, reference_R = self.reference_positions[frame], self.reference_rotations[frame]

        def objective(parameters):
            parameters = np.asarray(parameters, dtype=float)
            if parameters.shape != (6,) or not np.isfinite(parameters).all():
                raise ValueError("object block parameters must be six finite correction coordinates")
            translation_delta, rotation_delta = parameters[:3], parameters[3:]
            translation = reference_t + translation_delta
            rotation = Rotation.from_rotvec(rotation_delta).as_matrix() @ reference_R
            rotated_points, rotated_normals = self.local_points @ rotation.T, self.local_normals @ rotation.T
            point_error = rotated_points + translation - robot_points
            normal_error = rotated_normals + robot_normals
            energy = float(np.sum(position_weight[:, None]*point_error**2)
                           + np.sum(normal_weight[:, None]*normal_error**2))
            translation_gradient = 2.*np.sum(position_weight[:, None]*point_error, axis=0)
            rotation_gradient = 2.*np.sum(position_weight[:, None]*np.cross(rotated_points, point_error)
                                          + normal_weight[:, None]*np.cross(rotated_normals, normal_error), axis=0)
            translation_error, rotation_error = translation-target_t, rotation-target_R
            energy += float(gain_t**2*np.sum(translation_error**2) + gain_R**2*np.sum(rotation_error**2))
            translation_gradient += 2.*gain_t**2*translation_error
            rotation_gradient += 2.*gain_R**2*np.cross(rotation.T, rotation_error.T).sum(axis=0)
            if collision_callback is not None:
                collision_energy, collision_gradient = collision_callback(rotation, translation)
                collision_gradient = np.asarray(collision_gradient, dtype=float)
                if (not np.isfinite(collision_energy) or collision_energy < 0
                        or collision_gradient.shape != (6,) or not np.isfinite(collision_gradient).all()):
                    raise ValueError("object collision query returned invalid energy/left-tangent gradient")
                energy += float(collision_energy)
                translation_gradient += collision_gradient[:3]
                rotation_gradient += collision_gradient[3:]
            wt, wr = self.config["reference_translation_weight"], self.config["reference_rotation_weight"]
            # The optimization coordinates may cross pi in soft mode; reference
            # error remains the principal geodesic angle, not raw parameter size.
            canonical_rotation_delta = Rotation.from_rotvec(rotation_delta).as_rotvec()
            energy += float(wt*(translation_delta @ translation_delta)
                            + wr*(canonical_rotation_delta @ canonical_rotation_delta))
            gradient = np.r_[translation_gradient+2.*wt*translation_delta,
                             _left_jacobian(rotation_delta).T @ (
                                 rotation_gradient+2.*wr*canonical_rotation_delta)]
            if not np.isfinite(energy) or not np.isfinite(gradient).all():
                raise ValueError("nonfinite object conditional energy/gradient")
            return energy, gradient

        return objective

    def energy_gradient(self, frame, parameters, robot_points, robot_normals, collision_callback=None):
        """Evaluate a conditional block, including every touching temporal stencil."""
        frame = self._frame(frame)
        return self._energy_function(frame, robot_points, robot_normals, collision_callback)(parameters)

    def _constraints(self, frame):
        balls = []
        for start, field in ((0, "max_translation_m"), (3, "max_rotation_rad")):
            if field in self.config:
                balls.append((start, np.zeros(3), self.config[field]))
        for neighbor in (frame-1, frame+1):
            if 0 <= neighbor < len(self.positions):
                for start, offsets, field in (
                        (0, self.translation_offsets, "max_translation_offset_rate_m_s"),
                        (3, self.rotation_offsets, "max_rotation_offset_rate_rad_s")):
                    if field in self.config:
                        balls.append((start, offsets[neighbor].copy(), self.config[field]/self.fps))

        def constraint(parameters):
            return np.asarray([1.-np.sum((parameters[start:start+3]-center)**2)/radius**2
                               for start, center, radius in balls])

        def jacobian(parameters):
            rows = []
            for start, center, radius in balls:
                row = np.zeros(6)
                row[start:start+3] = -2.*(parameters[start:start+3]-center)/radius**2
                rows.append(row)
            return np.asarray(rows).reshape(-1, 6)

        return constraint, jacobian

    def _shared_poses(self, parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (6,) or not np.isfinite(parameters).all():
            raise ValueError("shared object pose requires six finite correction coordinates")
        anchor = self.config["stationary_reference_frame"]
        return (Rotation.from_rotvec(parameters[3:]).as_matrix() @ self.reference_rotations[anchor],
                self.reference_positions[anchor] + parameters[:3])

    def _shared_energy_function(self, robot_points, robot_normals, collision_callbacks):
        """Exact conditional energy of the one shared precontact pose.

        Temporal stencils are counted once with every shared frame varied
        together. Summing per-frame conditional energies would incorrectly
        penalize motion of the whole otherwise static interval.
        """
        stop = self.config["contact_frame"]
        if stop == 0:
            raise ValueError("No precontact frames to optimize")
        robot_points = np.asarray(robot_points, dtype=float)
        robot_normals = np.asarray(robot_normals, dtype=float)
        expected = (stop, len(self.local_points), 3)
        if (robot_points.shape != expected or robot_normals.shape != expected
                or not np.isfinite(robot_points).all() or not np.isfinite(robot_normals).all()):
            raise ValueError("shared object block requires finite [precontact frames, contacts, 3] anchors/normals")
        callbacks = [None]*stop if collision_callbacks is None else list(collision_callbacks)
        if len(callbacks) != stop or any(c is not None and not callable(c) for c in callbacks):
            raise ValueError("shared object collision callbacks must match every precontact frame")
        position_weight = self.gains[:stop]*self.weights[:, 0]
        normal_weight = self.gains[:stop]*self.weights[:, 1]
        # Only stencils whose summed prefix coefficient is nonzero depend on
        # the shared pose. Stencils wholly inside the constant interval vanish.
        indicator = np.zeros(len(self.positions))
        indicator[:stop] = 1.
        temporal = []
        for order, label in enumerate(_ORDERS, 1):
            coefficients = np.diff(indicator, n=order)
            touched = coefficients != 0.
            for kind, values in (("translation", self.positions), ("rotation", self.rotations)):
                weight = self.config[f"{kind}_{label}_weight"]*self.fps**(2*order)
                if weight and np.any(touched):
                    temporal.append((kind, weight, coefficients[touched],
                                     np.diff(values, n=order, axis=0)[touched], values[0].copy()))
        reference_t = self.reference_positions[:stop]
        reference_R = self.reference_rotations[:stop]
        wt, wr = self.config["reference_translation_weight"], self.config["reference_rotation_weight"]

        def objective(parameters):
            rotation, translation = self._shared_poses(parameters)
            rotated_points = self.local_points @ rotation.T
            rotated_normals = self.local_normals @ rotation.T
            point_error = rotated_points[None] + translation - robot_points
            normal_error = rotated_normals[None] + robot_normals
            energy = float(np.sum(position_weight[..., None]*point_error**2)
                           + np.sum(normal_weight[..., None]*normal_error**2))
            translation_gradient = 2.*np.sum(position_weight[..., None]*point_error, axis=(0, 1))
            rotation_gradient = 2.*np.sum(
                position_weight[..., None]*np.cross(rotated_points[None], point_error)
                + normal_weight[..., None]*np.cross(rotated_normals[None], normal_error), axis=(0, 1))
            translation_error = translation-reference_t
            rotation_error = Rotation.from_matrix(rotation @ reference_R.transpose(0, 2, 1)).as_rotvec()
            energy += float(wt*np.sum(translation_error**2) + wr*np.sum(rotation_error**2))
            translation_gradient += 2.*wt*translation_error.sum(axis=0)
            rotation_gradient += 2.*wr*rotation_error.sum(axis=0)
            for kind, weight, coefficients, initial_difference, initial_pose in temporal:
                value = translation if kind == "translation" else rotation
                shape = (-1,) + (1,)*value.ndim
                c = coefficients.reshape(shape)
                difference = initial_difference + c*(value-initial_pose)
                energy += float(weight*np.sum(difference**2))
                derivative = 2.*weight*np.sum(c*difference, axis=0)
                if kind == "translation":
                    translation_gradient += derivative
                else:
                    rotation_gradient += np.cross(rotation.T, derivative.T).sum(axis=0)
            for callback in callbacks:
                if callback is None:
                    continue
                collision_energy, collision_gradient = callback(rotation, translation)
                collision_gradient = np.asarray(collision_gradient, dtype=float)
                if (not np.isfinite(collision_energy) or collision_energy < 0
                        or collision_gradient.shape != (6,) or not np.isfinite(collision_gradient).all()):
                    raise ValueError("shared object collision query returned invalid energy/left-tangent gradient")
                energy += float(collision_energy)
                translation_gradient += collision_gradient[:3]
                rotation_gradient += collision_gradient[3:]
            gradient = np.r_[translation_gradient,
                             _left_jacobian(np.asarray(parameters)[3:]).T @ rotation_gradient]
            if not np.isfinite(energy) or not np.isfinite(gradient).all():
                raise ValueError("nonfinite shared object conditional energy/gradient")
            return energy, gradient
        return objective

    def precontact_energy_gradient(self, parameters, robot_points, robot_normals, collision_callbacks=None):
        """Evaluate the shared block against unchanged per-frame references."""
        return self._shared_energy_function(robot_points, robot_normals, collision_callbacks)(parameters)

    def _shared_constraints(self, parameters):
        """Original-reference caps and rates, including the editable join."""
        rotation, translation = self._shared_poses(parameters)
        stop = self.config["contact_frame"]
        positions, rotations = self.positions[:stop+1].copy(), self.rotations[:stop+1].copy()
        positions[:stop], rotations[:stop] = translation, rotation
        dt, dr = _offsets(self.reference_rotations[:stop+1], self.reference_positions[:stop+1],
                          rotations, positions)
        values = []
        for offsets, cap, rate in (
                (dt, "max_translation_m", "max_translation_offset_rate_m_s"),
                (dr, "max_rotation_rad", "max_rotation_offset_rate_rad_s")):
            if cap in self.config:
                values.extend(1.-np.sum(offsets[:stop]**2, axis=1)/self.config[cap]**2)
            if rate in self.config:
                values.extend(1.-np.sum(np.diff(offsets, axis=0)**2, axis=1)
                              /(self.config[rate]/self.fps)**2)
        return np.asarray(values)

    def _shared_constraints_jacobian(self, parameters):
        # Central differences avoid the forward-difference bias that stalls
        # SLSQP just outside tight correction-rate balls. The independent audit
        # still enforces the original configured limits without relaxation.
        steps = np.eye(6)*1e-6
        return np.asarray([(self._shared_constraints(parameters+step)
                            - self._shared_constraints(parameters-step))/2e-6
                           for step in steps]).T

    def update_precontact(self, robot_points, robot_normals, collision_callbacks=None):
        """Move the whole precontact interval as one pose, or leave it unchanged."""
        stop, anchor = self.config["contact_frame"], self.config["stationary_reference_frame"]
        row = dict(shared_pose=True, frames=list(range(stop)), accepted=False, iterations=0)
        if not self.optimization_enabled:
            return dict(row, reason="STATIC_ORIGINAL_PASSTHROUGH")
        if not stop:
            return dict(row, reason="NO_PRECONTACT_FRAMES")
        old = np.r_[self.positions[0]-self.reference_positions[anchor],
                    Rotation.from_matrix(self.rotations[0] @ self.reference_rotations[anchor].T).as_rotvec()]
        objective = self._shared_energy_function(robot_points, robot_normals, collision_callbacks)
        constraints = ([dict(type="ineq", fun=self._shared_constraints,
                             jac=self._shared_constraints_jacobian)]
                       if _OPTIONAL_LIMIT_FIELDS & set(self.config) else [])
        try:
            before, _ = objective(old)
            result = minimize(objective, old, method="SLSQP", jac=True, constraints=constraints,
                options=dict(maxiter=self.config["max_iterations"], ftol=1e-10))
            proposed = np.asarray(result.x, dtype=float)
            rotation, translation = self._shared_poses(proposed)
            after, _ = objective(proposed)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            return dict(row, reason="INVALID_PROPOSAL", error=str(error))
        row.update(iterations=int(result.nit), cost_before=float(before), cost_after=float(after),
                   optimizer_success=bool(result.success), optimizer_message=str(result.message))
        feasibility = self._shared_constraints(proposed)
        if feasibility.size and float(feasibility.min()) < -1e-8:
            return dict(row, reason="OUTSIDE_HARD_CORRIDOR")
        if after >= before - 1e-12*max(1., abs(before)):
            return dict(row, reason="NO_IMPROVEMENT")
        proposed_positions, proposed_rotations = self.positions.copy(), self.rotations.copy()
        proposed_positions[:stop], proposed_rotations[:stop] = translation, rotation
        audit = audit_object_trajectory(self.reference_rotations, self.reference_positions,
            proposed_rotations, proposed_positions, self.fps, self.config)
        if not audit["passed"]:
            return dict(row, reason="INDEPENDENT_CORRIDOR_CHECK_FAILED")
        self.positions[:stop], self.rotations[:stop] = translation, rotation
        dt, dr = _offsets(self.reference_rotations[:stop], self.reference_positions[:stop],
                          self.rotations[:stop], self.positions[:stop])
        self.translation_offsets[:stop], self.rotation_offsets[:stop] = dt, dr
        self.apply_schedule()
        return dict(row, accepted=True, reason="FEASIBLE_SHARED_POSE_IMPROVEMENT")

    def update(self, frame, robot_points, robot_normals, collision_callback=None):
        """Try one constrained coordinate step; rejected proposals change nothing."""
        frame = self._frame(frame)
        if not self.optimization_enabled:
            return dict(frame=frame, accepted=False, reason="STATIC_ORIGINAL_PASSTHROUGH", iterations=0)
        if frame < self.config["first_edit_frame"]:
            return dict(frame=frame, accepted=False, reason="USE_SHARED_PRECONTACT_BLOCK", iterations=0)
        old = np.r_[self.translation_offsets[frame], self.rotation_offsets[frame]]
        objective = self._energy_function(frame, robot_points, robot_normals, collision_callback)
        constraint, jacobian = self._constraints(frame)
        try:
            before, _ = objective(old)
            result = minimize(objective, old, method="SLSQP", jac=True,
                constraints=[dict(type="ineq", fun=constraint, jac=jacobian)] if constraint(old).size else [],
                options=dict(maxiter=self.config["max_iterations"], ftol=1e-10))
            proposed = np.asarray(result.x, dtype=float)
            if proposed.shape != (6,) or not np.isfinite(proposed).all():
                raise ValueError("nonfinite object optimizer proposal")
            after, _ = objective(proposed)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            return dict(frame=frame, accepted=False, reason="INVALID_PROPOSAL", error=str(error), iterations=0)
        row = dict(frame=frame, accepted=False, iterations=int(result.nit), cost_before=float(before),
                   cost_after=float(after), optimizer_success=bool(result.success), optimizer_message=str(result.message))
        constraints = constraint(proposed)
        if constraints.size and float(constraints.min()) < -1e-8:
            return dict(row, reason="OUTSIDE_HARD_CORRIDOR")
        if after >= before - 1e-12*max(1., abs(before)):
            return dict(row, reason="NO_IMPROVEMENT")
        proposed_positions, proposed_rotations = self.positions.copy(), self.rotations.copy()
        proposed_positions[frame] = self.reference_positions[frame] + proposed[:3]
        proposed_rotations[frame] = Rotation.from_rotvec(proposed[3:]).as_matrix() @ self.reference_rotations[frame]
        checked = audit_object_trajectory(self.reference_rotations, self.reference_positions,
            proposed_rotations, proposed_positions, self.fps, self.config)
        if not checked["passed"]:
            return dict(row, reason="INDEPENDENT_CORRIDOR_CHECK_FAILED")
        # An iteration-limit result is still usable only if independently
        # feasible and improving; termination status alone never accepts it.
        self.translation_offsets[frame] = proposed[:3]
        self.rotation_offsets[frame] = Rotation.from_rotvec(proposed[3:]).as_rotvec()
        self.positions[frame] = proposed_positions[frame]
        self.rotations[frame] = proposed_rotations[frame]
        self.apply_schedule(frame)
        return dict(row, accepted=True, reason="FEASIBLE_IMPROVEMENT")

    def report(self):
        report = audit_object_trajectory(self.reference_rotations, self.reference_positions,
            self.rotations, self.positions, self.fps, self.config)
        report["initialization"] = ("unchanged original static trajectory" if not self.optimization_enabled else
            "shared optimizable precontact pose initialized at reference; original suffix unless an explicit correction-rate cap requires bounded decay")
        return report
