"""Explicit native-hinge refinement support; roots and objects remain exact.

This restricts the variables of the complete trajectory objective. It does not
truncate residuals, omit temporal stencils or claim collision-free motion.
"""
import numpy as np


def validate_config(value):
    if not isinstance(value, dict) or set(value) != {"windows"}:
        raise ValueError("refinement_scope requires exactly windows")
    windows = value["windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("refinement_scope windows must be nonempty")
    clean = []
    for row in windows:
        if not isinstance(row, dict) or set(row) != {"frames", "joints"}:
            raise ValueError("refinement window requires frames and joints")
        frames, joints = row["frames"], row["joints"]
        if (not isinstance(frames, list) or len(frames) != 2
                or any(type(x) is not int for x in frames)
                or not 0 <= frames[0] < frames[1]):
            raise ValueError("refinement frames must be a nonempty half-open integer interval")
        if (not isinstance(joints, list) or not joints
                or any(not isinstance(x, str) or not x or x != x.strip() for x in joints)
                or len(set(joints)) != len(joints)):
            raise ValueError("refinement joints must be unique explicit native scalar names")
        clean.append(dict(frames=list(frames), joints=list(joints)))
    return {"windows": clean}


def resolve(value, *, frames, dofs_per_frame, coordinate_count, coordinates,
            dofs, names, object_variables):
    config = validate_config(value)
    if (any(type(x) is not int or x < 1 for x in (frames, dofs_per_frame, coordinate_count))
            or type(object_variables) is not int or object_variables < 0):
        raise ValueError("Invalid complete native refinement dimensions")
    coordinates, dofs = np.asarray(coordinates), np.asarray(dofs)
    names = list(names)
    if (coordinates.ndim != 1 or coordinates.dtype.kind not in "iu"
            or dofs.shape != coordinates.shape or dofs.dtype.kind not in "iu"
            or len(names) != len(coordinates) or len(set(names)) != len(names)
            or any(not isinstance(x, str) or not x for x in names)
            or len(set(coordinates.tolist())) != len(coordinates)
            or len(set(dofs.tolist())) != len(dofs)
            or np.any(coordinates < 0) or np.any(coordinates >= coordinate_count)
            or np.any(dofs < 0) or np.any(dofs >= dofs_per_frame)):
        raise ValueError("Invalid bound native scalar layout for refinement")
    tangent = np.zeros(frames*dofs_per_frame+object_variables, dtype=bool)
    coordinate = np.zeros((frames, coordinate_count), dtype=bool)
    resolved = []
    for row in config["windows"]:
        start, stop = row["frames"]
        if stop > frames:
            raise ValueError("Refinement window exceeds original source clock")
        indices = []
        for name in row["joints"]:
            found = [i for i, x in enumerate(names) if x == name]
            if not found:
                found = [i for i, x in enumerate(names) if x.rsplit("/", 1)[-1] == name]
            if len(found) != 1:
                raise ValueError("Refinement joint is absent or ambiguous: "+name)
            i = found[0]
            if coordinate[start:stop, coordinates[i]].any():
                raise ValueError("Repeated native refinement coordinate/window")
            coordinate[start:stop, coordinates[i]] = True
            tangent[np.arange(start, stop)*dofs_per_frame+dofs[i]] = True
            indices.append(i)
        resolved.append(dict(frames=[start, stop], joints=[names[i] for i in indices]))
    evidence = dict(config=config, resolved_windows=resolved,
                    interval_convention="half_open_original_source_frame_indices",
                    active_variable_indices=np.flatnonzero(tangent).tolist(),
                    active_variable_count=int(tangent.sum()),
                    complete_variable_count=len(tangent),
                    fixed_native_coordinate_count=int((~coordinate).sum()),
                    root_policy="all non-scalar coordinates fixed to prior global state",
                    object_policy="all actual prior-global object poses fixed",
                    objective_scope="all original trajectory rows and cross-boundary stencils retained",
                    motion_accepted=False)
    return tangent, coordinate, evidence


def check_fixed(initial, final, coordinate_mask):
    if set(initial) != set(final) or np.asarray(initial['q']).shape != coordinate_mask.shape:
        raise ValueError("Refinement state fields or coordinate mask differ")
    for key in initial:
        before, after = np.asarray(initial[key]), np.asarray(final[key])
        if key == "q":
            before, after = before[~coordinate_mask], after[~coordinate_mask]
        if (before.shape != after.shape or before.dtype != after.dtype
                or before.tobytes() != after.tobytes()):
            raise ValueError("Refinement changed a fixed prior-state field: "+key)
    return True


def validate_result(report, initial, final, *, frames, dofs_per_frame,
                    coordinate_count, coordinates, dofs, names, object_variables):
    config = report["config"]
    if "refinement_scope" not in config:
        if "refinement_scope" in report or "active_variable_indices" in report["solver_report"]:
            raise ValueError("Undeclared constrained refinement execution")
        return
    tangent, mask, expected = resolve(config["refinement_scope"], frames=frames,
        dofs_per_frame=dofs_per_frame, coordinate_count=coordinate_count,
        coordinates=coordinates, dofs=dofs, names=names, object_variables=object_variables)
    expected["fixed_state_byte_exact"] = check_fixed(initial, final, mask)
    if report.get("refinement_scope") != expected:
        raise ValueError("Refinement execution evidence differs from bound scope")
    solver = report["solver_report"]
    if (solver.get("active_variable_indices") != expected["active_variable_indices"]
            or solver.get("active_variable_count") != int(tangent.sum())
            or solver.get("stationarity_scope") != "declared active tangent columns only"):
        raise ValueError("Sparse linear solve did not bind the declared refinement subset")
