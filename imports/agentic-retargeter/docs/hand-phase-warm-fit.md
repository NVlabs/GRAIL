# Transition-aware phase spline initialization

`trajectory_parameterization.warm_fit_scope` is optional for
`phase_cubic_bspline`. The default, `strong_contact`, preserves the existing
initialization exactly. `coordinate_support` also fits the object-coordinate
position and rotation branch to the original warm trajectory during the
existing approach/release coordinate transition.

```json
{
  "trajectory_parameterization": {
    "kind": "phase_cubic_bspline",
    "control_count": 49,
    "finger_space": "bounded_joint_controls",
    "transition_frames": 18,
    "allowed_transition_roles": ["no_contact", "contact_acquisition", "unknown"],
    "warm_fit_scope": "coordinate_support"
  }
}
```

Keep each case's actual allowed transition roles; the example is not a request
to relabel phases. The new option changes initialization only. Coordinate
weights, source timestamps, contact masks, loss functions, native joint limits
and source-relative wrist boxes remain unchanged. Support and sliding motion
remain variable.

The position branch uses its existing coordinate weight as the weighted
least-squares fit weight. The rotation reference includes frames where that
weight is positive. The existing constrained position fit still applies when
the decoded warm state would exceed the original source box. The runtime
decoder and feasibility-filtered optimization steps are unchanged.

The motivation was measured acquisition distortion from extrapolating a
strong-contact branch backward into its transition: in the 49-control Pan and
Hammer warm decoders, right-wrist errors reached74mm and100mm. A fit-only
counterfactual reduced those maxima to5.5mm and1.9mm while leaving the source
boxes feasible. These measurements establish an initialization issue; they do
not establish collision-free motion or success of downstream optimization.

Upstream tests (not shipped in this snapshot) covered exact legacy equality,
position/rotation approach fidelity, unchanged world-only states and masks,
native finger limits, source-box projection and live decoder gradients.
