# Relative wrist smoothing across active contact roles

The synchronous SOMA solver can optionally smooth wrist acceleration and jerk across a change between declared contact roles. This targets a discontinuity that a same-role-only objective can move to a hold/support/slide boundary.

```json
{
  "temporal_optimization": {
    "relative_hand_temporal": {
      "roles": ["hold", "support", "slide"],
      "role_boundary_policy": "all_active",
      "translation_acceleration_weight": 0.032,
      "translation_jerk_weight": 0.000032,
      "rotation_acceleration_weight": 0.00032,
      "rotation_jerk_weight": 0.00000032
    }
  }
}
```

This is a partial configuration: retain the rest of the case's existing synchronous configuration and bound robot-hand targets. The example weights are experimental, not universal settings.

Omitting `role_boundary_policy`, or setting it to `same_role`, preserves the original normalized configuration, residuals, Jacobians and telemetry. Omitting the entire relative block still disables this objective. The new policy is opt-in.

For acceleration or jerk of order `n`, `all_active` admits a stencil starting at frame `i` only when every frame `i` through `i+n` is both contact-active and in `roles`. The roles within that stencil may differ. Inactive frames and undeclared roles still break the stencil; an acquisition frame is not admitted just because it is adjacent to a hold. No labels, contact schedule or timestamps are edited.

Translation uses the actual wrist position in the actual object's coordinates, with the existing source-time-scaled finite difference. Rotation uses the object-frame SO(3) angular increment divided by the original timestep, then its time derivatives. The existing analytic robot and object tangent maps, including shared precontact object variables, supply the Jacobian. This adds no GPU FK query or device transfer.

Velocity guidance remains complete same-role **hold-only**. Support and slide are never assigned a zero-velocity objective. The independent world-wrist objective is unchanged: mixed active-role joins can receive both world and object-relative acceleration/jerk terms. Free hands receive no new object-relative coupling.

The report includes the first frame and count of added cross-role stencils. Saved runtime evidence binds the policy and rejects inconsistent edited telemetry. Smoother contact-role joins can slightly increase motion within individual phases or tracking error; review those quantities, actual finger region coverage, motion amplitude, speed and original geometry together. This objective does not certify contact, force closure, collision clearance or dynamic feasibility.

Validation includes physical finite differences for robot and moving/shared-prefix object variables, genuine mixed-role and offcontact masks, exact default equality, actual Newton runtime/evidence validation and full121-frame CUDA runs. The local Gas20 and Hammer30 experiments preserved their original source/hand/contact inputs and all other objective weights; their recorded numerical results remain diagnostic rather than accepted motions.
