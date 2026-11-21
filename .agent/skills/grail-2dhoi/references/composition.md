# Orientation, composition and render review

Read camera and placement defaults from the selected scene YAML; do not maintain
a second hardcoded copy of scene coordinates in a skill. `obj_rot` is Euler
degrees. Determine the mesh's actual up-axis before applying a 90-degree X
rotation; an already Z-up mesh does not need the Y-up conversion.

## Choose an informative orientation

| Object geometry | Useful starting view |
|---|---|
| Vehicles, carts and bulky asymmetric objects | Three-quarter view showing both length and handles/wheels |
| Long-handled tools and elongated horizontal objects | Long axis approximately parallel to the image plane |
| Labelled containers or clear front panels | Informative face toward camera |
| Baskets, helmets and open boxes | Opening visible without hiding the contact surfaces |
| Chairs | Seat, back and support geometry visible |

Measure orientation after the actual asset transform. Raw mesh axes and category
names do not identify the front reliably. A historical elongation screen used a
largest/second-largest extent ratio of 2.5; verify the projected silhouette rather
than forcing that threshold onto all objects. Use modest Z randomization when
diversity is requested and keep all sampled views usable. Check whether the pickup
scene defines `obj_rot_rand_max`; do not assume an inherited perturbation exists.

## Placement and camera

The rendering implementation supports per-object overrides for placement,
rotation/scale/opacity, character placement/rotation, and camera radius, azimuth,
elevation and offset. Confirm the current whitelist in
`grail/pipelines/blender/render_scene.py` before using a new key.

- Preserve the physical support surface. On a finite table, shifting the object
  for image separation can move it off the tabletop; consider a small azimuth
  change instead. Floor scenes have different available placement space.
- Increase camera radius only as needed for large geometry. Check interaction
  reach and intended visibility together rather than maximizing empty separation.
- High camera elevation can put the camera outside room walls. A target-offset
  change can improve vertical framing without the same camera displacement.
- Respect the selected character and its scale; do not replace it based on an
  older skill's default character name.

## Review and correction

Review every new render selected for video generation. Label contact sheets with
dataset/category/scene identifiers, cover all pages, and enlarge ambiguous cells.
Keep the coverage record and reuse it for unchanged inputs. Apply the current
task's framing policy; minor permitted cropping is not automatically a failure.

| Symptom | Inspect or adjust |
|---|---|
| Wrong face or upside-down mesh | Actual local axes, `obj_rot`, original geometry |
| Thin end-on silhouette | Orientation relative to camera ray |
| Object hides the character | Placement/support limits and camera azimuth |
| Object or body clipped | Camera radius and target offset; accepted crop policy |
| Door frame / wallpaper edge intrudes | Camera position relative to room; elevation |
| Floating object or phantom slab | Original support geometry and simulation placement |

Save revised renders and their camera/depth/initial-state metadata together in a
new attempt namespace. Do not delete earlier stage trees to force regeneration.
An orientation or scale change can invalidate simulation and downstream metadata;
do not skip those stages merely because a prior render exists. Generate videos
only from the selected reviewed render bindings.
