# Public body profiles

The public release currently carries one embodiment profile: Unitree G1.
All bundled character and body-model assets live under this directory; rendering,
reconstruction, retargeting, and motion export use the profile paths directly.

`unitree_g1/` binds three separate assets that must not be conflated:

- `characters/dhgen_00008/` is the default dressed DHGen character used to render
  2D-HOI source videos. The source USD is Y-up and centimetre-authored, so the
  renderer imports it at scale `0.01`.
- `soma/` contains the fixed native SOMA03 identity used by reconstruction after
  GEM-X predicts the source motion. Raw GEM-X output keeps its predicted identity;
  the fixed G1 identity is applied only when constructing the reconstruction model.
- `smplx/` retains the legacy G1-proportioned SMPL-X assets for explicit SMPL-X
  workflows and backwards compatibility.

The DHGen character is a rendering asset, not an authorization to replace SOMA
identity parameters or change contact labels. Public reconstruction continues to
use the original contact-label optimization stages in the reconstruction YAMLs.
