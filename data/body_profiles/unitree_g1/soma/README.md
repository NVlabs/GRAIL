# Unitree G1 SOMA identity

`soma03_custom_smplx_identity.npz` is the fixed downstream SOMA 0.3 identity
used by the public G1 reconstruction profile. GEM-X keeps its predicted source
identity during pose estimation; GRAIL applies this identity only when building
the reconstruction model.

The NPZ stores native SOMA PCA identity coefficients and local scale parameters.
Its paired JSON records the fixed global scale and file hash. The identity is an
approximate human body envelope for G1 motion generation, not a contact-quality
or hardware-safety certification.
