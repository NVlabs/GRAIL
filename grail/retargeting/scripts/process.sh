#!/bin/bash
# Step 2 — add hand actions + table geometry to retargeted motions.
#
# Usage:  bash grail/retargeting/scripts/process.sh <output_folder>
# Example: bash grail/retargeting/scripts/process.sh benchmark_v3_0203
#
# Reads   data/motion_lib/<output_folder>/
# Writes  data/motion_lib/<output_folder>_ha/ (training-ready)

set -euo pipefail

OUTPUT_FOLDER="${1:?output folder name under data/motion_lib/}"
shift
INPUT_DIR="data/motion_lib/${OUTPUT_FOLDER}"
OUTPUT_DIR="data/motion_lib/${OUTPUT_FOLDER}_ha"
META_PKL="data/body_profiles/unitree_g1/smplx/g1_skeleton_meta.pkl"
GRASP_ANTICIPATION_FRAMES="${GRAIL_GRASP_ANTICIPATION_FRAMES:-10}"

eval "$(conda shell.bash hook)"
conda activate "${GRAIL_SONIC_ENV:-sonic}"

python -u -m grail.retargeting.process \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --meta_pkl "${META_PKL}" \
    --include_contact_points \
    --grasp_from_lift \
    --lift_threshold 0.02 \
    --grasp_anticipation_frames "${GRASP_ANTICIPATION_FRAMES}" \
    --skip_no_lift \
    --per_object \
    "$@"

echo "Processed output: ${OUTPUT_DIR}/{robot,objects,meta}/"
