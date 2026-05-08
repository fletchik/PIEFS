#!/usr/bin/env bash
# =============================================================================
# package_for_colab.sh  —  Creates efdo_source.zip for upload to Google Drive.
#
# Run from project root:  bash scripts/package_for_colab.sh
# Then upload efdo_source.zip to MyDrive/EFDO_Colab/ in Google Drive.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="efdo_source.zip"
rm -f "$OUT"

zip -r "$OUT" \
    train.py \
    src/ \
    scripts/extract_cnn_features.py \
    -x "*.pyc" \
    -x "*/__pycache__/*" \
    -x "*.DS_Store" \
    -x "src/configs/train.yaml"   # will be patched inside notebook

# Patch train.yaml: add to zip separately (we need writer.mode=disabled default)
python3 - <<'EOF'
import zipfile, os

patch = """defaults:
  - _self_
  - dataset: two_moon
  - criterion: spectral
  - optimizer: adam

hydra:
  run:
    dir: ${hydra:runtime.cwd}/logs/${run_id}
  job:
    chdir: false

run_id: ???

trainer:
  total_steps: 60000
  log_step: 15000
  save_period: 30000
  batch_size: 256
  device: auto
  seed: 42
  skip_oom: true

augmentation:
  noise_std: 0.0
  wide_normal_fraction: 0.0

model:
  K: 6
  hidden_dims: [64, 64, 64]
  metric_type: 'off'
  metric_hidden_dims: [64, 64]
  task: binary
  output_bias: false

writer:
  project_name: efdo-colab
  mode: disabled
  entity: null
  run_name: null
  tags: null
  notes: null

pretrain:
  graph_laplacian: false
  n_points: 1000
  k_neighbors: 10
  sigma: null
  distill_steps: 2000
  distill_lr: 0.001
  update_t_class: true
"""

with zipfile.ZipFile("efdo_source.zip", "a") as zf:
    zf.writestr("src/configs/train.yaml", patch)

print("Patched train.yaml (writer.mode=disabled by default)")
EOF

echo ""
echo "Created: $OUT"
ls -lh "$OUT"
echo ""
echo "Next step: upload $OUT to Google Drive at:  MyDrive/EFDO_Colab/efdo_source.zip"
