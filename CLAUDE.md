# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LAW (Latent World Model)** — ICLR 2025 paper implementing a self-supervised end-to-end autonomous driving framework. It predicts future scene features from current multi-view camera features and ego trajectories, then uses those world model predictions to supervise trajectory planning.

## Commands

### Training
```bash
./tools/nusc_my_train.sh law/default 8   # config name, num GPUs
```
Outputs to `work_dirs/law/default/<YYYYMMDD_HHMMSS>/` (timestamped run dirs).

### Testing / Evaluation
```bash
./tools/dist_test.sh $CONFIG $CKPT $NUM_GPU
```

### Single-GPU Training (for debugging)
```bash
PYTHONPATH="$(pwd)" python tools/train.py projects/configs/law/default.py \
    --work-dir work_dirs/debug --deterministic
```

### Data Preprocessing
```bash
python tools/create_data.py nuscenes --root-path data/nuscenes/nuscenes \
    --out-dir data/nuscenes/nuscenes --extra-tag vad_nuscenes
```

## Architecture

### Model Hierarchy
- `LAW` (`projects/mmdet3d_plugin/LAW/LAW.py`) extends `VAD` extends `MVXTwoStageDetector`
- The `VAD` base class (`projects/mmdet3d_plugin/VAD/VAD.py`) handles multi-view feature extraction and dataset interfacing
- `LAW` overrides the forward pass to add world model supervision

### Core Components

**Backbone**: SwinTransformer3D (Swin-tiny) extracts per-frame multi-view image features across 6 cameras. Pretrained from MMAction.

**WaypointHead** (`projects/mmdet3d_plugin/LAW/dense_heads/waypoint_query_decoder.py`):
- Learnable view query features `(1, 6 views, 256 channels, 6 proposals)`
- Per-view spatial TransformerDecoders (one per camera, 6 total) aggregate spatial information
- Waypoint TransformerDecoder fuses across views into 6 trajectory proposals
- Each proposal predicts 6 waypoints (x, y displacements)
- Final trajectory selected by ego command (straight/left/right)

**World Model** (inside `WaypointHead`):
- Action-aware encoder: concatenates view features with trajectory predictions
- WM decoder (2-layer TransformerDecoder): predicts next-frame latent features
- Supervised by MSE loss against actual observed features of the next frame

**Temporal processing**: A queue of 4 frames is maintained. At each step, the world model predicts features for the current frame from the previous frame, and the reconstruction loss is computed against the actual extracted features.

### Loss Functions
- `loss_waypoint`: L1 loss on trajectory waypoints (masked by valid GT)
- `loss_rec`: MSE between world-model-predicted and observed view features
- Combined: `loss_waypoint + 0.2 * loss_rec`

### Data Flow
1. Multi-view images (6 cameras, 1600×900) → SwinTransformer3D → image features
2. Features + LID positional embeddings → per-view spatial attention
3. Cross-view fusion → 6 trajectory proposals (x, y for 6 timesteps each)
4. Ego command selects one trajectory
5. World model: previous frame features + selected trajectory → predicted current features
6. Reconstruction loss vs. actual current features

## Key Configuration

**Main config**: `projects/configs/law/default.py`
- 12 training epochs, evaluation every 3 epochs
- Batch size: 3 per GPU (8 GPUs = 24 total)
- Optimizer: AdamW, lr=5e-5, weight_decay=0.01
- LR schedule: cosine annealing with linear warmup
- Queue length: 4 frames for temporal context
- Point cloud range: `[-15, -30, -2, 15, 30, 2]` (x, y, z)

**Variant**: `projects/configs/law/default_with_semantic.py` adds VLM-based semantic backbone (`CLIPSemanticBackbone` at `projects/mmdet3d_plugin/models/backbones/clip_semantic_backbone.py`).

## Dataset

Uses NuScenes with custom VAD-format pickle files (`vad_nuscenes_infos_temporal_train.pkl`, `vad_nuscenes_infos_temporal_val.pkl`). Dataset class: `VADCustomNuScenesDataset`.

Expected data layout:
```
data/nuscenes/
├── can_bus/
└── nuscenes/
    ├── maps/, samples/, sweeps/
    ├── v1.0-trainval/, v1.0-test/
    ├── vad_nuscenes_infos_temporal_train.pkl
    └── vad_nuscenes_infos_temporal_val.pkl
```

## Evaluation Metrics

Computed by `PlanningMetric` (`projects/mmdet3d_plugin/VAD/planner/metric_stp3.py`):
- **L2 error** (meters) at 1s, 2s, 3s horizons
- **Collision rate** (%) at 1s, 2s, 3s horizons

Published results (Perception-Free): L2 avg=0.62m, Collision avg=0.21%

## Installation Notes

Requires Python 3.8, CUDA 11.1, PyTorch 1.9.1, MMCV 1.4.0, MMDetection3D v0.17.1. The MM* ecosystem versions are strict — mismatches cause silent failures or crashes. See README for full install sequence.
