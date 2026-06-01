# LAW — Recent Development Report

*Written 2026-05-28. Covers changes made since the initial public release commit.*

---

## Overview

The recent development phase has two main threads running in parallel:

1. **Infrastructure improvements** — making training runs reproducible, self-contained, and easier to inspect.
2. **Semantic conditioning** — augmenting the model with language-grounded visual features from CLIP, extending the original LAW architecture with a second backbone path.

---

## 1. Infrastructure: Timestamped Run Directories

### What changed

`tools/nusc_my_train.sh` now creates a timestamped subdirectory for every run:

```
work_dirs/<exp_name>/<YYYYMMDD_HHMMSS>/
```

Previously all runs of the same config wrote into a single flat directory, so artifacts from different runs would overwrite each other. Now each run is isolated.

The script also exports a `LAW_RUN_ID` environment variable so Python processes can discover the run identity without recomputing it.

`tools/train.py` was extended with race-condition-safe run-ID logic for distributed training. All worker processes launched by `torch.distributed.launch` share the same parent PID, so a lock file keyed by `MASTER_PORT + parent_PID` in `/tmp` lets every rank land in the same subdirectory without a race. The fallback chain is: (1) `LAW_RUN_ID` env var, (2) lock file, (3) fresh timestamp.

### Why it matters

The `work_dirs/law/default/` directory now contains `all_old/` (everything before the change) and one dated folder per run, making it trivial to compare runs and roll back checkpoints.

---

## 2. Default Config Tuning

Several practical changes were made to `projects/configs/law/default.py`:

- **`samples_per_gpu` 1 → 3.** The original value (1) was set conservatively. Increasing to 3 fills GPU memory better and gives more gradient signal per step. With 8 GPUs, total batch is 24.
- **Evaluation interval 12 → 3.** Previously evaluation only ran at the very end of training. Evaluating every 3 epochs lets you track convergence and catch regressions early.
- **`find_unused_parameters = False`** added explicitly. DDP warns if this is left unset when all parameters receive gradients; setting it to `False` avoids the overhead of scanning for unused params.
- **`use_semantic = False`** flags added to both the model and the head. These make the default config a clean baseline that can be compared against the semantic variant without ambiguity.

---

## 3. CLIP Semantic Backbone (`CLIPSemanticBackbone`)

### The problem it solves

The original LAW backbone is SwinTransformer3D, which is trained on visual recognition tasks with ImageNet-style supervision. It encodes texture, shape, and motion well, but has no explicit grounding to language or semantics. A traffic light being red vs green, a pedestrian actively crossing vs standing still, brake lights activating on the car ahead — these are the cues that most directly affect what the ego vehicle should do next. SwinTransformer can learn to respond to them implicitly, but CLIP's vision encoder was trained to align images with natural language descriptions, so these semantic states are more explicitly represented in its features.

### Implementation

`projects/mmdet3d_plugin/models/backbones/clip_semantic_backbone.py` wraps HuggingFace's `CLIPVisionModel` (ViT-B/16 by default, `openai/clip-vit-base-patch16`).

**Input/output contract:**
```
Input:  [B, C, H, W]     — mmdet ImageNet-normalized images, any resolution
Output: [B, 768]         — CLS token per image (language-grounded global descriptor)
```

For the multi-view setting, the caller flattens views into the batch dimension:
```python
flat = cur_img.reshape(B * N_view, C, H, W)
sem  = backbone(flat).reshape(B, N_view, 768)
```

**Normalization handling:** The mmdet pipeline uses ImageNet mean/std. CLIP requires its own normalization. The backbone stores both as buffers and handles the conversion: undo mmdet normalization → re-normalize to CLIP statistics → resize to 224×224 (required by ViT).

**Frozen by default.** The backbone's weights are frozen (`requires_grad=False`). This keeps training cost down and avoids fine-tuning a large model on a relatively small dataset. The `use_lora` option provides a path to parameter-efficient fine-tuning via LoRA (targeting `q_proj`, `k_proj`, `v_proj`, `out_proj`) if needed later.

**PyTorch 1.9 compatibility patch.** `torch.frombuffer` was added in PyTorch 1.10. Since this repo runs on 1.9.1, a patch is applied at import time using a numpy-backed fallback so safetensors can load the pretrained weights.

---

## 4. Semantic Conditioning in `WaypointHead`

The semantic features are wired into `WaypointHead` via a `use_semantic` flag. When enabled, the head instantiates a single linear projection layer:

```python
self.semantic_proj = nn.Linear(768, 256)   # vlm_hidden_channel → hidden_channel
```

### In the forward pass (trajectory prediction path)

After the per-view spatial attention produces `spatial_view_feat` (shape `[B, num_views * num_tokens, 256]`), the projected CLIP CLS tokens are appended to this memory:

```python
proj_semantic_feat = self.semantic_proj(semantic_feat)   # [B, num_views, 256]
memory_feat = torch.cat([spatial_view_feat, proj_semantic_feat], dim=1)
```

The waypoint cross-attention then attends over this extended memory, which includes both the spatially-grounded geometric features and the per-view semantic tokens. The trajectory proposals are thus influenced by semantic scene state, not just spatial structure.

### In the world model prediction path

The action-aware encoder that feeds the world model decoder is also extended. In the semantic variant, it takes:

```
[view_features || pooled_semantic || flattened_waypoints]
```

(dimension: `256 + 256 + 12 = 524`), rather than the original `256 + 12 = 268`. This means the world model's prediction of the next latent scene representation is conditioned on both the current action (trajectory) and the current semantic understanding.

**Graceful fallback for history frames.** `obtain_history_feat` processes frames from the history queue through `WaypointHead` without semantic features (CLIP is only run on the current frame to save compute). When `proj_semantic_feat` is `None`, zeros of the correct shape are substituted so the action-aware encoder always receives a consistent-dimension input.

---

## 5. New Config: `default_with_semantic.py`

`projects/configs/law/default_with_semantic.py` inherits from `default.py` and overrides the full model dict with `_delete_=True` to prevent key conflicts. Key additions:

- `use_semantic=True` in both `LAW` and `WaypointHead`
- `semantic_img_backbone` pointing to `CLIPSemanticBackbone` (frozen ViT-B/16)
- `swin_input_channel=768` and `vlm_hidden_channel=768` for consistency
- `find_unused_parameters=True` — `semantic_proj` only receives gradients through the current-frame path, not through history frames, so DDP needs permission to skip it for those backward passes

---

## 6. Visualization Infrastructure

Two additions were made to support visual inspection of trajectory predictions during evaluation.

### `prj_ego_traj_to_2d`

A new utility function (`projects/mmdet3d_plugin/LAW/utils/visualization.py`) that projects ego trajectory waypoints from 3D ego-frame coordinates into 2D image pixel coordinates for any given camera. It uses the `lidar2img` calibration matrices stored in `img_metas`.

### Overlay rendering in `simple_test_pts`

`simple_test_pts` now accepts an `img` parameter. When a `show_dir` is set, it:
1. Denormalizes the front camera image back to RGB
2. Projects GT waypoints and predicted waypoints into 2D using the above utility
3. Draws GT as green dots and predictions as red dots using `draw_lidar_pts`
4. Writes the annotated frame to `show_dir/<call_count>.jpg`

The visualization is currently disabled by default (`show_dir = None`) with a `TODO` comment. A `call_count` tracker on the `LAW` object subsamples frames (every other call) and drives a periodic average metric print every 500 calls, which is useful for monitoring long evaluation runs.

---

## 7. Experiments

### Baseline (default) — `work_dirs/law/default/20260527_091231/`

The baseline config ran to full completion (12 epochs) on 2026-05-27 on what appears to be 2 GPUs (inferred from memory and throughput). Training stabilized well: reconstruction loss settled around 0.010 and waypoint loss around 0.18–0.20 by epoch 12, with smooth cosine-annealed LR from 5e-5 down to ~9e-7. The previous-frame waypoint loss (`prev_frame_loss_waypoint_0`) closely tracks the current-frame loss throughout, showing that the model generalizes its trajectory prediction across temporal positions. This run established the reference checkpoint for comparison with the semantic variant.

### Semantic Run 1 — `work_dirs/law/default_with_semantic/20260527_101401/`

Started on the same day as the baseline but on 2 A100 80GB GPUs with `samples_per_gpu=6` (batch size 12). The initial step time was very high (~6s per step vs the baseline's ~1.2s), driven by CLIP forward passes loading into memory and the larger per-GPU batch. Memory usage was ~45,570 MB per GPU — substantially higher than the baseline's ~23,020 MB. The run completed 3 epochs before being interrupted. 

The unusually high memory footprint in this run was traced to the larger `samples_per_gpu` setting (6 rather than 3), not to CLIP itself. Since CLIP is frozen and operates on resized 224×224 images, its contribution to memory is modest.

### Semantic Run 2 — `work_dirs/law/default_with_semantic/20260528_084506/`

Started fresh on 2026-05-28 on 4 V100 32GB GPUs with the corrected `samples_per_gpu=3` (batch size 12). Memory settled at ~23,415 MB per GPU — essentially the same as the baseline. Step time was ~1.52s vs the baseline's ~1.19s, a ~28% overhead attributable to the CLIP forward pass (six cameras, frozen ViT-B/16). This overhead is predictable and acceptable given that CLIP is frozen and the extra cost is limited to the current-frame path. 

Training was ongoing at epoch 6 as of the last log entry, with the loss landscape following the same qualitative pattern as the baseline.

---

## Summary

| Component | Purpose |
|---|---|
| `CLIPSemanticBackbone` | Frozen CLIP ViT-B/16 CLS token per camera view; language-grounded semantic features |
| Semantic conditioning in `WaypointHead` | Appends CLIP tokens to waypoint attention memory; extends action-aware encoder for world model |
| `default_with_semantic.py` | Config wiring for the semantic variant |
| Timestamped run dirs | Isolates each training run; prevents artifact overwrite |
| Default config tuning | Larger per-GPU batch, frequent eval, cleaner flags |
| Visualization tools | Trajectory overlay on front-camera images for qualitative inspection |
