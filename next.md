# Plan: Replace CLIP with Qwen2-VL-2B Visual Encoder and Fuse Patch Tokens Spatially

## Context

The current `CLIPSemanticBackbone` extracts a single CLS token (768-dim) per camera view using CLIP ViT-B/16. CLIP is trained with a contrastive objective on 400M noisy short-caption pairs — the CLS token is a coarse global descriptor that aligns an image with a short sentence.

Qwen2-VL's visual encoder is trained to support dense VQA, OCR, spatial reasoning, and object counting. Its patch tokens must encode per-region semantic detail (traffic light colour, sign text, brake light state, pedestrian pose) to answer fine-grained questions — making them richer than CLIP's single CLS for driving-relevant semantics.

The additional design goal is to fuse Qwen's spatial patch tokens directly into the per-view `_spatial_decoder`, so the spatial attention can attend to both geometric (SwinTransformer) tokens and semantic (Qwen) tokens in the same cross-attention operation.

---

## Why Qwen Patch Tokens Are Richer

| | CLIP ViT-B/16 | Qwen2-VL-2B ViT |
|---|---|---|
| Training objective | Contrastive image↔short caption | Generative VQA, OCR, dense captioning |
| Output | 1 CLS token per image | 256 spatial patch tokens per image |
| Spatial granularity | None (single vector) | 16×16 grid over 448×448 input |
| Semantic depth | Global scene match to caption | Per-region fine-grained attributes |

---

## Architecture Overview

```
Camera images (6 views)
    │
    ├──► SwinTransformer3D ──► [B, 6, 256, H, W]  (geometric spatial features)
    │
    └──► QwenSemanticBackbone (frozen) ──► patches [B, 6, 256_patches, 1536]
                                           pooled  [B, 6, 1536]
                                               │
                          ┌────────────────────┘
                          ▼
                   WaypointHead
                   ┌─────────────────────────────────────────────────────────┐
                   │  For each view i:                                        │
                   │    memory = cat([swin_tokens, qwen_patch_proj(patches)]) │
                   │    spatial_view_feat[i] = _spatial_decoder[i](query,     │
                   │                                               memory)    │
                   │                                                          │
                   │  wp_attn(waypoint_query, cat([spatial_feat, sem_token])) │
                   │                                                          │
                   │  wm_prediction: action_aware_encoder(                   │
                   │      cat([view_feat, pooled_sem_proj, waypoints]))       │
                   └─────────────────────────────────────────────────────────┘
```

---

## Files to Create / Modify

### 1. NEW — `projects/mmdet3d_plugin/models/backbones/qwen_semantic_backbone.py`

- Register as `@BACKBONES.register_module()` named `QwenSemanticBackbone`
- Load `Qwen2VLForConditionalGeneration("Qwen/Qwen2-VL-2B-Instruct")`, keep only `model.visual`
- Input: `[B, 3, H, W]` with mmdet ImageNet normalization
- Preprocessing: undo mmdet normalization → resize to 448×448 → apply Qwen normalization (nearly same as ImageNet: mean=[0.485,0.456,0.406], std=[0.228,0.224,0.225])
- Construct Qwen's packed input format for fixed 448×448 (`grid_thw = [[1, 32, 32]]` per image)
- Use **post-merger** output (256 tokens/image, 1536-dim for 2B) — avoids the 1024-token cost of pre-merger while still using Qwen's learned spatial compression
- Return tuple: `(pooled [B, 1536], patches [B, 256, 1536])` where `pooled = patches.mean(dim=1)`
- Frozen by default; optional LoRA via `peft`

### 2. MODIFY — `projects/mmdet3d_plugin/models/backbones/__init__.py`

Add import and `__all__` entry for `QwenSemanticBackbone` (follow the existing pattern for `CLIPSemanticBackbone`).

### 3. MODIFY — `projects/mmdet3d_plugin/LAW/LAW.py`

In both `forward_train` (line ~193) and `simple_test` (line ~309), replace the flat reshape with:

```python
result = self.semantic_img_backbone(flat_img)
if isinstance(result, tuple):
    pooled, patches = result
    semantic_feat = {
        'pooled':   pooled.reshape(B, N_view, -1),          # [B, N_view, 1536]
        'patches':  patches.reshape(B, N_view, patches.shape[-2], -1),  # [B, N_view, 256, 1536]
    }
else:
    semantic_feat = result.reshape(B, N_view, -1)  # CLIP backward-compat
```

No other changes to LAW.py needed — `semantic_feat` is forwarded as-is.

### 4. MODIFY — `projects/mmdet3d_plugin/LAW/dense_heads/waypoint_query_decoder.py`

**New `__init__` parameters** (lines ~53-55):
```python
use_qwen_spatial_fusion=False,
qwen_vit_dim=1536,
```

**New layer** (when `use_qwen_spatial_fusion=True`):
```python
self.qwen_patch_proj = nn.Linear(qwen_vit_dim, hidden_channel)
```

**`semantic_proj` stays** but now sized `vlm_hidden_channel (1536) → hidden_channel (256)`.

**`action_aware_encoder` input dim stays 524** — the pooled Qwen feature is still projected to 256 before concatenation, so the encoder width is unchanged.

**In `forward()` (lines ~226-230)**: unpack `semantic_feat` dict, then:
```python
qwen_patches = semantic_feat['patches'] if isinstance(semantic_feat, dict) else None
pooled_sem   = semantic_feat['pooled']  if isinstance(semantic_feat, dict) else semantic_feat

for i in range(self.num_views):
    if self.use_qwen_spatial_fusion and qwen_patches is not None:
        qwen_proj_i = self.qwen_patch_proj(qwen_patches[:, i])   # [B, 256, 256]
        memory_i = torch.cat([img_feat_emb[:, i], qwen_proj_i], dim=1)  # [B, ~576, 256]
    else:
        memory_i = img_feat_emb[:, i]
    spatial_view_feat[:, i] = self._spatial_decoder[i](init_view_query_feat[:, i], memory_i)
```

**In `forward()` semantic memory block** (lines ~234-241): use `pooled_sem` in place of `semantic_feat` for `semantic_proj` and `memory_feat` concatenation — same logic as now.

**In `wm_prediction()`**: same as current — `proj_semantic_feat` is the projected pooled feature; no change needed.

### 5. NEW — `projects/configs/law/default_with_qwen_semantic.py`

```python
_base_ = ['default.py']
model = dict(
    _delete_=True,
    type='LAW',
    use_grid_mask=True,
    video_test_mode=True,
    use_multi_view=True,
    use_swin=True,
    use_semantic=True,
    semantic_img_backbone=dict(
        type='QwenSemanticBackbone',
        model_name='Qwen/Qwen2-VL-2B-Instruct',
        frozen=True,
        use_lora=False,
    ),
    swin_input_channel=768,
    hidden_channel=256,
    img_backbone=dict(  # same as default
        type='SwinTransformer3D', arch='tiny', ...
    ),
    pts_bbox_head=dict(
        type='WaypointHead',
        num_proposals=6,
        num_views=6,
        hidden_channel=256,
        num_heads=8,
        dropout=0.1,
        use_wm=True,
        num_traj_modal=3,
        use_semantic=True,
        vlm_hidden_channel=1536,       # Qwen2-VL-2B post-merger dim
        use_qwen_spatial_fusion=True,
        qwen_vit_dim=1536,
    ),
)
find_unused_parameters = True
```

---

## Prerequisite

Qwen2-VL requires **PyTorch ≥ 2.0**. The current project pins to 1.9.1. The stack (PyTorch, MMCV, mmdet3d) must be upgraded before this feature can run. This is a hard dependency — plan accordingly.

---

## Verification

1. Single-GPU debug run:
   ```bash
   PYTHONPATH="$(pwd)" python tools/train.py \
       projects/configs/law/default_with_qwen_semantic.py \
       --work-dir work_dirs/debug_qwen --deterministic
   ```
2. Confirm `semantic_feat['patches']` shape is `[B, 6, 256, 1536]` and `pooled` is `[B, 6, 1536]` via a debug print in `forward_train`.
3. Check `loss_rec` and `loss_waypoint` are finite (non-NaN) for the first 10 iterations.
4. Verify `qwen_patch_proj` receives gradients and `vision_model` parameters do NOT (frozen check).
