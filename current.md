# Architecture Design Notes

## Queries vs. Content Features: Why CLIP Semantic Features Are Not Learnable Tokens

### The Core Distinction

There are two fundamentally different roles in the LAW attention pipeline:

| Role | Examples | Learnable? | Why |
|---|---|---|---|
| **Queries** — learn *how* to extract information | `view_query_feat`, `waypoint_query_feat` | Yes — `nn.Parameter` | Input-independent; learn to be good probes regardless of scene content |
| **Content features** — carry *what is* in the scene | Swin image features, CLIP CLS token | No — computed from input | Must change per-scene to carry actual visual content |

### Why `view_query_feat` and `waypoint_query_feat` Are Learnable

These are query tokens in the transformer sense. They start random and through training learn to extract spatial and trajectory-relevant information. The same query vector is used for every scene — they encode task knowledge ("what to look for"), not scene content.

### Why CLIP Features Cannot Be Replaced with Learnable Tokens

`CLIPSemanticBackbone` returns the CLS token of a frozen CLIP ViT per camera view (`last_hidden_state[:, 0]`, shape `[B, 768]`). Its value is that it is image-conditioned: it encodes what is actually visible — a red vs. green traffic light, an active brake light, a pedestrian's pose.

Replacing this with an `nn.Parameter` would yield the same 768-dim vector for every scene, destroying all semantic signal. You would just be learning a bias term.

### A Valid Architectural Improvement (Not Yet Implemented)

The current implementation discards CLIP's spatial patch tokens:

```python
# clip_semantic_backbone.py:109
return outputs.last_hidden_state[:, 0]  # only CLS — 196 patch tokens are dropped
```

A richer design: introduce **learnable query tokens that cross-attend into CLIP's patch tokens** (`last_hidden_state[:, 1:]`, shape `[B, 196, 768]`) as keys/values. This is analogous to how `view_query_feat` attends into Swin features — learnable queries extracting content from a frozen, scene-conditioned feature map. This pattern is known as visual prompt tuning or perceiver-style cross-attention on top of CLIP.

This would give you both:
- Learnable task-specific query tokens (what to extract from CLIP)
- Scene-conditioned content (the actual patch features from the image)

### Relevant Files

- `projects/mmdet3d_plugin/models/backbones/clip_semantic_backbone.py` — CLIP feature extraction
- `projects/mmdet3d_plugin/LAW/dense_heads/waypoint_query_decoder.py` — `view_query_feat`, `waypoint_query_feat`, `semantic_proj`, and where CLIP features are injected (lines ~234–244)
- `projects/configs/law/default_with_semantic.py` — config enabling the semantic path (`use_semantic=True`)
