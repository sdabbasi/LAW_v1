_base_ = ['default.py']

# Replace the full model dict so no leftover keys from the base bleed through.
model = dict(
    _delete_=True,
    type='LAW',
    use_grid_mask=True,
    video_test_mode=True,
    use_multi_view=True,
    use_swin=True,
    use_semantic=True,
    semantic_img_backbone=dict(
        type='CLIPSemanticBackbone',
        model_name='openai/clip-vit-base-patch16',
        frozen=True,
        use_lora=False,
    ),
    swin_input_channel=768,
    hidden_channel=256,
    img_backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.2,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True
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
        vlm_hidden_channel=768,
    ),
)

# semantic_proj is skipped in history-frame calls inside obtain_history_feat
# but is always reached in the current-frame path, so it always receives
# gradients. True is kept as a safety net for future changes.
find_unused_parameters = True
