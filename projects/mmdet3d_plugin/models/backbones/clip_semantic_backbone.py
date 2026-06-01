import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES

# torch.frombuffer was added in 1.10; patch it for 1.9.x so safetensors can load
if not hasattr(torch, 'frombuffer'):
    import numpy as np
    def _frombuffer_patch(buffer, *args, **kwargs):
        dtype = kwargs.get('dtype') or (args[0] if args else torch.uint8)
        arr = np.frombuffer(buffer, dtype=np.uint8).copy()
        return torch.from_numpy(arr).view(dtype)
    torch.frombuffer = _frombuffer_patch


@BACKBONES.register_module()
class CLIPSemanticBackbone(nn.Module):
    """
    Per-view semantic feature extractor using CLIP's frozen vision encoder.

    Takes images with the standard mmdet ImageNet normalization used in NuScenes
    pipelines, renormalizes them to CLIP statistics, resizes to 224×224, and
    returns the CLS token for each image.  The CLS token is language-grounded
    (CLIP training) so it captures semantic states like traffic-light colour,
    brake-light activation, and pedestrian pose that pure visual backbones
    encode only implicitly.

    Input:  [B, C, H, W]   — ImageNet-normalized, arbitrary spatial resolution
    Output: [B, hidden_size] — CLS-token feature (hidden_size=768 for ViT-B/16)

    Usage in multi-view setting (caller is responsible for view batching):
        flat = cur_img.reshape(B * N_view, C, H, W)
        sem  = backbone(flat).reshape(B, N_view, hidden_size)
    """

    def __init__(
        self,
        model_name='openai/clip-vit-base-patch16',
        frozen=True,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()

        try:
            from transformers import CLIPVisionModel
        except ImportError:
            raise ImportError('transformers is required for CLIPSemanticBackbone')

        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size  # 768 for ViT-B/16

        if frozen:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as e:
                raise ImportError(
                    'peft is required when use_lora=True (also check that '
                    'accelerate is compatible with your setuptools). '
                    f'Original error: {e}'
                )
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
                lora_dropout=lora_dropout,
                bias='none',
            )
            self.vision_model = get_peft_model(self.vision_model, lora_cfg)

        # mmdet NuScenes normalization: pixel_norm = (pixel/255 - mean) / std
        self.register_buffer('img_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # CLIP normalization
        self.register_buffer('clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, img):
        """
        img: [B, C, H, W] with mmdet ImageNet normalization.
        Returns: [B, hidden_size] — CLS token per image.
        """
        img = img.float()

        # Undo mmdet normalization → [0, 1] pixel range
        img = img * self.img_std + self.img_mean
        # Apply CLIP normalization
        img = (img - self.clip_mean) / self.clip_std

        # CLIP ViT requires exactly 224×224
        if img.shape[-2] != 224 or img.shape[-1] != 224:
            img = F.interpolate(
                img, size=(224, 224), mode='bilinear', align_corners=False
            )

        outputs = self.vision_model(pixel_values=img)
        # last_hidden_state: [B, num_patches+1, hidden_size]
        # index 0 is the CLS token before the pooler projection
        return outputs.last_hidden_state[:, 0]  # [B, hidden_size]
