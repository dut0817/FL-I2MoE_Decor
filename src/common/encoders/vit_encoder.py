import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, AutoFeatureExtractor

from src.common.model_paths import resolve_pretrained_model_path

class ViTImageEncoder(nn.Module):
    def __init__(self, name_or_path="models/clip-vit-base-patch16",
                 out_dim=256, freeze=True, use_cls=False, local_only=False):
        super().__init__()
        name_or_path = resolve_pretrained_model_path(name_or_path, "IMAGE_ENCODER_NAME_OR_PATH")
        self.backbone = AutoModel.from_pretrained(
            name_or_path, local_files_only=local_only
        )  # expected to be a CLIPModel-family checkpoint
        # CLIPModel has `.vision_model`; otherwise fallback to backbone itself.
        self.vit = getattr(self.backbone, "vision_model", self.backbone)
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                name_or_path, local_files_only=local_only, use_fast=True
            )
        except Exception:
            # Compatibility fallback for older checkpoints that only provide
            # `feature_extractor_type` in preprocessor_config.json.
            self.processor = AutoFeatureExtractor.from_pretrained(
                name_or_path, local_files_only=local_only
            )
        hid = self.vit.config.hidden_size
        self.use_cls = use_cls
        self.proj = nn.Linear(hid, out_dim)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, images):  # images: List[PIL] or Tensor(B,3,H,W) (raw)
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            images = [Image.open(p).convert("RGB") for p in images]
        
        if isinstance(images, list):
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        elif torch.is_tensor(images):
            # Route raw tensors through the processor to keep normalization consistent.
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        else:
            raise TypeError("images must be List[PIL.Image] or Tensor(B,3,H,W)")
        pixels = pixels.to(self.proj.weight.device)

        out = self.vit(pixel_values=pixels, output_hidden_states=False)
        feats = out.last_hidden_state  # (B, 1+T_img, H)
        if not self.use_cls:
            feats = feats[:, 1:, :]     # drop CLS token, keep patch tokens only
        feats = self.proj(feats)        # (B, T_img, out_dim)
        m_img = torch.ones(feats.size()[:2], device=feats.device)  # (B, T_img)
        return feats, m_img
