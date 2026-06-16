import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.common.model_paths import resolve_pretrained_model_path

class BERTTextEncoder(nn.Module):
    def __init__(self, name_or_path="models/roberta-base",
                 out_dim=256, max_len=320, freeze=True, local_only=False):
        super().__init__()
        name_or_path = resolve_pretrained_model_path(name_or_path, "TEXT_ENCODER_NAME_OR_PATH")
        self.backbone = AutoModel.from_pretrained(name_or_path, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, local_files_only=local_only)

        # Fallback in case the tokenizer has no pad token.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else self.tokenizer.unk_token

        hid = self.backbone.config.hidden_size
        self.proj = nn.Linear(hid, out_dim)
        self.max_len = max_len

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, texts):  # List[str]
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        enc = {k: v.to(self.proj.weight.device) for k, v in enc.items()}
        out = self.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        feats = self.proj(out.last_hidden_state)          # (B, T, out_dim)
        mask  = enc["attention_mask"].float()             # (B, T)
        return feats, mask
