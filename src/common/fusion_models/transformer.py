import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from src.common.modules.transformer import *
from copy import deepcopy


class Transformer(nn.Module):
    """
    baseline fusion module:
    No interaction loss and no ensemble.
    """

    def __init__(
        self,
        num_modalities,
        num_patches,
        hidden_dim,
        output_dim,
        num_layers,
        num_layers_pred,
        num_experts,
        num_routers,
        top_k,
        num_heads=2,
        dropout=0.5,
        mlp_sparse=False,
        gate="GShardGate",
    ):
        super(Transformer, self).__init__()
        layers = []
        layers.append(
            TransformerEncoderLayer(
                num_experts,
                num_routers,
                hidden_dim,
                num_heads,
                dropout=dropout,
                hidden_times=2,
                mlp_sparse=mlp_sparse,
                top_k=top_k,
                gate=gate,
            )
        )
        for j in range(num_layers - 1):
            tmp = (mlp_sparse) & (j % 2 == 1)
            layers.append(
                TransformerEncoderLayer(
                    num_experts,
                    num_routers,
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    hidden_times=2,
                    mlp_sparse=tmp,
                    top_k=top_k,
                    gate=gate,
                )
            )
        layers.append(Linear(hidden_dim * num_modalities, output_dim))

        self.network = nn.Sequential(*layers)
        self.pos_embed = None
        

    def _sinusoid_pe(self, L, H, device):
        pos = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)  # (L,1)
        i   = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(0)  # (1,H)
        angle = pos / torch.pow(10000, (2 * (i//2)) / H)
        pe = torch.zeros(L, H, device=device)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe  # (L,H)
    
    def forward(self, inputs, masks=None, return_latent=False):
        # inputs: [ (B,T_lang,H), (B,T_img,H), ... ]
        # masks:  [ (B,T_lang),   (B,T_img),   ... ]  (language: 0/1, image: all 1)
 
        chunk_size = [inp.shape[1] for inp in inputs]
        x = torch.cat(inputs, dim=1)                  # (B, sumT, H)
        B, L, H = x.shape

        pe = self._sinusoid_pe(L, H, x.device).unsqueeze(0)  # (1,L,H)
        x = x + pe
        
        masks_list = None
        if masks is not None:
            if isinstance(masks, (list, tuple)):
                masks_list = [m.to(x.device).float() if m is not None else None for m in masks]
            elif isinstance(masks, dict) and "language" in masks and "mask" in masks["language"]:
                lang_mask = masks["language"]["mask"].to(x.device).float()     # (B,T_lang)
                img_len = chunk_size[1] if len(chunk_size) > 1 else 0
                masks_list = [lang_mask]
                if img_len > 0:
                    masks_list.append(torch.ones((B, img_len), device=x.device))
            # Match input length (pad missing entries with ones).
            if masks_list is not None and len(masks_list) < len(chunk_size):
                for k in range(len(masks_list), len(chunk_size)):
                    masks_list.append(torch.ones((B, chunk_size[k]), device=x.device))

        # --- Optional: zero out padded positions at input stage ---
        if masks_list is not None:
            big_mask = torch.cat([m if m is not None else torch.ones((B, cs), device=x.device)
                                  for m, cs in zip(masks_list, chunk_size)], dim=1).unsqueeze(-1)  # (B,L,1)
            x = x * big_mask

        x = list(torch.split(x, chunk_size, dim=1))
        for i in range(len(self.network) - 1):
            x = self.network[i](x, masks=masks_list)              # pass masks through each transformer block

        # --- Masked mean pooling (works for both list and dict paths) ---
        if masks_list is not None:
            outs = []
            for seg, m in zip(x, masks_list):
                m = m.to(seg.device).float().unsqueeze(-1)     # (B,T,1)
                num = (seg * m).sum(dim=1)                     # (B,H)
                den = m.sum(dim=1).clamp_min(1.0)              # (B,1)
                outs.append(num / den)
            x = torch.cat(outs, dim=1)
        else:
            x = torch.cat([seg.mean(dim=1) for seg in x], dim=1)


        if return_latent:
            latent = x
        x = self.network[-1](x)     # Linear(hidden_dim*num_modalities -> output_dim)
        return (x, latent) if return_latent else x

    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            # print(mn)
            if hasattr(mm, "all_gates"):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f"{i}"].get_loss()
                    if i_loss is None:
                        print(
                            f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice."
                        )
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)
