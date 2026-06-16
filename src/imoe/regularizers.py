import torch
import torch.nn.functional as F


BARLOW_DEFAULT_WARMUP_EPOCHS = 5


def canonicalize_regularizer(name):
    name = str(name or "none").lower().replace("_", "-")
    aliases = {
        "": "none",
        "no": "none",
        "off": "none",
        "decor": "rep-cos",
        "decorrelation": "rep-cos",
        "repcos": "rep-cos",
        "rep-cosine": "rep-cos",
        "barlow": "rep-barlow",
        "repbarlow": "rep-barlow",
        "cka": "rep-cka",
        "repcka": "rep-cka",
    }
    return aliases.get(name, name)


def _first_float_arg(args, names, default):
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return float(value)
    return float(default)


def regularizer_weight_from_args(args, regularizer):
    regularizer = canonicalize_regularizer(regularizer)
    if getattr(args, "regularizer_weight", None) is not None:
        return float(args.regularizer_weight)
    if regularizer == "rep-cos":
        return _first_float_arg(args, ["rep_cos_loss_weight", "decor_loss_weight"], 3e-3)
    if regularizer == "rep-barlow":
        return _first_float_arg(args, ["rep_barlow_loss_weight", "barlow_loss_weight"], 10.0)
    if regularizer == "rep-cka":
        return _first_float_arg(args, ["rep_cka_loss_weight", "cka_loss_weight"], 0.0)
    return 0.0


def regularizer_warmup_epochs_from_args(args, regularizer):
    regularizer = canonicalize_regularizer(regularizer)
    explicit = getattr(args, "regularizer_warmup_epochs", None)
    if explicit is not None:
        return int(explicit)
    if regularizer == "rep-barlow":
        return BARLOW_DEFAULT_WARMUP_EPOCHS
    return 0


def regularizer_save_subdir(args, regularizer):
    regularizer = canonicalize_regularizer(regularizer)
    explicit = getattr(args, "save_subdir", "")
    if explicit:
        return explicit
    defaults = {
        "rep-cos": "rep_cos",
        "rep-barlow": "rep_barlow",
        "rep-cka": "rep_cka",
        "none": "no_regularizer",
    }
    return defaults.get(regularizer, regularizer_slug(regularizer))


def regularizer_slug(regularizer):
    return canonicalize_regularizer(regularizer).replace("-", "_")


def lambda_warmup_scale(epoch, warmup_epochs):
    if warmup_epochs <= 1:
        return 1.0
    return min(1.0, max(0.0, epoch / float(warmup_epochs - 1)))


def compute_rep_cos_loss(all_latents, routing_weights=None, normalize=True):
    """
    Rep-Cos: pairwise cosine decorrelation between I2MoE expert latents.
    """
    num_experts = len(all_latents)
    if num_experts <= 1:
        return all_latents[0].new_zeros(())

    latents = [
        F.normalize(z, p=2, dim=1) if normalize else z
        for z in all_latents
    ]

    terms = []
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            sim = (latents[i] * latents[j]).sum(dim=1)
            if routing_weights is not None:
                weight = 0.5 * (routing_weights[:, i] + routing_weights[:, j])
                sim = sim * weight
            terms.append((sim ** 2).mean())
    return torch.stack(terms).mean() if terms else all_latents[0].new_zeros(())


def _standardize_batch(z, eps=1e-5):
    z = z - z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, unbiased=False, keepdim=True)
    return z / (std + eps)


def _offdiag_square_mean(c):
    d = c.shape[0]
    if d <= 1:
        return c.new_zeros(())
    offdiag = c - torch.diag_embed(torch.diagonal(c))
    return offdiag.pow(2).sum() / (d * d - d)


def compute_expert_barlow_loss(all_latents, eps=1e-5):
    """
    Rep-Barlow: off-diagonal Barlow-style redundancy reduction between expert latents.
    """
    num_experts = len(all_latents)
    if num_experts <= 1:
        return all_latents[0].new_zeros(())

    z_norm = [_standardize_batch(z, eps=eps) for z in all_latents]
    terms = []
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            corr = (z_norm[i].T @ z_norm[j]) / z_norm[i].shape[0]
            terms.append(_offdiag_square_mean(corr))
    return torch.stack(terms).mean() if terms else all_latents[0].new_zeros(())


def _center_features(z):
    return z - z.mean(dim=0, keepdim=True)


def _linear_cka(x, y, eps=1e-8):
    x = _center_features(x)
    y = _center_features(y)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    hsic = (xty * xty).sum()
    norm_x = torch.linalg.norm(xtx, ord="fro")
    norm_y = torch.linalg.norm(yty, ord="fro")
    return hsic / (norm_x * norm_y + eps)


def compute_expert_cka_loss(all_latents):
    num_experts = len(all_latents)
    if num_experts <= 1:
        return all_latents[0].new_zeros(())

    terms = []
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            terms.append(_linear_cka(all_latents[i], all_latents[j]))
    return torch.stack(terms).mean() if terms else all_latents[0].new_zeros(())


def _zero_like(reference_tensor, all_latents=None):
    if reference_tensor is not None:
        return reference_tensor.new_zeros(())
    if all_latents:
        return all_latents[0].new_zeros(())
    return torch.tensor(0.0)


def compute_regularizer_loss(
    regularizer,
    *,
    all_latents,
    reference_tensor=None,
):
    regularizer = canonicalize_regularizer(regularizer)
    if regularizer == "none":
        return _zero_like(reference_tensor, all_latents), {}
    if regularizer == "rep-cos":
        loss = compute_rep_cos_loss(all_latents, routing_weights=None)
        return loss, {"rep_cos": loss}
    if regularizer == "rep-barlow":
        loss = compute_expert_barlow_loss(all_latents)
        return loss, {"off": loss}
    if regularizer == "rep-cka":
        loss = compute_expert_cka_loss(all_latents)
        return loss, {"cka": loss}
    raise ValueError(f"Unsupported regularizer: {regularizer}")


def regularizer_log_token(regularizer):
    regularizer = canonicalize_regularizer(regularizer)
    return {
        "none": "R",
        "rep-cos": "Rep-Cos",
        "rep-barlow": "Rep-Barlow",
        "rep-cka": "Rep-CKA",
    }.get(regularizer, regularizer.upper())


def format_regularizer_details(details):
    if not details:
        return ""
    parts = []
    for key, value in details.items():
        if torch.is_tensor(value):
            value = float(value.detach().item())
        parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)
