import os
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_pretrained_model_path(name_or_path, env_var=None):
    """Resolve a pretrained model from env override, local folders, or HF model id."""
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]

    candidate = Path(str(name_or_path)).expanduser()
    if candidate.exists():
        return str(candidate)

    repo_root = _repo_root()
    model_name = str(name_or_path).strip("/")
    short_name = model_name.split("/")[-1]
    search_roots = [repo_root / "models"]

    for root in search_roots:
        for rel in (model_name, short_name):
            path = root / rel
            if path.exists():
                return str(path)

    return name_or_path
