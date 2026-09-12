"""Build a tiny Hugging Face snapshot of the Tabby backbone.

The release ships no checkpoints, and ``*.safetensors`` is git-ignored, so the
checkpoint-loading tests generate their own sample snapshot instead of reading
one from the repository.  The layout produced here is the one the loaders
accept in production -- a directory holding ``config.json`` and
``model.safetensors`` with ``backbone.*`` tensor names, exactly what
``save_pretrained`` writes and what the published ``patchtst-fm-r1`` snapshot
looks like -- only with a ~35k-parameter configuration instead of the 165k-step
release weights.

Run standalone to materialize a snapshot for manual inspection::

    python tests/make_tiny_hf_snapshot.py /tmp/tiny_patchtstfm
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch

from tabby.models.configuration_patchtst_fm import PatchTSTFMConfig
from tabby.models.modeling_patchtst_fm import PatchTSTFMForPrediction


# Small but structurally faithful: >1 transformer block, d_model divisible by
# n_head, context divisible by d_patch.  ``pretrain_mask_cont`` is lowered from
# its release value (8) because ``PatchTSTFMForPrediction.forward`` floors the
# forecast span at ``d_patch * max(pretrain_mask_cont, 2)``, which has to stay
# well inside this shrunken context window.
TINY_CONFIG: Dict[str, Any] = {
    "context_length": 64,
    "prediction_length": 8,
    "d_patch": 8,
    "d_model": 32,
    "n_head": 4,
    "n_layer": 2,
    "num_quantile": 9,
    "pretrain_mask_cont": 2,
    "dropout": 0.0,
}

# The GIFT-Eval wrapper requires context_length + prediction_length to fit inside
# the backbone window, and its zero-shot path fixes prediction_length at 96, so
# the benchmark tests need a wider window than TINY_CONFIG provides.
GIFT_EVAL_CONFIG: Dict[str, Any] = {**TINY_CONFIG, "context_length": 256}

# Any fixed value works; pinning it keeps a regenerated snapshot reproducible.
SEED = 20240917


def build_tiny_config(**overrides) -> PatchTSTFMConfig:
    return PatchTSTFMConfig(**{**TINY_CONFIG, **overrides})


def build_tiny_model(seed: int = SEED, **overrides) -> PatchTSTFMForPrediction:
    """Return a deterministically initialized tiny prediction model."""
    torch.manual_seed(seed)
    model = PatchTSTFMForPrediction(build_tiny_config(**overrides))
    return model.eval()


def write_tiny_snapshot(dest: Path, seed: int = SEED, **overrides) -> Path:
    """Save a tiny model as an HF snapshot directory and return that path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    build_tiny_model(seed, **overrides).save_pretrained(dest, safe_serialization=True)
    return dest


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(__file__).name} OUTPUT_DIR", file=sys.stderr)
        return 2
    path = write_tiny_snapshot(Path(argv[1]))
    written = sorted(p.name for p in path.iterdir())
    print(f"wrote {path}: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
