"""Quantile subset for the training loss.

The backbone's native pinball loss is computed over a quantile grid that does not
match the quantiles GIFT-Eval actually scores. PatchTST-FM predicts 99 quantiles,
while only 9 of them (0.1 .. 0.9) enter the score, so 90 quantiles receive gradient
without ever affecting the metric. This module supplies the indices and the rescale

factor needed to restrict the loss to the scored quantiles. The sum over the subset is
multiplied by K_full / K_sub so the loss magnitude — and hence the effective learning
"""
from typing import List, Sequence

GIFT_EVAL_Q = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def gifteval_subset_idx(levels: Sequence[float], tol: float = 1e-6) -> List[int]:
    """Locate the 9 GIFT-Eval quantiles inside the model's native `levels` grid.

    Fail-fast: each of the 9 must match exactly once, otherwise ValueError. Falling back
    to a nearest-neighbour quantile silently would change what the experiment measures.
    """
    lv = [float(v) for v in levels]
    idx: List[int] = []
    for q in GIFT_EVAL_Q:
        hit = [i for i, mq in enumerate(lv) if abs(mq - q) < tol]
        if len(hit) != 1:
            raise ValueError(
                f"quantile subset: level {q} matched {len(hit)} times among {len(lv)} native levels "
                f"(tol={tol}); cannot form an exact subset. levels={lv[:12]}{'...' if len(lv) > 12 else ''}")
        idx.append(hit[0])
    return idx


def subset_rescale(k_full: int, k_sub: int) -> float:
    """Rescale factor putting a sum over k_sub quantiles on the scale of k_full."""
    if k_sub <= 0:
        raise ValueError("k_sub must be > 0")
    return k_full / k_sub
