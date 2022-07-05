import numpy as np
import matplotlib.pyplot as plt


def count_accuracy(W_true, W_est, W_und=None):
    """
    Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        W_true: ground truth graph
        W_est: predicted graph
        W_und: predicted undirected edges in CPDAG, asymmetric

    Returns in dict:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

    Referred from:
    - https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    B_true = W_true != 0
    B = W_est != 0
    B_und = None if W_und is None else W_und
    d = B.shape[0]

    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {
        'fdr': fdr,
        'tpr': tpr,
        'fpr': fpr,
        'shd': shd,
        'pred_size': pred_size
    }


def plot_recovered_graph(W_est, W, save_name=None):
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

    ax1.set_title('recovered_graph')
    map1 = ax1.imshow(W_est, cmap='Greys', interpolation='none')
    fig.colorbar(map1, ax=ax1)

    ax2.set_title('true_graph')
    map2 = ax2.imshow(W, cmap='Greys', interpolation='none')
    fig.colorbar(map2, ax=ax2)

    plt.show()

    if save_name is not None:
        fig.savefig(save_name)
