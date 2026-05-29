"""
Regression tests for MetricsDAG._count_accuracy.

Issue #188: TPR > 1.0 when B_est contains bidirectional edges due to
double-counting in pred_und construction (both flatnonzero(B_est==-1)
and flatnonzero(B_est.T==-1) are included, yielding two index entries
for a single undirected edge).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from castle.metrics import MetricsDAG


class TestMetricsDAGTPRBounded:
    """TPR must stay within [0, 1] for all valid inputs."""

    def test_tpr_bounded_bidirectional_single_edge(self):
        """
        Minimal repro from issue #188.
        B_true: single edge 0->1 (1 true edge).
        B_est:  bidirectional 0<->1.
        Pre-fix: TPR=2.0 (double-counted undirected TP).
        Post-fix: TPR<=1.0.
        """
        B_true = np.array([[0, 1],
                           [0, 0]])
        B_est = np.array([[0, 1],
                          [1, 0]])

        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['tpr'] <= 1.0, (
            f"TPR must be <= 1.0, got {metrics['tpr']}"
        )

    def test_tpr_correct_value_bidirectional(self):
        """
        For 1 true edge and a bidirectional estimate covering that skeleton edge,
        the undirected edge should count as 1 TP (favorable treatment).
        TPR = 1/1 = 1.0 exactly.
        """
        B_true = np.array([[0, 1],
                           [0, 0]])
        B_est = np.array([[0, 1],
                          [1, 0]])

        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['tpr'] == 1.0, (
            f"Expected TPR=1.0 for bidirectional edge covering true skeleton, "
            f"got {metrics['tpr']}"
        )

    def test_tpr_bounded_larger_graph_multiple_bidirectional(self):
        """
        Larger graph with multiple bidirectional edges cannot yield TPR > 1.0.
        B_true: 0->1, 1->2, 2->3 (3 edges).
        B_est:  all bidirectional (worst-case double-counting scenario).
        """
        d = 4
        B_true = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        # Make all estimated edges bidirectional
        B_est = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ])
        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['tpr'] <= 1.0, (
            f"TPR must be <= 1.0 with multiple bidirectional edges, "
            f"got {metrics['tpr']}"
        )

    def test_tpr_undirected_correct_no_overlap(self):
        """
        When estimated undirected edge does NOT overlap with any true edge,
        it should contribute 0 TP (not inflate count).
        B_true: 0->1
        B_est:  bidirectional 2<->3 (completely disjoint from true edge)
        TPR should be 0.0.
        """
        B_true = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        B_est = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['tpr'] == 0.0, (
            f"Expected TPR=0.0 for disjoint bidirectional estimate, "
            f"got {metrics['tpr']}"
        )


class TestMetricsDAGNNZAndFPR:
    """nnz and fpr must also be correct after the fix (issue #188 inflated both)."""

    def test_nnz_one_bidirectional_edge_counts_as_one(self):
        """
        nnz = len(pred) + len(pred_und).  One undirected edge is one edge — nnz must be 1,
        not 2 as master produced (pred_und double-counted the mirror index).
        B_true: 0->1
        B_est:  0<->1 (bidirectional)
        """
        B_true = np.array([[0, 1],
                           [0, 0]])
        B_est = np.array([[0, 1],
                          [1, 0]])
        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['nnz'] == 1, (
            f"Expected nnz=1 for one bidirectional edge, got {metrics['nnz']}"
        )

    def test_fpr_fp_undirected_edge_not_inflated(self):
        """
        fpr = (reverse + false_pos) / cond_neg_size.
        One undirected FP edge should count as 1 false positive (not 2).
        B_true: 0->1 (1 true edge, d=4 → cond_neg_size=5)
        B_est:  2<->3 (completely disjoint bidirectional FP)
        Expected fpr = 1 / 5 = 0.2, not 0.4 as master produced.
        """
        B_true = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        B_est = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['fpr'] == round(1 / 5, 4), (
            f"Expected fpr=0.2 for one FP undirected edge (d=4, 1 true edge), "
            f"got {metrics['fpr']}"
        )
        assert metrics['nnz'] == 1, (
            f"Expected nnz=1 for one FP undirected edge, got {metrics['nnz']}"
        )


class TestMetricsDAGDirectedDAGRegression:
    """For pure directed DAG inputs (no undirected edges), the fix must be a no-op."""

    def test_directed_dag_perfect_estimate_unchanged(self):
        """Perfect directed DAG estimate: all metrics identical to pre-fix behaviour."""
        B_true = np.array([
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        B_est = B_true.copy()
        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['tpr'] == 1.0
        assert metrics['fdr'] == 0.0
        assert metrics['fpr'] == 0.0
        assert metrics['shd'] == 0
        assert metrics['nnz'] == int(B_true.sum())

    def test_directed_dag_with_reversed_edge_unchanged(self):
        """Reversed directed edge: shd=1, tpr correct, nnz unchanged by fix."""
        B_true = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])
        # Reverse edge 0->1 to 1->0
        B_est = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0],
        ])
        metrics = MetricsDAG(B_est=B_est, B_true=B_true).metrics
        assert metrics['shd'] == 1, f"Expected shd=1, got {metrics['shd']}"
        assert metrics['nnz'] == int(B_est.sum()), (
            f"nnz should equal number of directed edges in estimate, "
            f"got {metrics['nnz']}"
        )
