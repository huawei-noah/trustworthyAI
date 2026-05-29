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
