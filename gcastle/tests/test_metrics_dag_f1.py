"""
Regression tests for MetricsDAG precision, recall and F1 when there are no
true positives (F1 used to be NaN from 0 / 0).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from castle.metrics import MetricsDAG


B_TRUE = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 0]])


@pytest.mark.parametrize("B_est", [
    np.array([[0, 0, 1],
              [0, 0, 0],
              [0, 0, 0]]),  # only a wrong edge
    B_TRUE.T.copy(),  # all edges reversed
    np.zeros((3, 3), dtype=int),  # empty estimate
])
def test_no_true_positive_gives_zero_scores(B_est):
    metrics = MetricsDAG(B_est=B_est, B_true=B_TRUE).metrics
    assert metrics['precision'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['F1'] == 0.0


def test_f1_unchanged_with_true_positives():
    B_est = np.array([[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0]])
    metrics = MetricsDAG(B_est=B_est, B_true=B_TRUE).metrics
    assert metrics['precision'] == 0.5
    assert metrics['recall'] == 0.5
    assert metrics['F1'] == 0.5
