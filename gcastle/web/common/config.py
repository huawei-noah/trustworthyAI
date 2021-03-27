# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     config
   Description :
   Author :       hWX889027
   date：          2020/11/11
-------------------------------------------------
"""
import os
from pathlib import Path

from castle.datasets.simulation import DAG
from castle.algorithms.pc.pc import PC
from castle.algorithms.gradient.notears import NotearsLowRank
from castle.algorithms.gradient.notears import NotearsMLP
from castle.algorithms.gradient.notears import NotearsSob
from castle.algorithms.ttpm import TTPM
from castle.algorithms.gradient.notears.linear import Notears

PROJECT_DIR = str(Path(__file__).resolve().parents[2])
FILE_PATH = os.path.join(PROJECT_DIR, "data")

if os.path.exists(os.path.abspath(FILE_PATH)):
    MY_DIR = os.path.abspath(FILE_PATH)
else:
    MY_DIR = ''

INLINE_DATASETS = {
    'gmB_binary': os.path.join(MY_DIR, 'gmB_binary.csv'),
    'gmD_discrete': os.path.join(MY_DIR, 'gmD_discrete.csv'),
    'gmG8_Gaussian': os.path.join(MY_DIR, 'gmG8_Gaussian.csv')
}

INLINE_ALGORITHMS = {"PC": PC, "GAE": "", "GRAN": "",
                     "LOW_RANK": NotearsLowRank, "MLP": NotearsMLP,
                     "NOTEARS": Notears, "SOB": NotearsSob, "TTPM": TTPM}

GRAPH_TYPE = {"ER": DAG.ER, "SF": DAG.SF, "BP": DAG.BP, "full": DAG.full,
              "hierarchical": DAG.hierarchical, "LR": DAG.LR}

SEM_TYPE = {"IID_linear": ["gauss", "exp", "gumbel", "uniform", "logistic",
                           "poisson"],
            "IID_nonlinear": ["mlp", "mim", "gp", "gp-add", "quadratic"],
            "EVENT": []}
