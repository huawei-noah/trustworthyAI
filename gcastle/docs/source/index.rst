.. gCastle documentation master file, created by
   sphinx-quickstart on Tue Feb 25 09:17:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gCastle's documentation!
===============================
.. image:: https://img.shields.io/pypi/v/gcastle?logo=pypi&logoColor=FFE873
   :target: https://pypi.org/project/gcastle/
.. image:: https://codecov.io/gh/shaido987/trustworthyAI/graph/badge.svg?token=TS0BG6CEI1
   :target: https://codecov.io/gh/shaido987/trustworthyAI
.. image:: https://img.shields.io/badge/python-3.6+-green?logo=python
.. image:: https://img.shields.io/badge/arXiv-2111.15155-b31b1b.svg
   :target: https://arxiv.org/abs/2111.15155

Introduction
------------

gCastle is a causal structure learning toolchain developed by Huawei Noah's Ark Lab. The package contains various functionalities related to causal learning and evaluation, including:

- Data generation and processing: data simulation, data reading operators, and data pre-processing operators（such as prior injection and variable selection).
- Causal structure learning: causal structure learning methods, including classic and recently developed methods, especially gradient-based ones that can handle large problems.
- Evaluation metrics: various commonly used metrics for causal structure learning, including F1, SHD, FDR, TPR, FDR, NNZ, etc.


.. toctree::
   :maxdepth: 2

   getting_started
   castle/castle


Algorithms
----------

+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| Algorithm                                                                    | Category                     | Description                                                                                                   | Status             |
+==============================================================================+==============================+===============================================================================================================+====================+
| `PC <https://arxiv.org/abs/math/0510436>`_                                   | IID/Constraint-based         | A classic causal discovery algorithm based on conditional independence tests                                  | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `ANM <https://webdav.tuebingen.mpg.de/causality/NIPS2008-Hoyer.pdf>`_        | IID/Function-based           | Nonlinear causal discovery with additive noise models                                                         | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `DirectLiNGAM <https://arxiv.org/abs/1101.2489>`_                            | IID/Function-based           | A direct learning algorithm for linear non-Gaussian acyclic model (LiNGAM>)                                   | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `ICALiNGAM <https://dl.acm.org/doi/10.5555/1248547.1248619>`_                | IID/Function-based           | An ICA-based learning algorithm for linear non-Gaussian acyclic model (LiNGAM>)                               | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `GES <https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf>`_ | IID/Score-based              | A classical Greedy Equivalence Search algorithm                                                               | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `PNL <https://arxiv.org/abs/1205.2599>`_                                     | IID/Function-based           | Causal discovery based on the post-nonlinear causal assumption                                                | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `NOTEARS <https://arxiv.org/abs/1803.01422>`_                                | IID/Gradient-based           | A gradient-based algorithm for linear data models (typically with least-squares loss)                         | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `NOTEARS-MLP <https://arxiv.org/abs/1909.13189>`_                            | IID/Gradient-based           | A gradient-based algorithm using neural network modeling for non-linear causal relationships                  | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `NOTEARS-SOB <https://arxiv.org/abs/1909.13189>`_                            | IID/Gradient-based           | A gradient-based algorithm using Sobolev space modeling for non-linear causal relationships                   | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `NOTEARS-lOW-RANK <https://arxiv.org/abs/2006.05691>`_                       | IID/Gradient-based           | Adapting NOTEARS for large problems with low-rank causal graphs                                               | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `DAG-GNN <https://arxiv.org/abs/1904.10098>`_                                | IID/Gradient-based           | DAG Structure Learning with Graph Neural Networks                                                             | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `GOLEM <https://arxiv.org/abs/2006.10201>`_                                  | IID/Gradient-based           | A more efficient version of NOTEARS that can reduce the number of optimization iterations                     | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `GraNDAG <https://arxiv.org/abs/1906.02226>`_                                | IID/Gradient-based           | A gradient-based algorithm using neural network modeling for non-linear additive noise data                   | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `MCSL <https://arxiv.org/abs/1910.08527>`_                                   | IID/Gradient-based           | A gradient-based algorithm for non-linear additive noise data by learning the binary adjacency matrix         | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `GAE <https://arxiv.org/abs/1911.07420>`_                                    | IID/Gradient-based           | A gradient-based algorithm using graph autoencoder to model non-linear causal relationships                   | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `RL <https://arxiv.org/abs/1906.04477>`_                                     | IID/Gradient-based           | A RL-based algorithm that can work with flexible score functions (including non-smooth ones)                  | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `CORL <https://arxiv.org/abs/2105.06631>`_                                   | IID/Gradient-based           | A RL- and order-based algorithm that improves the efficiency and scalability of previous RL-based approach    | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `TTPM <https://arxiv.org/abs/2105.10884>`_                                   | EventSequence/Function-based | A causal structure learning algorithm based on Topological Hawkes process for spatio-temporal event sequences | v1.0.3             |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+
| `HPCI <https://arxiv.org/abs/2105.03092>`_                                   | EventSequence/Hybrid         | A causal structure learning algorithm based on Hawkes process and CI tests for event sequences                | under development. |
+------------------------------------------------------------------------------+------------------------------+---------------------------------------------------------------------------------------------------------------+--------------------+


Citation
--------
If you find gCastle useful in your research, please consider citing the following paper:

.. code-block:: bibtex

   @misc{zhang2021gcastle,
      title={gCastle: A Python Toolbox for Causal Discovery},
      author={Keli Zhang and Shengyu Zhu and Marcus Kalander and Ignavier Ng and Junjian Ye and Zhitang Chen and Lujia Pan},
      year={2021},
      eprint={2111.15155},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
   }


Next Up & Contributing
----------------------
We'll be continuously complementing and optimizing the code and documentation. We welcome new contributors of all experience levels, the specifications about how to contribute code will be coming out soon. If you have any questions or suggestions (such as contributing new algorithms, optimizing code, or improving documentation), please submit an issue on our `GitHub <https://github.com/huawei-noah/trustworthyAI>`_. We will get back to you as soon as possible.
