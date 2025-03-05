.. gCastle documentation master file, created by
   sphinx-quickstart on Tue Feb 25 09:17:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting started
===============================


Installation
------------

Dependencies
-------------
gCastle requires:

- python (>= 3.6, <=3.9)
- tqdm (>= 4.48.2)
- numpy (>= 1.19.1)
- pandas (>= 0.22.0)
- scipy (>= 1.7.3)
- scikit-learn (>= 0.21.1)
- matplotlib (>=2.1.2)
- networkx (>= 2.5)
- torch (>= 1.9.0)


Pip installation
----------------

.. code-block:: bash

   pip install gcastle==1.0.4rc1


Usage Example (PC algorithm)
----------------------------

.. code-block:: python

   from castle.common import GraphDAG
   from castle.metrics import MetricsDAG
   from castle.datasets import IIDSimulation, DAG
   from castle.algorithms import PC

   # data simulation, simulate true causal dag and train_data.
   weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=10,
                                         weight_range=(0.5, 2.0), seed=1)
   dataset = IIDSimulation(W=weighted_random_dag, n=2000, method='linear',
                           sem_type='gauss')
   true_causal_matrix, X = dataset.B, dataset.X

   # structure learning
   pc = PC()
   pc.learn(X)

   # plot predict_dag and true_dag
   GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

   # calculate metrics
   mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
   print(mt.metrics)
You can visit examples to find more examples.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
