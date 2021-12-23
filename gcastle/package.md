# gCastle

## Introduction

`gCastle` is a causal structure learning toolchain developed by [Huawei Noah's Ark Lab](https://www.noahlab.com.hk/#/home). The package contains various functionality related to causal learning and evaluation, including:

* Data generation and processing: data simulation, data reading operators, and data pre-processing operators (such as prior injection and variable selection).
* Causal structure learning: causal structure learning methods, including both classic and recently developed methods, especially gradient-based ones that can handle large problems.
* Evaluation metrics: various commonly used metrics for causal structure learning, including F1, SHD, FDR, TPR, FDR, NNZ, etc.

## Algorithm List

| Algorithm | Category (based on data) | Description |
| :--: | :-- | :-- |
| [PC](https://arxiv.org/abs/math/0510436) | IID | A classic causal discovery algorithm based on conditional independence tests |
| [DirectLiNGAM](https://arxiv.org/abs/1101.2489) | IID | A direct learning algorithm for linear non-Gaussian acyclic model (LiNGAM) |
| [ICALiNGAM](https://dl.acm.org/doi/10.5555/1248547.1248619) | IID | An ICA-based learning algorithm for linear non-Gaussian acyclic model (LiNGAM) |
| [NOTEARS](https://arxiv.org/abs/1803.01422) | IID | A gradient-based algorithm for linear data models (typically with least-squares loss) |
| [NOTEARS-MLP](https://arxiv.org/abs/1909.13189) | IID | A gradient-based algorithm using neural network modeling for non-linear causal relationships |
| [NOTEARS-SOB](https://arxiv.org/abs/1909.13189) | IID | A gradient-based algorithm using Sobolev space modeling for non-linear causal relationships |
| [NOTEARS-lOW-RANK](https://arxiv.org/abs/2006.05691) | IID | Adapting NOTEARS for large problems with low-rank causal graphs |
| [GOLEM](https://arxiv.org/abs/2006.10201) | IID | A more efficient version of NOTEARS that can reduce number of optimization iterations |
| [GraNDAG](https://arxiv.org/abs/1906.02226) | IID | A gradient-based algorithm using neural network modeling for non-linear additive noise data  |
| [MCSL](https://arxiv.org/abs/1910.08527) | IID | A gradient-based algorithm for non-linear additive noise data by learning the binary adjacency matrix|
| [GAE](https://arxiv.org/abs/1911.07420) | IID | A gradient-based algorithm using graph autoencoder to model non-linear causal relationships|
| [RL](https://arxiv.org/abs/1906.04477) | IID | A RL-based algorithm that can work with flexible score functions (including non-smooth ones) |
| [CORL](https://arxiv.org/abs/2105.06631) | IID | A RL- and order-based algorithm that improves the efficiency and scalability of previous RL-based approach |
| ANM | IID | a causal discovery algorithm based on non-linear additive noise models |
| TTPM | EVENT SEQUENCE | A causal structure learning algorithm based on Topological Hawkes process for spatio-temporal event sequences |

## Installation

### Dependencies

gCastle requires:

* python (>= 3.6)
* tqdm (>= 4.48.2)
* numpy (>= 1.19.1)
* pandas (>= 0.22.0)
* scipy (>= 1.7.3)
* scikit-learn (>= 0.21.1)
* matplotlib (>=2.1.2)
* networkx (>= 2.5)
* torch (>= 1.9.0)

### PIP installation

```bash
pip install gcastle
```

## Usage Example (PC algorithm)

```python
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
GraphDAG(pc.causal_matrix, true_causal_matrix)

# calculate metrics
mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
print(mt.metrics)
```

You can visit [examples](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) to find more examples.

## Next Up & Contributing

You can find more news about [gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) on the github website.
