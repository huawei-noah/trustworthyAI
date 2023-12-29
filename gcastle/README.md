<h1 align="center">
<img width="500" alt="gcastle_logo" src="https://user-images.githubusercontent.com/1130029/184056068-8eaa21e5-6241-4682-94dc-b6231d238e23.png">
</h1>

[中文版本](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle/README.cn.md)

Version 1.0.4.rc1 (2023/12/29)  

[![PyPI - Version](https://img.shields.io/pypi/v/gcastle)](https://pypi.org/project/gcastle/)
[![codecov](https://codecov.io/gh/shaido987/trustworthyAI/graph/badge.svg?token=TS0BG6CEI1)](https://codecov.io/gh/shaido987/trustworthyAI)
[![Python - Version](https://img.shields.io/badge/python-3.6+-green)]()
[![arXiv](https://img.shields.io/badge/arXiv-2111.15155-b31b1b.svg)](https://arxiv.org/abs/2111.15155)


## Introduction

gCastle is a causal structure learning toolchain developed by [Huawei Noah's Ark Lab](https://www.noahlab.com.hk/#/home). The package contains various functionalities related to causal learning and evaluation, including: 
* Data generation and processing: data simulation, data reading operators, and data pre-processing operators（such as prior injection and variable selection).
* Causal structure learning: causal structure learning methods, including classic and recently developed methods, especially gradient-based ones that can handle large problems.
* Evaluation metrics: various commonly used metrics for causal structure learning, including F1, SHD, FDR, TPR, FDR, NNZ, etc.

## Algorithm List

| Algorithm | Category | Description | Status |
| :--: | :-- | :-- | :--: |
| [PC](https://arxiv.org/abs/math/0510436) | IID/Constraint-based | A classic causal discovery algorithm based on conditional independence tests | v1.0.3 |
| [ANM](https://webdav.tuebingen.mpg.de/causality/NIPS2008-Hoyer.pdf) | IID/Function-based | Nonlinear causal discovery with additive noise models | v1.0.3 |
| [DirectLiNGAM](https://arxiv.org/abs/1101.2489) | IID/Function-based | A direct learning algorithm for linear non-Gaussian acyclic model (LiNGAM) | v1.0.3 |
| [ICALiNGAM](https://dl.acm.org/doi/10.5555/1248547.1248619) | IID/Function-based | An ICA-based learning algorithm for linear non-Gaussian acyclic model (LiNGAM) | v1.0.3 |
| [GES](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf) | IID/Score-based | A classical Greedy Equivalence Search algorithm | v1.0.3 |
| [PNL](https://arxiv.org/abs/1205.2599) | IID/Function-based | Causal discovery based on the post-nonlinear causal assumption | v1.0.3 |
| [NOTEARS](https://arxiv.org/abs/1803.01422) | IID/Gradient-based | A gradient-based algorithm for linear data models (typically with least-squares loss) | v1.0.3 |
| [NOTEARS-MLP](https://arxiv.org/abs/1909.13189) | IID/Gradient-based | A gradient-based algorithm using neural network modeling for non-linear causal relationships | v1.0.3 |
| [NOTEARS-SOB](https://arxiv.org/abs/1909.13189) | IID/Gradient-based | A gradient-based algorithm using Sobolev space modeling for non-linear causal relationships | v1.0.3 |
| [NOTEARS-lOW-RANK](https://arxiv.org/abs/2006.05691) | IID/Gradient-based | Adapting NOTEARS for large problems with low-rank causal graphs | v1.0.3 |
| [DAG-GNN](https://arxiv.org/abs/1904.10098) | IID/Gradient-based | DAG Structure Learning with Graph Neural Networks | v1.0.3 |
| [GOLEM](https://arxiv.org/abs/2006.10201) | IID/Gradient-based | A more efficient version of NOTEARS that can reduce the number of optimization iterations | v1.0.3 |
| [GraNDAG](https://arxiv.org/abs/1906.02226) | IID/Gradient-based | A gradient-based algorithm using neural network modeling for non-linear additive noise data  | v1.0.3 |
| [MCSL](https://arxiv.org/abs/1910.08527) | IID/Gradient-based | A gradient-based algorithm for non-linear additive noise data by learning the binary adjacency matrix| v1.0.3 |
| [GAE](https://arxiv.org/abs/1911.07420) | IID/Gradient-based | A gradient-based algorithm using graph autoencoder to model non-linear causal relationships| v1.0.3 |
| [RL](https://arxiv.org/abs/1906.04477) | IID/Gradient-based | A RL-based algorithm that can work with flexible score functions (including non-smooth ones) | v1.0.3 |
| [CORL](https://arxiv.org/abs/2105.06631) | IID/Gradient-based | A RL- and order-based algorithm that improves the efficiency and scalability of previous RL-based approach | v1.0.3 |
| [TTPM](https://arxiv.org/abs/2105.10884) | EventSequence/Function-based | A causal structure learning algorithm based on Topological Hawkes process for spatio-temporal event sequences |v1.0.3 |
| [HPCI](https://arxiv.org/abs/2105.03092) | EventSequence/Hybrid | A causal structure learning algorithm based on Hawkes process and CI tests for event sequences | under development. |



## Installation

### Dependencies
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


### PIP installation
```bash
pip install gcastle==1.0.3
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
GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

# calculate metrics
mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
print(mt.metrics)
```
You can visit [examples](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle/example) to find more examples.


## Citation
If you find gCastle useful in your research, please consider citing the following [paper](https://arxiv.org/abs/2111.15155):
```
@misc{zhang2021gcastle,
  title={gCastle: A Python Toolbox for Causal Discovery}, 
  author={Keli Zhang and Shengyu Zhu and Marcus Kalander and Ignavier Ng and Junjian Ye and Zhitang Chen and Lujia Pan},
  year={2021},
  eprint={2111.15155},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Next Up & Contributing
This is the first released version of gCastle, we'll be continuously complementing and optimizing the code and documentation. We welcome new contributors of all experience levels, the specifications about how to contribute code will be coming out soon. If you have any questions or suggestions (such as contributing new algorithms, optimizing code, or improving documentation), please submit an issue here. We will get back to you as soon as possible.
