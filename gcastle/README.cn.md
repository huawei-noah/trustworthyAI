<h1 align="center">
<img width="500" alt="gcastle_logo" src="https://user-images.githubusercontent.com/1130029/184056068-8eaa21e5-6241-4682-94dc-b6231d238e23.png">
</h1>


[English Version](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle/README.md)

1.0.4 版本(2025/03/07|[API文档](https://gcastle.readthedocs.io/en/latest/))  

[![PyPI - Version](https://img.shields.io/pypi/v/gcastle?logo=pypi&logoColor=FFE873)](https://pypi.org/project/gcastle/)
[![codecov](https://codecov.io/gh/shaido987/trustworthyAI/graph/badge.svg?token=TS0BG6CEI1)](https://codecov.io/gh/shaido987/trustworthyAI)
[![Python - Version](https://img.shields.io/badge/python-3.6+-green?logo=python)]()
[![arXiv](https://img.shields.io/badge/arXiv-2111.15155-b31b1b.svg)](https://arxiv.org/abs/2111.15155)

## 简介

`gCastle`是[华为诺亚方舟实验室](https://www.noahlab.com.hk/#/home)自研的因果结构学习工具链，主要的功能和愿景包括：

* 数据生成及处理: 包含各种模拟数据生成算子，数据读取算子，数据处理算子（如先验灌入，变量选择，CRAM）。
* 因果图构建: 提供了一个因果结构学习python算法库，包含了主流的因果学习算法以及最近兴起的基于梯度的因果结构学习算法。
* 因果评价: 提供了常用的因果结构学习性能评价指标，包括F1, SHD, FDR, TPR, FDR, NNZ等。


## 算法列表

| 算法 | 分类 | 说明 |状态 |
| :--: | :-- | :-- | :--: |
| [PC](https://arxiv.org/abs/math/0510436) | IID/Constraint-based | 一种基于独立性检验的经典因果发现算法 | v1.0.3 |
| [ANM](https://webdav.tuebingen.mpg.de/causality/NIPS2008-Hoyer.pdf) | IID/Function-based | 一种非线性的加性噪声因果模型 | v1.0.3 |
| [DirectLiNGAM](https://arxiv.org/abs/1101.2489) | IID/Function-based | 一种线性非高斯无环模型的直接学习方法 | v1.0.3 |
| [ICALiNGAM](https://dl.acm.org/doi/10.5555/1248547.1248619) | IID/Function-based | 一种线性非高斯无环模型的因果学习算法 | v1.0.3 |
| [GES](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf) | IID/Score-based | 一种经典的基于贪婪等价搜索的结构学习算法 | v1.0.3 |
| [PNL](https://arxiv.org/abs/1205.2599) | IID/Funtion-based | 一种基于后非线性假设的因果发现方法 | v1.0.3 |
| [NOTEARS](https://arxiv.org/abs/1803.01422) | IID/Gradient-based | 一种基于梯度、针对线性数据模型的因果结构学习算法 | v1.0.3 |
| [NOTEARS-MLP](https://arxiv.org/abs/1909.13189) | IID/Gradient-based | 一种深度可微分、基于神经网络建模的因果结构学习算法 | v1.0.3 |
| [NOTEARS-SOB](https://arxiv.org/abs/1909.13189) | IID/Gradient-based | 一种深度可微分、基于Sobolev空间建模的因果结构学习算法 | v1.0.3 |
| [NOTEARS-lOW-RANK](https://arxiv.org/abs/2006.05691) | IID/Gradient-based | 基于low rank假定、针对线性数据模型的因果结构学习算法 | v1.0.3 |
| [DAG-GNN](https://arxiv.org/abs/1904.10098) | IID/Gradient-based | 一种基于GNN的因果结构学习算法 | v1.0.3 |
| [GOLEM](https://arxiv.org/abs/2006.10201) | IID/Gradient-based | 一种基于NOTEARS、通过减少优化循环次数提升训练效率的因果结构学习算法 | v1.0.3 |
| [GraNDAG](https://arxiv.org/abs/1906.02226) | IID/Gradient-based | 一种深度可微分、针对非线性加性噪声数据模型的因果结构学习算法 | v1.0.3 |
| [MCSL](https://arxiv.org/abs/1910.08527) | IID/Gradient-based | 一种基于掩码梯度的因果结构学习算法 | v1.0.3 |
| [GAE](https://arxiv.org/abs/1911.07420) | IID/Gradient-based | 一种基于图自编码器的因果发现算法 | v1.0.3 |
| [RL](https://arxiv.org/abs/1906.04477) | IID/Gradient-based | 一种基于强化学习的因果发现算法 | v1.0.3 |
| [CORL](https://arxiv.org/abs/2105.06631) | IID/Gradient-based | 一种基于强化学习搜索因果序的因果发现方法 | v1.0.3 |
| [TTPM](https://arxiv.org/abs/2105.10884) | EventSequence/Function-based | 一种针对时空事件序列的基于时空Hawkes Process的因果结构学习算法 | v1.0.3 |
| [HPCI](https://arxiv.org/abs/2105.03092) | EventSequence/Hybrid | 一种针对时序事件序列的基于Hawkes Process和CI tests的因果结构学习算法 | 开发中 |


## 获取和安装

### 依赖
- python (>= 3.6, <=3.9)
- tqdm (>= 4.48.2)
- numpy (>= 1.19.1)
- pandas (>= 0.22.0)
- scipy (>= 1.7.3)
- scikit-learn (>= 0.21.1)
- matplotlib (>=2.1.2)
- networkx (>= 2.5)
- torch (>= 1.9.0)

### PIP安装
```bash
pip install gcastle==1.0.4rc1
```

## 算法使用指导
以PC算法为例: 
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
大家可访问 [examples](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle/example) 获取更多的示例. 


## 引用
如果您想在公开的研究成果中使用gCastle, 请考虑引用如下[论文](https://arxiv.org/abs/2111.15155):
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


## 合作和贡献

欢迎大家使用`gCastle`. 该项目尚处于起步阶段，欢迎各个经验等级的贡献者，近期我们将公布具体的代码贡献规范和要求。当前有任何疑问及建议，包括修改bug、贡献算法、完善文档等，请在社区提交issue，我们会及时回复交流。
