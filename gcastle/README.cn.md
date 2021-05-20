# gCastle

[English Version](./README.md)

1.0.0 版本发布

## 简介

`gCastle`是[华为诺亚方舟实验室](https://www.noahlab.com.hk/#/home)自研的因果结构学习工具链，主要的功能和愿景包括：

* 数据生成及处理: 包含各种模拟数据生成算子，数据读取算子，数据处理算子（如先验灌入，变量选择，CRAM）。
* 因果图构建: 提供了一个因果结构学习python算法库，包含了主流的因果学习算法以及最近兴起的基于梯度的因果结构学习算法。
* 因果评价: 提供了常用的因果结构学习性能评价指标，包括F1, SHD, FDR, TPR, FDR, NNZ等。
<!--* 因果可视化: 直观展示因果结构（特别是大规模因果结构图），包含整体展示，局部展示及导出功能。-->


## 算法列表

| 算法 | 数据分类 | 说明 |状态 |
| :--: | :-- | :-- | :-- |
| [PC](https://arxiv.org/abs/math/0510436) | IID | 一种基于独立性检验的经典因果发现算法 | v1.0.0 |
| [DirectLiNGAM](https://arxiv.org/abs/1101.2489) | IID | 一种线性非高斯无环模型的直接学习方法 | v1.0.0 |
| [ICALiNGAM](https://dl.acm.org/doi/10.5555/1248547.1248619) | IID | 一种线性非高斯无环模型的因果学习算法 | v1.0.0 |
| [NOTEARS](https://arxiv.org/abs/1803.01422) | IID | 一种基于梯度、针对线性数据模型的因果结构学习算法 | v1.0.0 |
| [NOTEARS-MLP](https://arxiv.org/abs/1909.13189) | IID| 一种深度可微分、基于神经网络建模的因果结构学习算法 | v1.0.0 |
| [NOTEARS-SOB](https://arxiv.org/abs/1909.13189) | IID | 一种深度可微分、基于Sobolev空间建模的因果结构学习算法 | v1.0.0 |
| [NOTEARS-lOW-RANK](https://arxiv.org/abs/2006.05691) | IID | 基于low rank假定、针对线性数据模型的因果结构学习算法 | v1.0.0 |
| [GOLEM](https://arxiv.org/abs/2006.10201) | IID | 一种基于NOTEARS、通过减少优化循环次数提升训练效率的因果结构学习算法 | v1.0.0 |
| [GraN_DAG](https://arxiv.org/abs/1906.02226) | IID | 一种深度可微分、针对非线性加性噪声数据模型的因果结构学习算法 | v1.0.0 |
| [MCSL](https://arxiv.org/abs/1910.08527) | IID | 一种基于掩码梯度的因果结构学习算法 | v1.0.0 |
| [GAE](https://arxiv.org/abs/1911.07420) | IID | 一种基于图自编码器的因果发现算法 | v1.0.0 |
| [RL](https://arxiv.org/abs/1906.04477) | IID | 一种基于强化学习的因果发现算法 | v1.0.0 |
| CORL1 | IID | 一种基于强化学习搜索因果序的因果发现方法 | v1.0.0 |
| CORL2 | IID | 一种基于强化学习搜索因果序的因果发现方法 | v1.0.0 |
| TTPM | EVENT SEQUENCE | 一种针对时空事件序列的基于时空Hawkes Process的因果结构学习算法 | 开发中 |
| [HPCI](https://arxiv.org/abs/2105.03092) | EVENT SEQUENCE | 一种针对时序事件序列的基于Hawkes Process和CI tests的因果结构学习算法 | 开发中 |
| PCTS | TS | 一种针对时间序列数据的基于CI tests的因果结构学习算法（PC算法的时序版本） | 开发中 |


## 获取和安装

### 依赖
- python (>= 3.6)
- tqdm (>= 4.48.2)
- numpy (>= 1.19.2)
- pandas (>= 0.22.0)
- scipy (>= 1.4.1)
- scikit-learn (>= 0.21.1)
- matplotlib (>=2.1.2)
- python-igraph (>= 0.8.2)
- loguru (>= 0.5.3)
- networkx (>= 2.5)
- torch (>= 1.4.0)
- tensorflow (>= 1.15.0)

### PIP安装
```bash
pip install gcastle
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
大家可访问 [examples](./castle/example) 获取更多的示例. 

## 合作和贡献
这是`gCastle`的第一个发布版本，我们将持续对相关代码及文档进行完善和优化。下面是我们针对下一个版本的完善和优化计划（备注：下一个版本预计在2021年6月发布）：
* 更完善的文档说明：包括每个算法基本原理的介绍，使用gCastle工具快速设计因果结构学习实验的简明手册，更易读的API说明。
* 算法库拓展：新增`GES`,`HPCI`,`TTPM`等因果结构学习算法，并提供简易的可配置脚本辅助进行相应算法的调用和运行。
* 真实场景数据集：将陆续公开一批来源于真实AIOPS场景的时间序列和事件序列数据集，其中真实的因果图标注来源于业务专家经验。

欢迎大家使用`gCastle`. 该项目尚处于起步阶段，欢迎各个经验等级的贡献者。有任何疑问及建议，包括修改bug、贡献算法、完善文档等，请在社区提交issue，我们会及时回复交流。
