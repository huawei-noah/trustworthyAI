# GraNDAG

## 1. 算法介绍

----

结构学习和因果推理在遗传学、生物学和经济学等不同科学领域有许多重要的应用。学习因果图形模型的典型动机是预测各种干预的影响效果。当给出干预数据时，因果图模型（CGM）可以最好地被估计，但干预通常是昂贵的或不可能获得的。作为替代方案，GraNDAG提出了一种仅仅依赖观测数据，并依靠不同的假设，使得因果图可以从分布中识别。

## 2. 算法原理

GraNDAG 提出一种基于打分的结构学习方法，用于支持非线性关系，同时利用连续优化范式。该方法基于 NOTEARS 方法，对神经网络的非循环性进行了新的表征。在原始的方法 NOTEARS 中，有向图被编码为加权邻接矩阵，该矩阵表示线性结构方程模型（SEM）中的系数，并使用可有效计算和易于微分的约束强制非循环性，从而允许使用数值求解器。这种连续方法改进了流行的方法，同时避免了基于启发式的贪婪算法的设计。

结构学习的连续约束方法的优点是比其他近似贪婪方法更全局（因为它根据分数的梯度和非周期性约束更新每个步骤的所有边），并允许用适当的非周期性约束来取代特定于任务的贪婪算法货架数值求解器。

详细的原理介绍请参考 [Gradient-Based Neural DAG Learning](https://arxiv.org/pdf/1906.02226.pdf) 。

### 2.1 学习策略

我们使用专业术语“神经网络路径”来表示神经网络中的计算路径。例如，在一个含有2层隐藏层的神经网络中，权值序列（$W_{h_1 i}^{(1)},W_{h_2 h_1}^{(2)},W_{k h_2}^{(3)}$ ）是一条从输入 **_i_** 到输出  **_k_** 的神经网络路径。如果至少有一个沿着该路径的权值是 0 ，我们称这条神经网络路径是**未激活的**。如果所有从输入 **_i_** 到输出  **_k_** 的神经网络路径都是未激活的，则称输出  **_k_** 不依赖于输入 **_i_** ，或者说输入 **_i_** 将永远不会到达输出  **_k_** ，即不存在因果关系。

[原文 3.1 节有更详细的介绍](https://arxiv.org/pdf/1906.02226.pdf)

### 2.2 数学原理

GraNDAG 在 NOTEARS 的基础上提出了一种新的非线性扩展的框架。该算法包含三个重要组成部分：**神经网络模型**、**连接矩阵**、**邻接矩阵**。

#### 2.2.1 神经网络模型

$$
\theta_{(j)} \triangleq W_{(j)}^{(L+1)}g(...g(W_{(j)}^{(2)}g(W_{(j)}^{(1)}X_{\_j}))...) \ \ \ \ \forall_j
$$

其中： **_g_** 是激活函数 leaky-relu 或 sigmoid 。

#### 2.2.2 连接矩阵

用以计算神经网络路径
$$
C_j \triangleq |W_j^{(L+1)}|...|W_j^{(2)}||W_j^{(1)}| \in \mathbb R^{m×d}
$$
​其中，|| 表示绝对值。

#### 2.2.3 邻接矩阵

定义可用于约束的邻接矩阵：
$$
(A_\phi)_{ij} \triangleq \begin{cases} \sum_{k=1}^m (C_{(j)})_{ki}, \ \ \ if \ \ j \ne i \\ 0, \ \ \ \ \  \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ otherwise \end{cases}
$$
约束的定义为：
$$
h(\phi) \triangleq Tr\ e^{A_\phi} - d = 0
$$
实际情况中，我们使用 $10^{-8}$ 代替 0 值。

#### 2.2.4 目标函数及优化

$$
\max \limits_{\phi} \mathcal L(\phi, \lambda_t, \mu_t) \triangleq \mathbb E_X \sim P_X \sum_{j=1}^d log\ p_j(X_j|X_{\pi_j^\phi};\phi_{(j)}) - \lambda_t h(\phi) - \frac{\mu_t}{2}h(\phi)^2
$$

其中：_t_ = 1, 2, 3, ... 表示第 _t_ 次迭代。

参数的优化规则如下：
$$
\lambda_{t+1} \gets \lambda_t + \mu_t h(\phi_t^*) \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \\
\mu_{t+1} \gets \begin{cases}   \eta \mu_t , \ \ \ if \ h(\phi_t^*) > \gamma h(\phi_{t-1}^*) \\  \mu_t, \ \ \ \ \ otherwise  \end{cases}
$$

## 3. 使用指导

调用算法有两种模式实现：API模式、工具模式。

### 3.1 使用示例

#### API模式

```python
import pandas as pd
from castle.algorithms import GraNDAG

x = pd.read_csv('x.csv').values # x.csv 是观测数据，行-表示样本，列-表示特征

gnd = GraNDAG(input_dim=x.shape[1])
gnd.learn(data=x)

print(gnd.causal_matrix) # 打印 学到的因果图（矩阵）
```
#### 工具模式

1. 在 .yaml文件中进行参数配置，配置示例如下：

   ```python
   dataset_params:
     x_file: None  # None or .npz or .csv    样本数据所在路径
     dag_file: None  # None or .npz or .csv  真实dag矩阵所在路径
   
   model_params:
     input_dim: 10 # number of variable, X.shape[1]
     hidden_num: 2 # number of hidden layers
     hidden_dim: 10  # number of dimension per hidden layer
     lr: 0.001 # learning rate
     iterations: 100000  # times of iteration
     batch_size: 64  # batch size of per training of NN
     model_name: 'NonLinGaussANM' # name of model, 'NonlinearGauss' or 'NonlinearGaussANM'
     nonlinear: 'leaky-relu' # name of Nonlinear activation function
     optimizer: 'rmsprop'  # Method of optimize
     h_threshold: !!float 1e-8   # constrained threshold
     gpu: False  # whether use gpu
   
     pns: False  # variable number or neighbors number used in the pns
     pns_thresh: 0.75
     num_neighbors: None
   
     normalize: False  # whether normalize data
     precision: False  # whether use Double precision
     random_seed: 42
     jac_thresh: True
     lambda_init: 0.0
     mu_init: 0.001
     omega_lambda: 0.0001
     omega_mu: 0.9
     stop_crit_win: 100
     edge_clamp_range: 0.0001
     norm_prod: 'paths'
     square_prod: False
   ```

2. 执行如下命令：

   ```shell
   python run.py -m gran_dag -c example/gran_dag/gran_dag.yaml
   ```

   \-m :  表示模型  -model

   \-c ：表示配置文件路径  -config

### 3.2 算法输出

GraNDAG算法会输出模型的实例，可通过访问 GraNDAG.causal_matrix 或 GraNDAG.model.adjacency 查看模型所学到的因果图矩阵。

