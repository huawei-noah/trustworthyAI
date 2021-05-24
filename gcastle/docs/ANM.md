# ANM_Nonlinear

## 1. 算法介绍

因果关系是科学的基础，因为它们能够预测行动的后果。虽然受控随机实验是确定因果关系的主要工具，但在许多情况下，此类实验要么过于昂贵，要么技术上不可能。因此，从非受控数据推断因果关系的因果发现方法的发展构成了当前重要的研究课题。如果观察到的数据是连续的，则通常应用基于线性因果模型（又名结构方程模型）的方法。这不一定是因为真正的因果关系被认为是线性的，而是因为线性模型能被很好地理解和易于使用。标准的方法是从数据中估计有向无环图的所谓马尔可夫等价类（所有图都表示相同的条件独立性）。对于连续变量，独立性检验通常假定具有加性高斯噪声的线性模型。然而，最近的研究表明，对于线性模型，数据中的非高斯性实际上可以帮助区分因果方向，并允许人们在有利条件下唯一地识别生成图。

在 [Nonlinear causal discovery with additive noise models](https://is.tuebingen.mpg.de/fileadmin/user_upload/files/publications/NIPS2008-Hoyer-neu_5406[0].pdf) 一文中，作者证明了非线性可以发挥与非高斯性非常相似的作用，即当因果关系是非线性的时，它通常有助于打破观测变量之间的对称性，并允许识别因果方向。对于具有加性噪声的非线性模型，几乎任何非线性（可逆或不可逆）通常都会产生可识别的模型。

## 2. 算法原理

### 2.1 模型假设

我们假设被观测的数据是由如下方式所生成，即：有向无环图 $G$ 中的节点 $i$ 所关联的每一个观测变量 $x_i$ 的值是由 $G$ 中它的父节点的函数加上独立加性噪声 $n_i$ 所获得。即：
$$
x_i := f_i(\mathrm{Xpa(i)}) + n_i
$$
其中：

- $f_i$ 是任意函数（对每一个节点 $i$ 可能不同）
- $\mathrm{Xpa}(i)$ 是 DAG 中节点 $i$ 的所有父节点 $j$ 的关联向量的集合
- $n_i$ 是噪声变量，服从联合独立的任意概率密度函数。

我们的目标是，给定数据向量，尽可能多的推断生成机制，特别的，我们试图推断生成图 $G$ 。

### 2.2 搜索策略

从考虑两个观测变量 $x$ 和 $y$ 的案例开始，对每一对变量  $x$ 和 $y$ 进行如下步骤：

- 第一步：检验 $x$ 与 $y$ 的统计独立性。
- 第二步：如果独立，则 $x$ 和 $y$ 不存在因果关系；如果不独立，进一步使用模型 $y := f(x) + n$ 对数据进行拟合，简单的做法是对 $x$ 做非线性回归，得到估计值 $\hat f$ ，并计算对应的残差 $ \hat n = y - \hat f(x)$ ，检验 $\hat n$ 与变量 $x$ 是否独立。如果独立，则接受该模型，认为存在因果关系 $x  \to  y$；如果不独立，则拒绝该模型，认为不存在因果关系 $x \to y$。
- 第三步：使用反向模型 $x := g(y) + n$ 来拟合数据，并用第二步的方式检验是否存在因果关系 $y \to x$。

## 3. 使用指导

调用算法有两种模式实现：API模式、工具模式。

### 3.1 使用示例

使用模拟数据的示例 [demo]() 。

#### API 模式

```python
import pandas as pd
from castle.algorithms import ANM_Nonlinear

x = pd.read_csv('x.csv').values # x.csv 是观测数据，行-表示样本，列-表示特征

anm = ANM_Nonlinear(alpha=0.05)
anm.learn(data=x)

print(anm.causal_matrix) # 打印 学到的因果图（矩阵）
```

#### 工具模式

1. 在 .yaml文件中进行参数配置，配置示例如下：

   ```yaml
   dataset_params:
     x_file: None  # None or .npz or .csv    样本数据所在路径
     dag_file: None  # None or .npz or .csv  真实dag矩阵所在路径
   
   model_params:
     alpha: 0.05 
     gpr_alpha: !!float 1e-10
   ```

2. 执行如下命令：

   ```shell
   python run.py -m anm_nonlinear -c example/anm/anm_nonlinear.yaml
   ```

### 3.2 算法输出

算法会输出模型的实例，可通过访问 `ANM_Nonlinear.causal_matrix` 查看模型所推断的因果关系矩阵。

