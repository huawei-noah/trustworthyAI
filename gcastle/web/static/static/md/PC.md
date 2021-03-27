# PC算法

```由Peter和Clark提出并以他们名字命名的贝叶斯网络结构学习方法```

## 基本思想：

* 变量之间的图结构的学习（无向图学习）

```首先创建完全无向图，然后基于d-分离以及独立性或条件独立性假设检验等统计方法给出的变量之间的独立性，砍掉相应的边，从而获得变量间的无向图```

* 图结构中方向的推断（有向边学习）

```依赖于V-结构(V-Structure)等局部结构特性确定部分边的方向```
