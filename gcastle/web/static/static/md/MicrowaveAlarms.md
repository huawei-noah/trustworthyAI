# 微波告警数据集
## 训练数据集
按时间`(10 min时间窗)`及空间`(设备拓扑)`进行告警分组，且只选择出现的告警种类`不小于6`的告警分组：
* [数据集A](../data/alarms/sample_train_a.csv)：使用告警在分组内的出现次数作为度量值
* [数据集B](../data/alarms/sample_train_b.csv)：使用告警在分组内是否出现（0-1）作为度量值
* [数据集C](../data/alarms/sample_train_c.csv)：使用告警在分组内的最大持续时间作为度量值
* [数据集D](../data/alarms/sample_train_d.csv)：使用告警在分组内的最小发生时刻作为度量值

仅按时间`(10 min时间窗)`进行告警分组`，不考虑设备之间的连接拓扑：
* [数据集E](../data/alarms/sample_train_e.csv)：使用告警在分组内的出现次数作为度量值
* [数据集F](../data/alarms/sample_train_f.csv)：使用告警在分组内是否出现（0-1）作为度量值
* [数据集G](../data/alarms/sample_train_g.csv)：使用告警在分组内的最大持续时间作为度量值
* [数据集H](../data/alarms/sample_train_h.csv)：使用告警在分组内的最小发生时刻作为度量值

**说明:**  
* 数据集A~H均以`shape = (num_samples, num_variables)`的DataFrame形式（pickle或转存的csv格式）提供，数据集A~D的`shape =(22589,55)`，数据集E~H 的`shape=(864,57)`.
* 在生成数据集A~H过程中均去除了无训练样本的告警类型，所以两类数据集的的所包含的变量（告警类型）数量会不一致，在对各数据集训练生成 的因果图进行评测时只考虑在当前
  训练集中出现的告警类型.

**参考：**  
* 由原始告警数据及拓扑数据构建训练数据集A~H的相关处理代码： [alarm_rca/notebooks/make_dataset.ipynb](../notebooks/make_dataset.ipynb)或[alarm_rca/notebooks/make_dataset.py](../notebooks/make_dataset.py)
* 数据集A~H及标注数据（邻接矩阵形式）的存放目录：[alarm_rca/data/alarms/](../data/alarms/)
* 原始告警及拓扑下载地址：[raw_alarm_and_topology](https://onebox.huawei.com/p/2a27269ef77ceebcc10422e61bd857f7)

## 实验

**评价指标(metrics):**  
* `Precison = TP/(TP+FP`     
* `Recall = TP/(TP+FN)`   


* [ ] TP表示因果图本身存在的边,且学习到的结果中也存在边的数目
* [ ] FP表示因果图本身不存在的边，而学习到的结果存在的边的数目
* [ ] FN表示因果图本身存在的边，而学习到的结果不存在的边的数目
* [ ] 这里的边均指有向边

**结果:**  
`注：表中元组为（precision，recall）`

| 算法                                 | 数据集A         | 数据集B         | 数据集C         | 数据集E          | 数据集F       | 数据集G         | 原始数据集      | 总耗时                 |
| ------------------------------------ | --------------- | --------------- | --------------- | ---------------- | ------------- | --------------- | --------------- | ---------------------- |
| PC-fisherz                           | (0.265,0.078)   | (0.279,0.101)   | `(0.295,0.139)` | (0.309,0.101)    | (0.160,0.013) | (0.22,0.045)    | -               | 6个数据集，共耗时2501s |
| PC-gsquare                           | -               | `(0.249,0.179)` | -               | -                | (0.167,0.013) | -               | -               | -                      |
| NOTEARS                              | `(0.362,0.030)` | (0.667,0.01)    | (0.268,0.020)   | (0.217,0.108)    | (0.192,0.008) | `(0.260,0.106)` | -               | 6个数据集，共耗时2230s |
| TSDN                                 | (0.216,0.230)   | (0.211,0.231)   | (0.216,0.211)   | `(0.191,0.376) ` | (0.179,0.101) | `(0.208,0.358)` | -               | 6个数据集，共耗时28s   |
| Log-Linear                           | -               | -               | -               | -                | -             | -               | `(0.247,0.471)` | -                      |
| Scored-based Method + Hawkes Process | -               | -               | -               | -                | -             | -               | `(0.325,0.22) ` | -                      |

**说明:**  
* 上述实验使用的的标注数据（邻接矩阵形式）为[true_adjacency_matrix_62dim.csv](../data/alarms/true_adjacency_matrix_62dim.csv)，该标注数据通过专家手工标注完成，但由于涉及边数目较多（62*61 /2 = 1891 edges），且专家对非直接因果关系的标注不敏感，标注数据需持续进行优化，目前也在通过产品文档对标注辅助进行调整和确认.
* 当前告警数据主要来源为型号为OptiX RTN 950（A）的微波设备，其对应的产品手册：[RTN 950 V100R008C00 维护指南 02.pdf](../data/alarms/RTN950V100R008C00_manual.pdf)
* 原始的TSDN算法包含两个核心步骤:
    1.  `通过告警发生频率分布的的相似性来确定告警之间是否存在因果关系`
    2.  `通过告警间发生时序来确定其因果方向`
* 原始的TSDN算法实现代码：[tsdn.py](./src/algorithms/tsdn.py)
* 上述实验的测试代码：[do_experiment.ipynb](./experiment/do_experiment.ipynb)或[do_experiment.py](./experiment/do_experiment.py) 
* 产品手册:https://support.huawei.com/carrier/navi?coltype=product#col=product&path=PBI1-7851894/PBI1-23709925/PBI1-7275894