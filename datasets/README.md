# Datasets for Causal Structure Learning

## Synthetic datasets
We provide codes for generating synthetic datasets used in '[Causal Discovery with Reinforement Learning](../research/Causal%20Discovery%20with%20RL)'. Please see the [example notebook](examples_to_generate_synthetic_datasets.ipynb) for further details.

## Real datasets

### Telecom causal dataset
We release a very challenging [dataset](https://github.com/zhushy/causal-datasets/tree/master/Real_Dataset) from real telecommunication networks, to find causal structures based on time series data. 

### Data format
- **real_dataset_processed.csv**: each row counts the numbers of occurrences of the alarms (A_i,i=0,1,...,56) in 10 minutes. The rows are arranged in the time order, i.e., first 10 mins., second 10 mins., etc.
- **true_graph.csv**: the underlying causal relationships, according to expert experience.  `(i,j)=1` implies an edge `i->j`.

### PCIC competition datasets

The real datasets (id:10, 21, 22) used in [PCIC Causal Discovery Competition 2021 ](https://competition.huaweicloud.com/information/1000041487/introduction) have been made available online: [link](https://github.com/gcastle-hub/dataset).

The following table contains F1-scores of some competitive causal discovery algorithms on these datasets. 
We believe the THP algorithm (last line) is the current SOTA. You are welcome to contact us if you have another (public) algorithm that obtains good scores and we will add it to the table.
|  Algorithm  | 18V_55N_Wireless | 24V_439N_Microwave | 25V_474N_Microwave | paper                                                          |
|--------------| ---------------- | ------------------ | ------------------ | ------------------------------------------------------------------- |
| PC           | 0.4299           | 0.2270              | 0.1923             | [link](https://philarchive.org/archive/SPICPA-2)                            |
| DirectLiNGAM | 0.1609           | 0.0625             | 0.0761             | [link](https://www.jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf)      |
| ICALiNGAM    | 0.1915           | 0.0513             | 0.0442             | [link](https://www.jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf)      |
| GES          | 0.3393           | 0.3054             | 0.2008             | [link](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf) |
| NOTEARS      | 0.1957           | 0.1435             | 0.1441             | [link](https://arxiv.org/abs/1803.01422)                                    |
| ADM4         | 0.4074           | 0.2620              | 0.2518             | [link](https://proceedings.mlr.press/v31/zhou13a.html)                      |
| MLE_SGL      | 0.3739           | 0.3252             | 0.3050              | [link](https://arxiv.org/abs/1602.04511)                                    |
| PCMCI        | 0.4226           | 0.3268             | 0.2919             | [link](https://www.science.org/doi/10.1126/sciadv.aau4996)                  |
| THP          | 0.5765           | 0.3719             | 0.3387             | [link](https://arxiv.org/pdf/2105.10884.pdf)                                |

