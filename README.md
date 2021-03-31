# Trustworthy AI

This repository aims to include trustworthy AI related projects from Huawei Noah's Ark Lab. 
Current projects include:

- Causal Structure Learning

- Causal Disentangled Representation Learning

- gCastle (or pyCastle, pCastle)

### Causal Structure Learning

- **Causal_Discovery_RL**:  codes, datasets, and training logs of the experimental results for the paper
 ['Causal discovery with reinforcement learning'](https://openreview.net/forum?id=S1g2skStPB), ICLR, 2020. (oral)
- **GAE_Causal_Structure_Learning**: an implementation for '[A graph autoencoder approach to causal structure learning'](https://arxiv.org/abs/1911.07420), NeurIPS Causal Machine Learning Workshop, 2019.
- **Datasets**: 
    - Synthetic datasets: codes for generating synthetic datasets used in the paper
    - Real datasets: a very challenging real dataset, to find causal structures based on 
    time series data. The true graph is obtained from expert knowledge. We welcome everyone to try this dataset and 
    report the result!
- We will also release the codes of other gradient based causal structure learning methods.

### Causal Disentangled Representation Learning

- **CausalVAE**:  codes and datasets of the experimental results for the paper
 ['CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models'](https://arxiv.org/pdf/2004.08697.pdf), CVPR, 2021. (accepted)
- **Datasets**: codes for generating synthetic datasets used in the paper.

### gCastle
- It is a causal structure learning toolchain, which contains various functionality related to causal learning and evaluation.
- Most of causal discovery algorithms in `gCastle` are gradient based, which is why we call it gCastle : **g**radient-based **Ca**usal **St**ructure **Le**arning pipeline.
