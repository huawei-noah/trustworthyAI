# Trustworthy AI

This repository is a collection of trustworthy AI related works from Huawei Noah's Ark Lab.  

---

### gCastle

- A causal structure learning toolchain containing various functionalities related to causal learning and evaluation. A tech report describing the toolbox is available [here](https://arxiv.org/abs/2111.15155).
- The package offers a number of causal discovery algorithms, most of which are gradient-based, hence the name: **g**radient-based **Ca**usal **st**ructure **le**arning pipeline.

### Competition

- Information and baselines for causality-related competitions arranged by Noah's Ark Lab.
- Previous competitions were held at PCIC 2021, PCIC 2022, and NeurIPS 2023.

### Datasets

- Real-world datasets released by Huawei Noah's Ark Lab.
- Code for generating various synthetic datasets.

### Research 
 
- Research works related to causality. We will continuously add new methods here.
- Currently contains implementations of CausalVAE, GAE, and causal discovery with reinforcement learning.

---

### TabbyTSFM
 
- **Tabby**, a time series foundation model proposed by the Paris team of Huawei Noah's Ark Lab, supporting forecasting, classification, and anomaly detection. Its pretraining code is fully open-sourced here.
- **Data processing**: [`TabbyTSFM/src/tabby/data`](TabbyTSFM/src/tabby/data) holds the GIFT-Eval/BLAST shard reader, the KernelSynth Arrow reader, and the online CauKer V2 generator that samples series from structural causal models; shared input preprocessing is in [`TabbyTSFM/src/tabby/utils`](TabbyTSFM/src/tabby/utils).
- **Pretraining**: [`TabbyTSFM/recipes/pretrain`](TabbyTSFM/recipes/pretrain) contains the distributed mixed-data trainer and the `train_165k.sh` release entry point.
- **Tests and evaluation**: [`TabbyTSFM/tests`](TabbyTSFM/tests) covers the release contract, checkpoint loading across every accepted layout, and the GIFT-Eval script end to end; [`TabbyTSFM/benchmarks`](TabbyTSFM/benchmarks) holds the benchmark harnesses for GIFT-Eval and TIME (forecasting), UCR (classification), and TSB-AD (anomaly detection).
- **The weights** are published on Hugging Face at [`paris-noah/Tabby`](https://huggingface.co/paris-noah/Tabby)
 
