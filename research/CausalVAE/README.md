# CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models

Code for the paper [CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models](https://arxiv.org/abs/2004.08697.pdf) (CVPR, 2021), by Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, and Jun Wang.

If you find it useful, please consider to cite the paper.

This is a causal disentanglement image package. 

## Requirements
- python 3.7
- torch 1.4.1

## Process

### Generate synthetic data: 
- Generate flow data: python ./causal_data/flow.py
- Generate pendulum data: python ./causal_data/pendulum.py

### Train CausalVae:
- Train on flow data: python ./run_flow.py
- Train on pendulum data: python ./run_pendulum.py

### Intervention on synthetic data by CausalVae:
- Intervention on flow data: python ./inference_flow.py
- Intervention on pendulum data: python ./inference_pendulum.py
