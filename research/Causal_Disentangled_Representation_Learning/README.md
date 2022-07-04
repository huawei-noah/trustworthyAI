# CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models

Code for the paper [CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models](https://arxiv.org/abs/2004.08697.pdf) (CVPR, 2021), by Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, and Jun Wang.

If you find it useful, please consider to cite the paper.

This is a causal disentanglement image package. 

# Requirements
- python 3.7
- torch 1.4.1

# Process
## Generate synthetic data: 
- generate flow data: python ./causal_data/flow.py
- generate pendulum data: python ./causal_data/pendulum.py

## Train CausalVae:
- train on flow data: python ./run_flow.py
- train on pendulum data: python ./run_pendulum.py

## Intervention on synthetic data by CausalVae:
- intervention on flow data: python ./inference_flow.py
- intervention on pendulum data: python ./inference_pendulum.py
