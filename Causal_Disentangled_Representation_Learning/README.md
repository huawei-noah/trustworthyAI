# CausalVAE: Disentangled Representation Learningvia Neural Structural Causal Models.

Codes for the paper ['CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models'] (https://arxiv.org/pdf/2004.08697.pdf) (CVPR, 2021), by Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, Jun Wang.

If you find it useful, please consider to cite the paper.

This is a causal disentanglement image package. 

# Requirement
- python 3.7
- torch 1.4.1

# Process
## generate synthetic: 
- generate flow data: python ./causal_data/flow.py
- generate pendulum data: python ./causal_data/pendulum.py

## Train CausalVae:
- train on flow data: python ./run_pendulum.py
- train on pendulum data: python ./run_flow.py

## Intervention on synthetic data by CausalVae:
- train on flow data: python ./inference_pendulum.py
- train on pendulum data: python ./inference_flow.py
