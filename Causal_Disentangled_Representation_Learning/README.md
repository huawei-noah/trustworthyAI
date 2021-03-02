This is a causal image disentanglement image package. 

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