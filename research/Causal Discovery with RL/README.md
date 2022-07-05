# Causal Discovery with Reinforcement Learning

Code for the paper [Causal Discovery with Reinforcement Learning](https://openreview.net/forum?id=S1g2skStPB) (ICLR 2020, Oral), by Shengyu Zhu, Ignavier Ng, and Zhitang Chen.

If you find it useful, please consider citing:

```bibtex
@inproceedings{Zhu2020Causal,
	title={Causal discovery with reinforcement learning},
	author={Zhu, Shengyu and Ng, Ignavier and Chen, Zhitang},
	booktitle={International Conference on Learning Representations},
	year={2020}
}
```
## TL;DR

We apply reinforcement learning to score-based causal discovery, as outlined below, and achieve promising results on both synthetic and real datasets.

![](fig0.png)

## Setup

You may check the required Python packages in `requirements.txt`. After installing the `rpy2` package, the previous codes would automatically install the R packages `CAM` and `mboost` used in our experiments. However, `CAM` is currently unavailable at `cran` (see the [notice](https://CRAN.R-project.org/package=CAM)), and here is an approach to install `CAM` from a local source.

* Go to [the notice address](https://CRAN.R-project.org/package=CAM) and download the archived version `CAM_1.0.tar.gz`.
* Put `CAM_1.0.tar.gz` in the same folder and then run the [setup_CAM.py](setup_CAM.py) which will install the required dependencies.
* If there are additional dependencies, then simply add them in [setup_CAM.py](setup_CAM.py).

We have verified this setup in a minimal conda environment with only `rpy2` package. Please file an issue if there are any further questions.

## Experiments

### Graph Adjacency Matrix
We find there are two uses of the meaning of the `(i,j)th` entry in the adjacency matrix:
* **(1)** whether there is an edge from `node i` to `node j`;
* **(2)** whether there is an edge from `node j` to `node i`. 

In our experience, we find that most people adopt the use of **(1)**. In fact, **(2)** was what we used in the first working version of the codes and we also find it somewhat convenient in the implementation. In the current codes, we fix the following:

* The input true graph should follow **(1)** (if not, please add command --transpose).
* We transpose the graph adjacency matrix and the RL approach in our codes follows the graph format **(2)**.
* All the outputs (e.g., saved graphs and plots) are transposed to be in the form of **(1)**.

### Experiments in the paper

* Our datasets are generated with different random seeds.
* We do not set a particular seed for running the experiments, so the result may 
be slightly different from the released training logs. We rerun the codes on several
(but not all) datasets to verify the reported results. If you find a large deviation
from the released training log, please file an issue to let us know.
* We open-source three synthetic datasets that were used in our experiments. The Sachs dataset 
belongs to the authors, so please download the dataset by yourself 
(we do release the training_logs with this dataset).
* Codes for synthetic dataset generation are available [here](../../datasets). The used datasets and training logs in the paper can be found [here](https://github.com/zhushy/causal-datasets/tree/master/Causal_Discovery_RL).
Jupyter notebooks are also provided to illustrate the experiment results.
* You may need to install the rpy2 package when CAM pruning is used. Otherwise, simply comment out the CAM pruning import codes.


### Detailed commands for running the experiment:
```
# exp1: RL-BIC2, assuming the equal noise variances
cd src
python main.py  --max_length 12 \
                --data_size 5000 \
                --score_type BIC \
                --reg_type LR \
                --read_data  \
                --transpose \
                --data_path input_data_path \
                --lambda_flag_default \
                --nb_epoch 20000 \
                --input_dimension 64 \
                --lambda_iter_num 1000
```
```
# exp1: RL-BIC, assuming different noise variances
python main.py  --max_length 12 \
                --data_size 5000 \
                --score_type BIC_different_var \
                --reg_type LR \
                --read_data  \
                --transpose \
                --data_path input_data_path \
                --lambda_flag_default \
                --nb_epoch 20000 \
                --input_dimension 64 \
                --lambda_iter_num 1000
```
```            
# exp1: 30 nodes
python main.py  --max_length 30 \
                --data_size 5000 \
                --score_type BIC \
                --reg_type LR \
                --use_bias \
                --bias_initial_value -10 \
                --read_data  \
                --batch_size 128 \
                --transpose \
                --data_path input_data_path \
                --lambda_flag_default \
                --nb_epoch 40000 \
                --input_dimension 128 \
                --lambda_iter_num 1000
```
```                
# exp supp: different decoders
# default decoder_type=SingleLayerDecoder; 
# others: TransformerDecoder, BilinearDecoder, NTNDecoder
python main.py --max_length 12 \
                --data_size 5000 \
                --score_type BIC \
                --reg_type LR \
                --decoder_type NTNDecoder \
                --read_data  \
                --transpose \
                --data_path input_data_path \
                --lambda_flag_default \
                --nb_epoch 20000 \
                --input_dimension 64 \
                --lambda_iter_num 1000
```
```                
# exp2: quad with RL-BIC2
# note: data has been processed, with first 3000 samples (out of 5000 sampels generated)
#       according to sample L2 norms
python main.py --max_length 10 \
                --data_size 3000 \
                --score_type BIC \
                --reg_type QR \
                --read_data  \
                --transpose \
                --data_path input_data_path \
                --lambda_flag_default \
                --nb_epoch 30000 \
                --input_dimension 64 \
                --lambda_iter_num 1000
```
```                   
# exp3: GPR
python main.py --max_length 10 \
                --data_size 1000 \
                --score_type BIC \
                --reg_type GPR \
                --read_data  \
                --normalize \
                --data_path input_data_path/exp3_10_nodes_gp/1 \
                --lambda_flag_default \
                --nb_epoch 20000 \
                --input_dimension 128 \
                --lambda_iter_num 1000
```
```   
# exp4: sachs
python main.py --max_length 11 \
                --data_size 853 \
                --score_type BIC \
                --reg_type GPR \
                --read_data  \
                --use_bias \
                --bias_initial_value -10 \
                --normalize \
                --data_path input_data_path/sachs \
                --lambda_flag_default \
                --nb_epoch 20000 \
                --input_dimension 128 \
                --lambda_iter_num 1000
```

## License

This project is licensed under the  Apache License Version 2.0 - see the [LICENSE](../../LICENSE) file for details.

## Acknowledgments

* The RL part of our approach is implemented based on a [Tensorflow implementation of neural combinatorial optimizer](https://github.com/MichelDeudon/neural-combinatorial-optimization-rl-tensorflow).  
We thank the author for releasing his implementation.
* Some evaluation metrics (such as TPR, FDR, and SHD) are computed using the codes from [NOTEARS](https://github.com/xunzheng/notears).
* We are grateful to all the authors of the benchmark methods for releasing their codes.
