# A Graph Autoencoder Approach to Causal Structure Learning
This repository contains the implementation for the paper [A Graph Autoencoder Approach to Causal Structure Learning](https://arxiv.org/abs/1911.07420) (NeurIPS 2019 Workshop), by Ignavier Ng, Shengyu Zhu, Zhitang Chen and Zhuangyan Fang.

If you find it useful, please consider citing:

```bibtex
@article{Ng2019Graph,
  title={A graph autoencoder approach to causal structure learning},
  author={Ng, Ignavier and Zhu, Shengyu and Chen, Zhitang and Fang, Zhuangyan},
  journal={arXiv preprint arXiv:1911.07420},
  year={2019}
}
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiments for Synthetic Dataset
To train the model in the paper, for scalar-valued dataset, run this command:
```bash
python main.py  --seed 1230 \
                --d 20 \
                --n 3000 \
                --degree 3 \
                --dataset_type nonlinear_1 \
                --latent_dim 1 \
                --x_dim 1
```

For vector-valued dataset, run this command:
```bash
python main.py  --seed 1230 \
                --d 20 \
                --n 3000 \
                --degree 3 \
                --dataset_type nonlinear_1 \
                --latent_dim 3 \
                --x_dim 5
```

For different graph sizes, simply modify the value of `d` in the command.


## License

This project is licensed under the  Apache License Version 2.0 - see the [LICENSE](../../LICENSE) file for details.

## Acknowledgments

* Our implementation is based on [NOTEARS](https://github.com/xunzheng/notears) and its [Tensorflow implementation](https://github.com/ignavier/notears-tensorflow).
* We are grateful to all the authors of the benchmark methods for releasing their codes.
